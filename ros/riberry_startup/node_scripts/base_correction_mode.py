#!/usr/bin/env python3

from enum import Enum
import json
import os

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform
from kxr_controller.kxr_interface import KXRROSRobotInterface
from kxr_controller.msg import PressureControl
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from skrobot.interfaces.ros.tf_utils import tf_pose_to_coords
from skrobot.model import RobotModel
from skrobot.utils.urdf import no_mesh_load_mode
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from std_msgs.msg import String

from riberry.com.base import PacketType
from riberry.filecheck_utils import get_cache_dir
from riberry.mode import Mode
from riberry.select_list import SelectList
from riberry.utils.ros.namespace import get_base_namespace


class State(Enum):
    WAIT = 0
    RECORD_POSE = 1
    CHECK_POSE = 2
    SELECT_MOTION = 3


class ImageManager:
    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.color_camera_info = None
        self.depth_camera_info = None
        self.color_image = None
        self.depth_image = None
        self.subs = [Subscriber("/camera/color/image_raw", Image),
                     Subscriber("/camera/aligned_depth_to_color/image_raw", Image),
                     Subscriber("/camera/color/camera_info", CameraInfo),
                     Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo)]
        self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=30, slop=0.1)
        self.sync.registerCallback(self.image_callback)

    def image_callback(self, color_img_msg, depth_img_msg, color_info_msg, depth_info_msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(
                color_img_msg, color_img_msg.encoding)
            self.depth_image = self.bridge.imgmsg_to_cv2(
                depth_img_msg, depth_img_msg.encoding)
            if self.color_camera_info is None:
                self.color_camera_info = color_info_msg
                self.depth_camera_info = depth_info_msg
                rospy.loginfo("Camera intrinsics received.")
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def save_images(self, base_path):
        color_img_path = f"{base_path}_color.png"
        cv2.imwrite(color_img_path, cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))
        depth_img_path = f"{base_path}_depth.png"
        cv2.imwrite(depth_img_path, self.depth_image)
        return color_img_path, depth_img_path

    def load_images(self, base_path):
        color_img_path = f"{base_path}_color.png"
        depth_img_path = f"{base_path}_depth.png"
        color_image = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        return color_image, depth_image


def check_base_transform(R, t):
    success = True
    # 配置修正してもタスクが実行できない条件を考える。
    # 1. 吸着面と垂直な方向(Z方向)に移動する必要がある場合。直接教示の場合は5cm以上なら実行不可能とする。
    z_translation_threshold = 0.05  # [m]
    if t[2] < z_translation_threshold:
        x = t[0]
        y = t[1]
        rospy.loginfo(f"Correct robot position by moving {-1 * t[:2]} [m]")  # 修正量
    else:
        rospy.loginfo("Error! The robot needs to move in a direction perpendicular to the contact surface.")
        success = False
    # 2. 吸着面と垂直な方向以外の2方向(X, Y方向)周りに回転する必要がある場合。直接教示の場合は10度以上なら実行不可能とする。
    rot_threshold = 0.1
    condition1 = (
        np.abs(R[2, 0]) < rot_threshold and
        np.abs(R[2, 1]) < rot_threshold and
        np.abs(R[0, 2]) < rot_threshold and
        np.abs(R[1, 2]) < rot_threshold and
        np.abs(R[2, 2] - 1) < rot_threshold
    )
    condition2 = (
        np.abs(R[0, 0] - R[1, 1]) < rot_threshold and
        np.abs(R[1, 0] + R[0, 1]) < rot_threshold
    )
    if condition1 and condition2:
        cos = 0.5 * (R[0, 0] + R[1, 1])
        sin = 0.5 * (R[1, 0] - R[0, 1])
        theta_rad = np.arctan2(sin, cos)  # [rad}
    else:
        theta_rad = False
    if theta_rad:
        rospy.loginfo(f"Correct robot pose by rotating {-1 * np.rad2deg(theta_rad):.2f} [deg].\n")
    else:
        rospy.loginfo("The robot needs to rotate on non-contact surface.\n")
        success = False
    # Return
    if success:
        return (x, y, theta_rad)
    else:
        return None

class BaseCorrectionMode(Mode):
    def __init__(self):
        super().__init__()
        # Create timer callback first to call self.send_string() even if self.ri cannot be generated
        self.init_finished = False
        self.state = State.WAIT
        rospy.Timer(rospy.Duration(1), self.timer_callback)
        # Create robot model to control pressure
        robot_model = RobotModel()
        namespace = get_base_namespace()
        with no_mesh_load_mode():
            robot_model.load_urdf_from_robot_description(
                namespace + "/robot_description_viz")
        self.ri = KXRROSRobotInterface(
            robot_model, namespace=namespace, controller_timeout=60.0
        )

        rospy.Subscriber(
            "atom_s3_button_state", Int32, callback=self.button_cb, queue_size=1
        )

        self.image_manager = ImageManager()
        self.pubs = [rospy.Publisher("~color/1", Image, queue_size=1),
                     rospy.Publisher("~depth/1", Image, queue_size=1),
                     rospy.Publisher("~color/2", Image, queue_size=1),
                     rospy.Publisher("~depth/2", Image, queue_size=1),
                     rospy.Publisher("~color/camera_info", CameraInfo, queue_size=1),
                     rospy.Publisher("~depth/camera_info", CameraInfo, queue_size=1),
                     rospy.Publisher("~mask_prompt", String, queue_size=1)]

        self.setup_pressure_control()
        self.load_play_list()
        self.selected_json_path = None
        self.additional_msg = "Select Motion first"

        self.init_finished = True

    # Copied from pressure_control_mode.py
    def setup_pressure_control(self):
        # Pressure control
        self.pressure_control_state = {}
        rospy.Subscriber(
            "fullbody_controller/pressure_control_interface/state",
            PressureControl,
            callback=self.pressure_control_cb,
            queue_size=1,
        )
        self.air_work_list = SelectList()
        self.delimiter = ":"

        # Read pressure
        self.pressures = {}
        for idx in range(38, 66):
            rospy.Subscriber(
                f"fullbody_controller/pressure/{idx}",
                Float32,
                self.read_pressure,
                callback_args=idx,
            )
            self.pressures[idx] = None

    # Copied from teaching_mode.py
    def load_play_list(self):
        self.play_list = SelectList()
        self.play_list.set_extract_pattern(r"teaching_(.*?)\.json")
        self.json_dir = get_cache_dir()

        def get_teaching_files(json_dir):
            """Loads all teaching JSON files from the configured directory."""
            teaching_files = [
                os.path.join(json_dir, file)
                for file in os.listdir(json_dir)
                if file.startswith("teaching_") and file.endswith(".json")
            ]
            teaching_files_sorted = sorted(
                teaching_files,
                key=lambda f: os.path.getctime(f)
            )
            return teaching_files_sorted

        for file in get_teaching_files(self.json_dir):
            self.play_list.add_option(file)

    def save_base_correction_information(self):
        # Save images
        base_path = os.path.splitext(self.selected_json_path)[0]
        color_img_path, depth_img_path = self.image_manager.save_images(base_path)
        # Save joint states (Copied from motion_manager.py)
        # Average multiple angle vectors to reduce the noise
        # of the servo motor's potentiometer
        joint_states = {}
        avs = []
        for _ in range(5):
            avs.append(self.ri.angle_vector())
        av_average = np.mean(avs, axis=0)
        for j, a in zip(self.ri.robot.joint_names, av_average):
            joint_states[str(j)] = float(a)
        # Update json
        with open(self.selected_json_path) as f:
            json_data = json.load(f)
            if "base_correction" not in json_data:
                json_data["base_correction"] = {}
            json_data["base_correction"]["color_image"] = color_img_path
            json_data["base_correction"]["depth_image"] = depth_img_path
            json_data["base_correction"]["color_camera_info"] = {}
            json_data["base_correction"]["color_camera_info"]["height"] = self.image_manager.color_camera_info.height
            json_data["base_correction"]["color_camera_info"]["width"] = self.image_manager.color_camera_info.width
            json_data["base_correction"]["color_camera_info"]["K"] = self.image_manager.color_camera_info.K
            json_data["base_correction"]["color_camera_info"]["D"] = self.image_manager.color_camera_info.D
            json_data["base_correction"]["depth_camera_info"] = {}
            json_data["base_correction"]["depth_camera_info"]["height"] = self.image_manager.depth_camera_info.height
            json_data["base_correction"]["depth_camera_info"]["width"] = self.image_manager.depth_camera_info.width
            json_data["base_correction"]["depth_camera_info"]["K"] = self.image_manager.depth_camera_info.K
            json_data["base_correction"]["depth_camera_info"]["D"] = self.image_manager.depth_camera_info.D
            json_data["base_correction"]["joint_states"] = joint_states
        with open(self.selected_json_path, 'w') as f:
            f.write(json.dumps(json_data, indent=4, separators=(",", ": ")))

    def save_mask_prompt(self, mask_prompt):
        with open(self.selected_json_path) as f:
            json_data = json.load(f)
            if "base_correction" not in json_data:
                json_data["base_correction"] = {}
            json_data["base_correction"]["mask_prompt"] = mask_prompt
        with open(self.selected_json_path, 'w') as f:
            f.write(json.dumps(json_data, indent=4, separators=(",", ": ")))

    def button_cb(self, msg):
        """
        When AtomS3 is BaseCorrectionMode and single-click pressed,
        toggle pressure control.
        """
        if self.mode != "BaseCorrectionMode":
            return
        if self.state == State.WAIT:
            if msg.data == 1:
                self.state = State.RECORD_POSE
            elif msg.data == 2:
                self.state = State.CHECK_POSE
            elif msg.data == 3:
                self.state = State.SELECT_MOTION
        elif self.state == State.RECORD_POSE:
            if msg.data == 1:
                if self.selected_json_path is None:
                    self.additional_msg = "Error: select motion first"
                    rospy.logerr(self.additional_msg)
                    self.ri.servo_off()
                    self.state = State.WAIT
                    return
                # Save images and write their path to json
                self.ri.servo_on()
                rospy.sleep(3)
                self.save_base_correction_information()
                self.additional_msg = "Update base correction info"
                rospy.loginfo(f"{self.additional_msg} to {self.selected_json_path}")
            elif msg.data == 2:
                if self.selected_json_path is None:
                    self.additional_msg = "Error: select motion first"
                    rospy.logerr(self.additional_msg)
                    self.ri.servo_off()
                    self.state = State.WAIT
                    return
                # Teach prompt for image mask
                try:
                    self.additional_msg = "Wait for mask prompt topic"
                    rospy.loginfo(self.additional_msg)
                    mask_prompt_msg = rospy.wait_for_message(
                        'chatbot_node/output', String, timeout=15)
                    self.save_mask_prompt(mask_prompt_msg.data)
                    self.additional_msg = f"Update mask prompt: {mask_prompt_msg.data}"
                    rospy.loginfo(f"{self.additional_msg} to {self.selected_json_path}")
                except rospy.exceptions.ROSException:
                    self.additional_msg = "Mask prompt topic has not come"
                    rospy.logerr(self.additional_msg)
            elif msg.data == 3:
                self.ri.servo_off()
                self.state = State.WAIT
                self.additional_msg = "Finish record pose"
        elif self.state == State.CHECK_POSE:
            # Check pose
            if msg.data == 1:
                if self.selected_json_path is None:
                    self.additional_msg = "Error: select motion first"
                    rospy.logerr(self.additional_msg)
                    self.state = State.WAIT
                    return
                # Get current image
                self.ri.servo_on()
                with open(self.selected_json_path) as f:
                    json_data = json.load(f)
                    check_av = list(json_data["base_correction"]["joint_states"].values())
                self.ri.angle_vector(check_av, 3)
                self.ri.wait_interpolation()
                rospy.sleep(3)
                color_image2 = self.image_manager.color_image
                depth_image2 = self.image_manager.depth_image
                color_image2_msg = self.image_manager.bridge.cv2_to_imgmsg(
                    color_image2, encoding="rgb8")
                depth_image2_msg = self.image_manager.bridge.cv2_to_imgmsg(
                    depth_image2, encoding="passthrough")
                # Load previous image
                base_path = os.path.splitext(self.selected_json_path)[0]
                color_image1, depth_image1 = self.image_manager.load_images(base_path)
                color_image1_msg = self.image_manager.bridge.cv2_to_imgmsg(
                    color_image1, encoding="rgb8")
                depth_image1_msg = self.image_manager.bridge.cv2_to_imgmsg(
                    depth_image1, encoding="passthrough")
                # Load camera info
                with open(self.selected_json_path) as f:
                    json_data = json.load(f)
                    color_camera_info = CameraInfo()
                    color_camera_info.height = json_data["base_correction"]["color_camera_info"]["height"]
                    color_camera_info.width = json_data["base_correction"]["color_camera_info"]["width"]
                    color_camera_info.K = json_data["base_correction"]["color_camera_info"]["K"]
                    color_camera_info.D = json_data["base_correction"]["color_camera_info"]["D"]
                    depth_camera_info = CameraInfo()
                    depth_camera_info.height = json_data["base_correction"]["depth_camera_info"]["height"]
                    depth_camera_info.width = json_data["base_correction"]["depth_camera_info"]["width"]
                    depth_camera_info.K = json_data["base_correction"]["depth_camera_info"]["K"]
                    depth_camera_info.D = json_data["base_correction"]["depth_camera_info"]["D"]
                # Set current timestamp
                stamp = rospy.Time.now()
                color_image1_msg.header.stamp = stamp
                depth_image1_msg.header.stamp = stamp
                color_image2_msg.header.stamp = stamp
                depth_image2_msg.header.stamp = stamp
                color_camera_info.header.stamp = stamp
                depth_camera_info.header.stamp = stamp
                # Publish images
                self.pubs[0].publish(color_image1_msg)
                self.pubs[1].publish(depth_image1_msg)
                self.pubs[2].publish(color_image2_msg)
                self.pubs[3].publish(depth_image2_msg)
                self.pubs[4].publish(color_camera_info)
                self.pubs[5].publish(depth_camera_info)
                self.additional_msg = "Publish image and camera info"
                # Publish mask prompt if found
                if "mask_prompt" in json_data["base_correction"]:
                    self.pubs[6].publish(String(data=json_data["base_correction"]["mask_prompt"]))
                    self.additional_msg += ", and mask prompt"
                rospy.loginfo(self.additional_msg)
                # Receive and check base transform
                try:
                    self.additional_msg = "Wait for base transform topic"
                    transform_msg = rospy.wait_for_message(
                        'base_correction_mode/transform', Transform, timeout=60)
                    transform_coords = tf_pose_to_coords(transform_msg)
                    R = transform_coords.rotation
                    t = transform_coords.translation
                    # Empry coords means match failure
                    if np.array_equal(R, np.eye(3)) and np.array_equal(t, np.zeros(3)):
                        self.additional_msg = "Feature matching failed"
                        rospy.logerr(self.additional_msg)
                        return
                    result = check_base_transform(R, t)
                    if result:
                        x, y, theta = result
                        self.additional_msg = "Correct pose\n"
                        self.additional_msg += f"x: {int(x*100)}[cm]\n"
                        self.additional_msg += f"y: {int(y*100)}[cm]\n"
                        self.additional_msg += f"rot: {int(np.rad2deg(theta))}[deg]\n"
                    else:
                        self.additional_msg = "Move object instead of robot"
                except rospy.exceptions.ROSException:
                    self.additional_msg = "Base transform topic has not come"
                    rospy.logerr(self.additional_msg)
            elif msg.data == 2:  # Omit increment idx
                selected = self.air_work_list.selected_option()
                idx = int(selected.split(self.delimiter)[0])
                rospy.loginfo(
                    "AtomS3 is BaseCorrectionMode and double-click-pressed."
                    + f" Toggle ID {idx} pressure control."
                )
                self.toggle_pressure_control(idx)
            elif msg.data == 3:
                self.ri.servo_off()
                self.state = State.WAIT
                self.additional_msg = "Finish check pose"
        elif self.state == State.SELECT_MOTION:
            if len(self.play_list.options) == 0:
                self.state = State.WAIT
                self.additional_msg = "Motion data is not found"
                return
            if msg.data == 1:
                self.play_list.increment_index()
            if msg.data == 2:
                idx = self.play_list.get_index()
                if idx > 0:
                    self.play_list.set_index(idx - 1)
            elif msg.data == 3:
                # Select motion
                self.selected_json_path = self.play_list.selected_option()
                self.additional_msg = f"Select: {self.selected_json_path.split('/')[-1]}"
                self.state = State.WAIT

    def pressure_control_cb(self, msg):
        self.pressure_control_state[f"{msg.board_idx}"] = msg

    def toggle_pressure_control(self, idx):
        """
        Toggle release status
        """
        if self.ri is None:
            rospy.logwarn("KXRROSRobotInterface instance is not created.")
            return
        state = self.pressure_control_state[f"{idx}"]
        # toggle release state
        if state.release_duration == 0:
            release_duration = 2  # release air
        elif state.release_duration > 0:
            release_duration = 0  # start air work
        # Set pressures
        if state.trigger_pressure == 0 and state.target_pressure == 0:
            trigger_pressure = -10
            target_pressure = -30
        else:
            trigger_pressure = state.trigger_pressure
            target_pressure = state.target_pressure
        # Send goal
        self.ri.send_pressure_control(
            board_idx=int(idx),
            trigger_pressure=trigger_pressure,
            target_pressure=target_pressure,
            release_duration=release_duration
        )

    def read_pressure(self, msg, idx):
        self.pressures[idx] = msg.data

    def timer_callback(self, event):
        self.send_string()

    def send_string(self):
        if self.mode != "BaseCorrectionMode":
            return
        if self.init_finished is False:
            sent_str = chr(PacketType.BASE_CORRECTION_MODE)
            sent_str += "Base Correction Mode\n\n"
            sent_str += "Wait until KXR interface is available."
            self.write(sent_str)
            return

        sent_str = chr(PacketType.BASE_CORRECTION_MODE)
        sent_str += "Base Corretion\n\n"  # Intentional typo to shorten word
        if self.state == State.WAIT:
            sent_str += "1 Record pose\n"
            sent_str += "2 Check pose\n"
            sent_str += "3 Select motio\n"
        elif self.state == State.RECORD_POSE:
            sent_str += "1 Save images\n"
            sent_str += "2 Tell prompt\n"
            sent_str += "3 Servo off\n"
        # The same visualization as PressureControlMode
        elif self.state == State.CHECK_POSE:
            air_work_index = self.air_work_list.get_index()
            self.air_work_list.remove_all_options()
            for idx, value in self.pressures.items():
                if value is None:
                    continue
                else:
                    air_work_str = f"{idx}{self.delimiter} {value:.1f}"
                    if f"{idx}" in self.pressure_control_state:
                        release_duration = self.pressure_control_state[f"{idx}"].release_duration
                        if release_duration == 0:
                            air_work_str += "\x1b[32m ON\x1b[39m"
                        elif release_duration > 0:
                            air_work_str += "\x1b[31m OFF\x1b[39m"
                    if air_work_str.count(self.delimiter) != 1:
                        rospy.logerr(f"The number of delimiter {self.delimiter} must be 1")
                    self.air_work_list.add_option(air_work_str)
            self.air_work_list.set_index(air_work_index)
            sent_str += self.air_work_list.string_options(3)
            sent_str += "1 Check\n"
            sent_str += "2 Vacuum\n"
            sent_str += "3 Servo off\n"
        elif self.state == State.SELECT_MOTION:
            sent_str += "1 Next\n"
            sent_str += "2 Back\n"
            sent_str += "3 Select\n\n"
            sent_str += self.play_list.string_options(5)
        # Send message on AtomS3 LCD
        self.write(sent_str + "\n" + self.additional_msg)


if __name__ == "__main__":
    rospy.init_node("base_correction_mode")
    BaseCorrectionMode()
    rospy.spin()
