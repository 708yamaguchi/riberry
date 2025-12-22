#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import sys
import numpy as np
import rospy
import tf  # 追加: 座標変換用

from std_msgs.msg import String, Int32
from geometry_msgs.msg import Pose

from skrobot.model import RobotModel
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import interpolate_rotation_matrices, quaternion2matrix

# KXRインターフェース
from kxr_controller.check_ros_master import is_ros_master_local
from kxr_controller.kxr_interface import KXRROSRobotInterface

# --- ROSサービスのImport ---
try:
    from riberry_startup.srv import VisualPose, VisualPoseRequest
except ImportError:
    rospy.logwarn("VisualPose service definition not found. Vision mode will fail.")
    VisualPose = None
    VisualPoseRequest = None


class CleaningTask(object):
    def __init__(self, ri, robot_model, target_link_name):
        self.ri = ri
        self.robot_model = robot_model

        # --- TF (座標変換) の初期化 ---
        self.tf_listener = tf.TransformListener()
        # ※実機の構成に合わせてフレーム名を変更してください
        self.base_frame = "base_link"       # ロボットの基準座標
        self.camera_frame = "camera_color_optical_frame"   # カメラの座標フレーム名

        # --- ROS Control / Display用の変数 ---
        self.atom_mode = ""
        self.current_button_state = 0

        # Atom S3への表示用Publisher
        self.pub_display = rospy.Publisher("/atom_s3_additional_info", String, queue_size=1)

        # ROSトピックの購読
        rospy.Subscriber("/atom_s3_mode", String, self._cb_atom_mode)
        rospy.Subscriber("/atom_s3_button_state", Int32, self._cb_atom_button)

        # 保存用変数
        self.corners = []
        self.corner_avs = []
        self.av_seq = []
        self.times = []
        
        # IKターゲットのオフセット設定 [x, y, z]（ロボットのベース座標系）
        # たわみ補正のため、デフォルトでZ方向に+5cm履かせる
        self.ik_target_offset = np.array([0.0, 0.0, 0.05])

        # リンク取得ロジック
        if hasattr(self.robot_model, target_link_name):
            self.target_coords = getattr(self.robot_model, target_link_name)
        else:
            found_link = None
            for link in self.robot_model.link_list:
                if link.name == target_link_name:
                    found_link = link
                    break
            if found_link is None:
                raise ValueError(f"Link '{target_link_name}' not found in robot model.")
            self.target_coords = found_link

        self.link_list = []
        link = self.target_coords.parent
        while link:
            if link == self.robot_model: break
            if hasattr(link, 'joint') and link.joint:
                self.link_list.append(link)
            link = link.parent
        self.link_list.reverse()
        rospy.loginfo(f"Target Link: {self.target_coords.name}")

    # --- ディスプレイ更新用メソッド ---
    def update_display(self, text):
        self.pub_display.publish('\n' + text)

    # --- コールバック関数 ---
    def _cb_atom_mode(self, msg):
        self.atom_mode = msg.data

    def _cb_atom_button(self, msg):
        self.current_button_state = msg.data

    # --- ボタン入力待機関数 ---
    def wait_for_button_press(self, valid_buttons=[1, 2, 3]):
        self.current_button_state = 0
        while not rospy.is_shutdown():
            if self.atom_mode == "DisplayInformationMode":
                btn = self.current_button_state
                if btn in valid_buttons:
                    rospy.loginfo(f"Button {btn} accepted.")
                    self.current_button_state = 0
                    return btn
            time.sleep(0.05)
        return None

    # ==========================================================
    #  Helper: カメラ座標系 -> Base座標系 変換
    # ==========================================================
    def convert_camera_pose_to_base(self, pose_msg):
        """
        geometry_msgs/Pose (カメラ相対) を skrobot.Coordinates (Base相対) に変換する
        回転は考慮せず、位置のみを変換して返す（回転はidentity）
        """
        try:
            # 1. カメラ座標系での位置 (回転は単位行列とする)
            pos = pose_msg.position
            co_cam_to_obj = Coordinates(
                pos=[pos.x, pos.y, pos.z],
                rot=np.eye(3)
            )

            # 2. Base -> Camera の座標変換を取得
            #    tfが利用可能になるまで少し待つ
            self.tf_listener.waitForTransform(
                self.base_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(1.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )

            # skrobot用回転行列に変換 [x,y,z,w] -> [w,x,y,z] -> matrix
            rot_matrix = quaternion2matrix([rot[3], rot[0], rot[1], rot[2]])
            co_base_to_cam = Coordinates(pos=trans, rot=rot_matrix)

            # 3. 座標変換: T(Base->Obj) = T(Base->Cam) * T(Cam->Obj)
            co_base_to_obj = co_base_to_cam.copy()
            co_base_to_obj.transform(co_cam_to_obj)

            return co_base_to_obj

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {e}")
            return None

    # ==========================================================
    #  Teach Mode Entry
    # ==========================================================
    def teach_mode_entry(self):
        """Teachモードの入り口"""
        self.update_display("1clk:Manual\n2clk:Vision")

        # 1(Single) か 2(Double) が来るのを待つ
        btn = self.wait_for_button_press(valid_buttons=[1, 2])

        if btn == 1:
            rospy.loginfo("Input: 1 (Single) -> Manual Mode")
            self.teach_corners_manual()

        elif btn == 2:
            rospy.loginfo("Input: 2 (Double) -> Vision Mode")
            self.teach_corners_vision()

    # ==========================================================
    #  Method: 手動教示 (Manual Teaching)
    # ==========================================================
    def teach_corners_manual(self):
        temp_corners = []
        temp_avs = []

        rospy.loginfo("Start Manual Teaching: Servo OFF")
        self.ri.servo_off()

        for i in range(4):
            msg = f"Manual {i+1}/4\n1:Set"
            self.update_display(msg)
            rospy.loginfo(f"Waiting for Corner {i+1} (Button 1)...")

            btn = self.wait_for_button_press(valid_buttons=[1])
            if btn is None: return

            current_av = self.ri.angle_vector()
            self.robot_model.angle_vector(current_av)
            coord = self.target_coords.copy_worldcoords()

            temp_corners.append(coord)
            temp_avs.append(current_av)
            rospy.loginfo(f"Captured Corner {i+1}")

        self.update_display("Manual Done\nWait")
        rospy.loginfo("Manual Teaching finished. Servo ON.")
        self.ri.servo_on()
        time.sleep(1.0)

        self.set_corners_and_plan(temp_corners, temp_avs)

    # ==========================================================
    #  Method: 画像認識教示 (Vision Teaching)
    # ==========================================================
    def teach_corners_vision(self):
        if VisualPose is None:
            rospy.logerr("VisualPose service not imported.")
            self.update_display("Err:NoSrv\nCheckCode")
            return

        self.update_display("Vision\nCapture!")
        rospy.loginfo("Vision Capture!...")

        service_name = "/estimate_corners"
        rospy.loginfo(f"Waiting for service: {service_name}")

        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            vision_srv = rospy.ServiceProxy(service_name, VisualPose)
        except rospy.ROSException:
            rospy.logerr("Vision service is not available.")
            self.update_display("Srv Timeout\nCheck ROS")
            return

        req = VisualPoseRequest()
        req.prompt = "detect cleaning area"
        req.mode = "corners"

        rospy.loginfo(f"Calling Vision Service... with prompt {req.prompt}")
        self.update_display("Vision\nThinking...")

        # --- 画像認識した瞬間の手先姿勢を保存 ---
        current_av_seed = self.ri.angle_vector()
        self.robot_model.angle_vector(current_av_seed)
        current_rot = self.target_coords.worldrot() # 現在の回転行列

        try:
            res = vision_srv(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            self.update_display("Call Fail\nRetry")
            return

        if not res.poses or len(res.poses) != 4:
            rospy.logwarn(f"Vision returned invalid number of poses: {len(res.poses)}")
            self.update_display("Vision Fail\nRetry")
            return

        rospy.loginfo("Vision Success! Calculating IK with TF...")

        temp_corners = []
        temp_avs = []

        for i, pose_msg in enumerate(res.poses):
            # 1. 座標変換 (Camera -> Base)
            converted_coord = self.convert_camera_pose_to_base(pose_msg)

            if converted_coord is None:
                rospy.logerr(f"TF Transformation failed for Corner {i+1}")
                self.update_display("TF Fail")
                return

            # Base座標系での位置を取得
            world_pos = converted_coord.worldpos()
            # rospy.loginfo(f"Debug [Corner {i+1}]: Target World Pos (Base Frame) = {world_pos}")

            # 2. 姿勢 (Rotation) は「認識した瞬間の手先姿勢 (current_rot)」を使う
            target_coord = Coordinates(pos=world_pos, rot=current_rot)
            temp_corners.append(target_coord)

            # 3. IKを解く
            self.robot_model.angle_vector(current_av_seed)
            ik_result = self.robot_model.inverse_kinematics(
                target_coords=target_coord,
                link_list=self.link_list,
                move_target=self.target_coords,
                rotation_axis=True,
                stop=50,
                revert_if_fail=False
            )
            if ik_result is False:
                rospy.logerr(f"IK Failed for Vision Corner {i+1}")
                self.update_display(f"IK Fail\nCorner{i+1}")
                return

            temp_avs.append(self.robot_model.angle_vector())

        rospy.loginfo("All vision corners processed.")
        self.update_display("Vision Done\nWait")

        self.set_corners_and_plan(temp_corners, temp_avs)

    # ==========================================================
    #  Method: 共通処理
    # ==========================================================
    def set_corners_and_plan(self, corners, avs):
        self.corners = corners
        self.corner_avs = avs

        rospy.logwarn(f"Applying IK Target Offset: {self.ik_target_offset}")

        if len(self.corners) != 4:
            rospy.logwarn("No corners taught yet.")
            return

        waypoints, seed_avs = self.generate_zigzag_trajectory(
            self.corners,
            self.corner_avs,
            step_width=0.01
        )

        seq = self.solve_full_ik(waypoints, seed_avs)

        if seq is None:
            rospy.logerr("Planning failed.")
            self.update_display("IK Fail\n1:Retry")
            self.av_seq = []
            return

        self.av_seq = seq
        self.times = [1.0] * len(self.av_seq)
        rospy.loginfo("Plan ready.")


    def generate_zigzag_trajectory(self, corners, corner_avs, step_width=0.01):
        p1, p2 = corners[0].worldpos(), corners[1].worldpos()
        p3, p4 = corners[2].worldpos(), corners[3].worldpos()
        r1, r2 = corners[0].worldrot(), corners[1].worldrot()
        r3, r4 = corners[2].worldrot(), corners[3].worldrot()
        av1, av2 = corner_avs[0], corner_avs[1]
        av3, av4 = corner_avs[2], corner_avs[3]

        vec_advance = p4 - p1
        len_advance = np.linalg.norm(vec_advance)

        n_steps = int(len_advance / step_width)
        if n_steps < 1: n_steps = 1

        waypoints = []
        seed_avs = []

        for i in range(n_steps + 1):
            ratio_adv = float(i) / n_steps
            base_left = p1 + (p4 - p1) * ratio_adv
            base_right = p2 + (p3 - p2) * ratio_adv
            left_point = base_left + self.ik_target_offset
            right_point = base_right + self.ik_target_offset
            left_rot = interpolate_rotation_matrices(ratio_adv, r1, r4)
            right_rot = interpolate_rotation_matrices(ratio_adv, r2, r3)
            left_av = av1 + (av4 - av1) * ratio_adv
            right_av = av2 + (av3 - av2) * ratio_adv

            if i % 2 == 0:
                waypoints.append(Coordinates(pos=left_point, rot=left_rot))
                seed_avs.append(left_av)
                waypoints.append(Coordinates(pos=right_point, rot=right_rot))
                seed_avs.append(right_av)
            else:
                waypoints.append(Coordinates(pos=right_point, rot=right_rot))
                seed_avs.append(right_av)
                waypoints.append(Coordinates(pos=left_point, rot=left_rot))
                seed_avs.append(left_av)
        return waypoints, seed_avs

    def solve_full_ik(self, waypoints, seed_avs):
        rospy.loginfo("Solving IK...")
        self.update_display("Plan\nWait")

        av_sequence = []
        error_tolerance = 0.02

        for idx, wp in enumerate(waypoints):
            target_seed = seed_avs[idx]
            self.robot_model.angle_vector(target_seed)

            result = self.robot_model.inverse_kinematics(
                target_coords=wp,
                link_list=self.link_list,
                move_target=self.target_coords,
                rotation_axis=True,
                stop=50,
                revert_if_fail=False
            )

            dist_err = np.linalg.norm(self.target_coords.worldpos() - wp.worldpos())
            is_success = (result is not False) and (result is not None)

            if is_success:
                av_sequence.append(self.robot_model.angle_vector())
            else:
                if dist_err < error_tolerance:
                    av_sequence.append(self.robot_model.angle_vector())
                else:
                    rospy.logwarn(f"Point {idx}: IK Failed. Error: {dist_err*1000:.1f}mm")
                    return None

        rospy.loginfo("IK Solved successfully.")
        return av_sequence

    def execute_motion(self):
        if not self.av_seq:
            rospy.logwarn("No plan available.")
            self.update_display("NoPlan\n1:Teach")
            time.sleep(2.0)
            return

        rospy.loginfo("Executing motion...")
        self.update_display("Playing\n...")

        self.ri.servo_on()
        self.ri.angle_vector_sequence(self.av_seq, times=self.times)
        self.ri.wait_interpolation()
        rospy.loginfo("Motion Finished.")

    def run(self):
        rospy.loginfo("Task Node Ready.")

        while not rospy.is_shutdown():
            # メニュー: 1:Teach, 2:Play, 3:Free
            menu_msg = "1:Teach\n2:Play\n3:Free"
            self.update_display(menu_msg)

            btn = self.wait_for_button_press(valid_buttons=[1, 2, 3])

            if btn == 1:
                # Teach Mode
                self.teach_mode_entry()

            elif btn == 2:
                # Play Mode
                self.execute_motion()

            elif btn == 3:
                # Free Mode
                rospy.loginfo("Servo OFF (Free Mode)")
                self.update_display("Free\nMode")
                self.ri.servo_off()
                time.sleep(3.0)

def main():
    parser = argparse.ArgumentParser(description="Cleaning Task with KXR")
    parser.add_argument("--namespace", type=str, help="Specify the ROS namespace", default="")
    args = parser.parse_args()

    rospy.init_node("kxr_cleaning_task", anonymous=True)

    if is_ros_master_local() is True:
        robot_description = args.namespace + "/robot_description"
    else:
        from kxr_models.download_urdf import download_urdf_mesh_files
        download_urdf_mesh_files(args.namespace)
        robot_description = args.namespace + "/robot_description_viz"

    robot_model = RobotModel()
    from skrobot.utils.urdf import no_mesh_load_mode
    with no_mesh_load_mode():
        robot_model.load_urdf_from_robot_description(robot_description)

    ri = KXRROSRobotInterface(
        robot_model, namespace=args.namespace, controller_timeout=60.0
    )

    try:
        task = CleaningTask(ri, robot_model, target_link_name='module5_base_link')
        task.run()

    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
