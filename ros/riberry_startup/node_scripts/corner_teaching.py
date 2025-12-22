#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import rospy
import tf

from std_msgs.msg import String, Int32
from geometry_msgs.msg import Pose

from skrobot.model import RobotModel
from skrobot.coordinates import Coordinates, make_cascoords
from skrobot.coordinates.math import interpolate_rotation_matrices, quaternion2matrix

from kxr_controller.check_ros_master import is_ros_master_local
from kxr_controller.kxr_interface import KXRROSRobotInterface

try:
    from riberry_startup.srv import VisualPose, VisualPoseRequest
except ImportError:
    rospy.logwarn("VisualPose service definition not found.")
    VisualPose = None


class CleaningTask(object):
    def __init__(self, ri, robot_model):
        self.ri = ri
        self.robot_model = robot_model

        # --- TF初期化 ---
        self.tf_listener = tf.TransformListener()
        self.base_frame = "base_link"
        self.camera_frame = "camera_color_optical_frame"

        # --- ROS通信 ---
        self.pub_display = rospy.Publisher("/atom_s3_additional_info", String, queue_size=1)
        rospy.Subscriber("/atom_s3_mode", String, self._cb_atom_mode)
        rospy.Subscriber("/atom_s3_button_state", Int32, self._cb_atom_button)

        self.atom_mode = ""
        self.current_button_state = 0

        # --- 保存データ ---
        self.corners = []
        self.corner_avs = []
        self.av_seq = []
        self.times = []

        # --- 設定値 ---
        self.error_tolerance = 0.02  # IK許容誤差[m]

        # [Vision Mode Only] 画像認識時のみ使用するパラメータ
        # たわみ補正等のためのZ方向オフセット [m] (Visionモードのみ適用)
        self.vision_target_offset = np.array([0.0, 0.0, 0.03])
        # 認識領域の拡大・縮小マージン [m] (プラスで拡大、マイナスで縮小)
        self.vision_area_margin = 0.00

        # --- リンク設定 ---
        self._setup_kinematics()

    def _setup_kinematics(self):
        """ハードコードされたリンクを用いて手先座標系とIKチェーンを設定"""
        
        # 1. 物理リンク名のハードコード
        target_link_name = 'module5_base_link'
        
        # 2. 物理リンクオブジェクトの取得
        if hasattr(self.robot_model, target_link_name):
            physical_link = getattr(self.robot_model, target_link_name)
        else:
            # 属性としてアクセスできない場合の検索
            found = next((l for l in self.robot_model.link_list if l.name == target_link_name), None)
            if not found:
                raise ValueError(f"Link '{target_link_name}' not found.")
            physical_link = found

        # 3. 手先座標系 (self.end_coords) の作成
        # make_cascoordsを使って、物理リンクを親とする座標系を作成
        self.end_coords = make_cascoords(parent=physical_link)
        
        # 指定した物理リンクのローカル座標系(wrt='local')でエンドエフェクタ位置を指定
        ee_offset = (0.0, 0.0, 0.085)
        self.end_coords.translate(ee_offset, wrt="local")

        # 4. IK計算用のリンクチェーン生成
        self.link_list = []
        link = physical_link.parent
        while link and link != self.robot_model:
            if hasattr(link, 'joint') and link.joint:
                self.link_list.append(link)
            link = link.parent
        self.link_list.reverse()

        rospy.loginfo(f"Target Physical Link: {physical_link.name}")
        rospy.loginfo(f"End Effector set with local offset {ee_offset}")

    # --- ディスプレイ更新 ---
    def update_display(self, text):
        self.pub_display.publish('\n' + text)

    # --- コールバック ---
    def _cb_atom_mode(self, msg):
        self.atom_mode = msg.data

    def _cb_atom_button(self, msg):
        self.current_button_state = msg.data

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
    #  Helper: 共通処理 (TF変換 / IK計算)
    # ==========================================================
    def get_base_coords_from_camera(self, pose_msg):
        """カメラ座標系のPoseをBase座標系のCoordinatesに変換して返す"""
        try:
            self.tf_listener.waitForTransform(self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {e}")
            return None

        # Camera -> Base の変換行列
        rot_matrix = quaternion2matrix([rot[3], rot[0], rot[1], rot[2]])
        co_base_to_cam = Coordinates(pos=trans, rot=rot_matrix)

        # Camera -> Object (回転は無視して単位行列)
        co_cam_to_obj = Coordinates(
            pos=[pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
            rot=np.eye(3)
        )

        # Base -> Object
        co_base_to_obj = co_base_to_cam.copy()
        co_base_to_obj.transform(co_cam_to_obj)
        return co_base_to_obj

    def _solve_ik(self, target_coords, seed_av):
        """IKを解く共通関数。失敗時は許容誤差内なら採用、だめならNoneを返す"""
        self.robot_model.angle_vector(seed_av)

        result = self.robot_model.inverse_kinematics(
            target_coords=target_coords,
            link_list=self.link_list,
            move_target=self.end_coords,
            rotation_axis=True,
            stop=50,
            revert_if_fail=False
        )

        dist_err = np.linalg.norm(self.end_coords.worldpos() - target_coords.worldpos())
        is_success = (result is not False) and (result is not None)

        if is_success:
            return self.robot_model.angle_vector()
        elif dist_err < self.error_tolerance:
            # 厳密解ではないが許容範囲内
            return self.robot_model.angle_vector()
        else:
            return None

    # ==========================================================
    #  Teach Mode Entry
    # ==========================================================
    def teach_mode_entry(self):
        self.update_display("1clk:Manual\n2clk:Vision")
        btn = self.wait_for_button_press(valid_buttons=[1, 2])

        if btn == 1:
            rospy.loginfo("Input: 1 -> Manual Mode")
            self.teach_corners_manual()
        elif btn == 2:
            rospy.loginfo("Input: 2 -> Vision Mode")
            self.teach_corners_vision()

    # ==========================================================
    #  Manual Teaching
    # ==========================================================
    def teach_corners_manual(self):
        """手動教示: 終了時にサーボONせず、脱力状態を維持する"""
        temp_corners = []
        temp_avs = []

        rospy.loginfo("Start Manual Teaching: Servo OFF")
        self.ri.servo_off()

        for i in range(4):
            self.update_display(f"Manual {i+1}/4\n1:Set")
            rospy.loginfo(f"Waiting for Corner {i+1}...")

            if self.wait_for_button_press(valid_buttons=[1]) is None:
                return

            current_av = self.ri.angle_vector()
            self.robot_model.angle_vector(current_av)

            # マニュアル時は現在値をそのまま使う（オフセットなし）
            temp_corners.append(self.end_coords.copy_worldcoords())
            temp_avs.append(current_av)
            rospy.loginfo(f"Captured Corner {i+1}")

        self.update_display("Manual Done\nWait")
        rospy.loginfo("Manual Teaching finished. Keeping Servo OFF.")

        self.set_corners_and_plan(temp_corners, temp_avs)

    # ==========================================================
    #  Vision Teaching
    # ==========================================================
    def teach_corners_vision(self):
        if VisualPose is None:
            rospy.logerr("VisualPose service not imported.")
            self.update_display("Err:NoSrv\nCheckCode")
            return

        # self.update_display("Vision\nCapture!")
        # rospy.loginfo("Vision Capture!...")
        rospy.loginfo("Vision Mode: Servo ON. Starting countdown.")
        self.ri.servo_on()
        # 3秒カウントダウン (3 -> 2 -> 1)
        for i in range(3, 0, -1):
            msg = f"Vision\n{i}..."
            self.update_display(msg)
            rospy.loginfo(f"Countdown: {i}")
            time.sleep(1.0)
        # 撮影タイミング表示
        self.update_display("Vision\nCapture!")
        rospy.loginfo("Capturing now...")

        try:
            rospy.wait_for_service("/estimate_corners", timeout=5.0)
            vision_srv = rospy.ServiceProxy("/estimate_corners", VisualPose)
        except rospy.ROSException:
            rospy.logerr("Vision service is not available.")
            self.update_display("Srv Timeout\nCheck ROS")
            return

        # req = VisualPoseRequest(prompt="detect cleaning area", mode="corners")
        req = VisualPoseRequest(prompt="Green tape area", mode="corners")
        rospy.loginfo(f"Calling Vision Service... prompt={req.prompt}")
        self.update_display("Vision\nThinking...")

        # 認識時の姿勢（回転）を基準にするため保存
        seed_av = self.ri.angle_vector()
        self.robot_model.angle_vector(seed_av)
        base_rot = self.end_coords.worldrot()

        try:
            res = vision_srv(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            self.update_display("Call Fail\nRetry")
            return

        if not res.poses or len(res.poses) != 4:
            rospy.logwarn(f"Invalid poses count: {len(res.poses)}")
            self.update_display("Vision Fail\nRetry")
            return

        rospy.loginfo("Vision Success! Applying offsets & Calculating IK...")

        # 1. まず4点のBase座標を取得
        base_coords_list = []
        for pose_msg in res.poses:
            c = self.get_base_coords_from_camera(pose_msg)
            if c is None:
                self.update_display("TF Fail")
                return
            base_coords_list.append(c)

        # 2. 中心点（重心）を計算
        positions = np.array([c.worldpos() for c in base_coords_list])
        center_pos = np.mean(positions, axis=0)

        temp_corners = []
        temp_avs = []

        # 3. 拡大/縮小 と Zオフセットの適用、IK計算
        for i, raw_coord in enumerate(base_coords_list):
            pos = raw_coord.worldpos()

            # --- 領域サイズの拡大・縮小 (Vision Modeのみ) ---
            # 中心から各頂点へのベクトル
            vec = pos - center_pos
            vec_len = np.linalg.norm(vec)
            if vec_len > 1e-6:
                # ベクトル方向にマージン分だけ移動させる
                direction = vec / vec_len
                pos = pos + direction * self.vision_area_margin

            # --- Z方向オフセット (Vision Modeのみ) ---
            pos += self.vision_target_offset

            # ターゲット作成
            target = Coordinates(pos=pos, rot=base_rot)

            # IK計算
            av = self._solve_ik(target, seed_av)
            if av is None:
                rospy.logerr(f"IK Failed for Vision Corner {i+1}")
                self.update_display(f"IK Fail\nCorner{i+1}")
                return

            temp_corners.append(target)
            temp_avs.append(av)

        rospy.loginfo("All vision corners processed.")
        self.update_display("Vision Done\nWait")
        self.set_corners_and_plan(temp_corners, temp_avs)

    # ==========================================================
    #  Planning & Execution
    # ==========================================================
    def set_corners_and_plan(self, corners, avs):
        self.corners = corners
        self.corner_avs = avs

        if len(self.corners) != 4:
            return

        # 軌道生成（座標補間のみ）
        waypoints, seed_avs = self.generate_zigzag_trajectory(self.corners, self.corner_avs)

        # IKで全関節角度列を生成
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
        """
        コーナー座標間を補間してジグザグ軌道を生成する。
        Manual/Visionですでにオフセット処理済みの座標が渡されるため、
        ここでは単純な補間のみを行う。
        """
        p = [c.worldpos() for c in corners]
        r = [c.worldrot() for c in corners]
        av = corner_avs

        len_advance = np.linalg.norm(p[3] - p[0])
        n_steps = max(1, int(len_advance / step_width))

        waypoints = []
        seed_avs = []

        for i in range(n_steps + 1):
            ratio = float(i) / n_steps
            # 位置の補間
            base_left = p[0] + (p[3] - p[0]) * ratio
            base_right = p[1] + (p[2] - p[1]) * ratio

            # 回転とAVの補間
            rot_left = interpolate_rotation_matrices(ratio, r[0], r[3])
            rot_right = interpolate_rotation_matrices(ratio, r[1], r[2])
            av_left = av[0] + (av[3] - av[0]) * ratio
            av_right = av[1] + (av[2] - av[1]) * ratio

            # 往復動作の生成 (オフセット加算は行わない)
            wp_l = Coordinates(pos=base_left, rot=rot_left)
            wp_r = Coordinates(pos=base_right, rot=rot_right)

            if i % 2 == 0:
                waypoints.extend([wp_l, wp_r])
                seed_avs.extend([av_left, av_right])
            else:
                waypoints.extend([wp_r, wp_l])
                seed_avs.extend([av_right, av_left])

        return waypoints, seed_avs

    def solve_full_ik(self, waypoints, seed_avs):
        rospy.loginfo("Solving IK for full trajectory...")
        self.update_display("Plan\nWait")

        av_sequence = []
        for i, (wp, seed) in enumerate(zip(waypoints, seed_avs)):
            av = self._solve_ik(wp, seed)
            if av is None:
                dist = np.linalg.norm(self.end_coords.worldpos() - wp.worldpos())
                rospy.logwarn(f"Point {i}: IK Failed. Error: {dist*1000:.1f}mm")
                return None
            av_sequence.append(av)

        rospy.loginfo("IK Solved successfully.")
        return av_sequence

    def execute_motion(self):
        if not self.av_seq:
            rospy.logwarn("No plan available.")
            self.update_display("NoPlan\n1:Teach")
            time.sleep(2.0)
            return

        # --- 1. 開始時の姿勢を保存 ---
        rospy.loginfo("Saving start pose...")
        start_av = self.ri.angle_vector()

        rospy.loginfo("Executing motion...")
        self.update_display("Playing\n...")
        self.ri.servo_on()
        rospy.loginfo("Moving to trajectory start (3.0s)...")
        self.ri.angle_vector(self.av_seq[0], 3.0)
        self.ri.wait_interpolation()

        # 動作再生
        self.ri.angle_vector_sequence(self.av_seq, times=self.times)
        self.ri.wait_interpolation()

        # --- 2. 開始時の姿勢に戻る ---
        rospy.loginfo("Returning to start pose...")
        self.update_display("Back to\nStart")
        # 少し時間をかけてゆっくり戻る (例: 3.0秒)
        self.ri.angle_vector(start_av, 3.0)
        self.ri.wait_interpolation()

        rospy.loginfo("Motion Finished.")
        self.update_display("Done")

    def run(self):
        rospy.loginfo("Task Node Ready.")
        while not rospy.is_shutdown():
            self.update_display("1:Teach\n2:Play\n3:Free")
            btn = self.wait_for_button_press()

            if btn == 1:
                self.teach_mode_entry()
            elif btn == 2:
                self.execute_motion()
            elif btn == 3:
                rospy.loginfo("Servo OFF (Free Mode)")
                self.update_display("Free\nMode")
                self.ri.servo_off()
                time.sleep(1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default="")
    args = parser.parse_args()

    rospy.init_node("kxr_cleaning_task", anonymous=True)

    desc_param = args.namespace + "/robot_description" if is_ros_master_local() else args.namespace + "/robot_description_viz"
    if not is_ros_master_local():
        from kxr_models.download_urdf import download_urdf_mesh_files
        download_urdf_mesh_files(args.namespace)

    robot_model = RobotModel()
    from skrobot.utils.urdf import no_mesh_load_mode
    with no_mesh_load_mode():
        robot_model.load_urdf_from_robot_description(desc_param)

    ri = KXRROSRobotInterface(robot_model, namespace=args.namespace, controller_timeout=60.0)

    try:
        task = CleaningTask(ri, robot_model)
        task.run()
    except Exception as e:
        rospy.logerr(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
