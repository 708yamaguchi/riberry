#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import math
import numpy as np
import rospy
import tf

from std_msgs.msg import String, Int32
from sensor_msgs.msg import Imu
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


# ==========================================================
#  Trajectory Generators (動作パターン関数)
# ==========================================================
def generate_zigzag_trajectory(corners, corner_avs, step_width=0.02, **kwargs):
    """
    コーナー座標間を補間してジグザグ軌道を生成する関数。
    """
    p = [c.worldpos() for c in corners]
    r = [c.worldrot() for c in corners]
    av = corner_avs

    # 進行方向(0->3)の長さを基準にステップ数を計算
    len_advance = np.linalg.norm(p[3] - p[0])
    n_steps = max(1, int(len_advance / step_width))

    waypoints = []
    seed_avs = []

    for i in range(n_steps + 1):
        ratio = float(i) / n_steps
        # 位置の補間 (進行方向の左右の点を計算)
        base_left = p[0] + (p[3] - p[0]) * ratio
        base_right = p[1] + (p[2] - p[1]) * ratio

        # 回転とAVの補間
        rot_left = interpolate_rotation_matrices(ratio, r[0], r[3])
        rot_right = interpolate_rotation_matrices(ratio, r[1], r[2])
        av_left = av[0] + (av[3] - av[0]) * ratio
        av_right = av[1] + (av[2] - av[1]) * ratio

        # 座標系作成
        wp_l = Coordinates(pos=base_left, rot=rot_left)
        wp_r = Coordinates(pos=base_right, rot=rot_right)

        # 偶数回は左→右、奇数回は右→左（ジグザグ動作）
        if i % 2 == 0:
            waypoints.extend([wp_l, wp_r])
            seed_avs.extend([av_left, av_right])
        else:
            waypoints.extend([wp_r, wp_l])
            seed_avs.extend([av_right, av_left])

    return waypoints, seed_avs

def generate_radial_gathering_trajectory(corners, corner_avs, step_width=0.03, lift_offset=None, **kwargs):
    """
    領域の外周（辺）から中心に向かって掃き寄せるような放射状の軌道を生成する。
    戻る動作の際にアームを持ち上げて、次の開始点へ移動する。

    Args:
        corners (list[Coordinates]): コーナー4点の座標リスト
        corner_avs (list[numpy.ndarray]): コーナー4点の関節角度リスト
        step_width (float): 辺上の刻み幅 [m]
        lift_offset (tuple): 持ち上げ移動時のオフセット (x, y, z) [m]
    """
    p = [c.worldpos() for c in corners]
    r = [c.worldrot() for c in corners]
    av = corner_avs

    # オフセットをnumpy配列化
    lift_vec = np.array(lift_offset)

    # 1. 領域の中心点を計算
    center_pos = np.mean(p, axis=0)

    waypoints = []
    seed_avs = []

    # 4つの辺を順に処理 (0->1, 1->2, 2->3, 3->0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for (idx_start, idx_end) in edges:
        p_start, p_end = p[idx_start], p[idx_end]
        r_start, r_end = r[idx_start], r[idx_end]
        av_start, av_end = av[idx_start], av[idx_end]

        # 辺の長さを計算してステップ数を決定
        edge_len = np.linalg.norm(p_end - p_start)
        n_steps = max(1, int(edge_len / step_width))

        for i in range(n_steps):
            # 現在のステップの割合
            ratio = float(i) / n_steps
            # 次のステップの割合（次の開始点用）
            ratio_next = float(i + 1) / n_steps

            # --- 現在の点 (Current Start) ---
            pos_edge = p_start + (p_end - p_start) * ratio
            rot_edge = interpolate_rotation_matrices(ratio, r_start, r_end)
            av_edge = av_start + (av_end - av_start) * ratio

            # --- 中心側の点 (Current Inner) ---
            vec_to_center = center_pos - pos_edge
            sweep_ratio=1.0  # 中心への移動量の割合 (0.0~1.0)
            pos_inner = pos_edge + vec_to_center * sweep_ratio
            rot_inner = rot_edge # 回転は維持

            # --- 次の開始点 (Next Start) ---
            # ※最後のステップでは辺の終点（次の辺の始点）になる
            pos_next_edge = p_start + (p_end - p_start) * ratio_next
            rot_next_edge = interpolate_rotation_matrices(ratio_next, r_start, r_end)
            av_next_edge = av_start + (av_end - av_start) * ratio_next

            # --- 持ち上げ点 (Lifted Points) ---
            pos_inner_lift = pos_inner + lift_vec
            pos_next_edge_lift = pos_next_edge + lift_vec

            # 座標系(Coordinates)を作成
            wp_edge = Coordinates(pos=pos_edge, rot=rot_edge)
            wp_inner = Coordinates(pos=pos_inner, rot=rot_inner)
            wp_inner_lift = Coordinates(pos=pos_inner_lift, rot=rot_inner)
            wp_next_lift = Coordinates(pos=pos_next_edge_lift, rot=rot_next_edge)
            wp_next_edge = Coordinates(pos=pos_next_edge, rot=rot_next_edge)

            # --- 軌道生成シーケンス ---
            # 1. 辺の上 (Start)
            # 2. 内側へ (Sweep In)
            # 3. 内側で持ち上げ (Lift Up)
            # 4. 持ち上げたまま次の開始位置へ (Move to Next Top)
            # 5. 次の開始位置へ下ろす (Down to Next)
            waypoints.extend([
                wp_edge,
                wp_inner,
                wp_inner_lift,
                wp_next_lift,
                wp_next_edge
            ])

            # IKのSeed AV設定
            seed_avs.extend([
                av_edge,
                av_edge,
                av_edge,
                av_next_edge,
                av_next_edge
            ])

    return waypoints, seed_avs

# ==========================================================
#  Corner Teaching Task Class
# ==========================================================
class CornerTeachingTask(object):
    def __init__(self, ri, robot_model, prompt, trajectory_generator=None, vision_area_margin=0.0):
        self.ri = ri
        self.robot_model = robot_model

        # --- 設定値の受け取り ---
        self.prompt = prompt
        # 軌道生成関数が指定されていなければ、デフォルトのジグザグを使用
        self.trajectory_generator = trajectory_generator if trajectory_generator else generate_zigzag_trajectory
        self.target_speed = 0.15  # [m/s]
        self.min_time_step = 0.3

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
        self.manual_seed_av = None

        # --- IK設定値 ---
        self.error_tolerance = 0.02  # IK許容誤差[m]

        # --- IMUと重力関連 ---
        # 重力ベクトル (初期値はとりあえずZ下向きと仮定)
        self.gravity_vector = None
        rospy.Subscriber("/imu", Imu, self._cb_imu)

        # [Vision Mode Only] 画像認識時のみ使用するパラメータ[m]
        # 画像認識時の補正高さ (重力の逆方向にこの距離だけオフセット)
        self.gravity_compensation_offset = 0.02
        self.vision_area_margin = vision_area_margin

        # --- リンク設定 ---
        self._setup_kinematics()

    def _setup_kinematics(self):
        """ハードコードされたリンクを用いて手先座標系とIKチェーンを設定"""
        target_link_name = 'module5_base_link'

        if hasattr(self.robot_model, target_link_name):
            physical_link = getattr(self.robot_model, target_link_name)
        else:
            found = next((l for l in self.robot_model.link_list if l.name == target_link_name), None)
            if not found:
                raise ValueError(f"Link '{target_link_name}' not found.")
            physical_link = found

        self.end_coords = make_cascoords(parent=physical_link)
        # ee_offset = (-0.1, 0.0, 0.2)  # 刷毛把持用
        # ee_offset = (-0.12, 0.0, 0.08)  # 糊用グリッパ
        # ee_offset = (0.0, 0.0, 0.08)  # デフォルトグリッパ
        ee_offset = (-0.03, 0.0, 0.08)  # 布巾を持つとき（カメラから離した場所が先端になる）
        self.end_coords.translate(ee_offset, wrt="local")

        rospy.loginfo(f"Target Physical Link: {physical_link.name}")
        rospy.loginfo(f"End Effector set with local offset {ee_offset}")

    # --- ディスプレイ更新 ---
    def update_display(self, text):
        self.pub_display.publish('\n' + text)

    # --- コールバック ---
    def _cb_imu(self, data):
        """IMUデータからロボットベース座標系における重力方向ベクトルを算出・保存する"""
        # 1. IMU座標系(x, y, z) -> Base座標系(x, -y, -z)への変換と、加速度->重力(-acc)の変換
        gx, gy, gz = -data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z
        # 2. 正規化して保存
        norm = math.sqrt(gx**2 + gy**2 + gz**2)
        if norm > 0:
            self.gravity_vector = np.array([gx/norm, gy/norm, gz/norm])

    def _cb_atom_mode(self, msg):
        self.atom_mode = msg.data

    def _cb_atom_button(self, msg):
        self.current_button_state = msg.data

    def wait_for_button_press(self, valid_buttons=[1, 2, 3], timeout=None):
        self.current_button_state = 0
        start_time = time.time()

        while not rospy.is_shutdown():
            if timeout is not None and (time.time() - start_time > timeout):
                return None

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
        try:
            self.tf_listener.waitForTransform(self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {e}")
            return None

        rot_matrix = quaternion2matrix([rot[3], rot[0], rot[1], rot[2]])
        co_base_to_cam = Coordinates(pos=trans, rot=rot_matrix)

        co_cam_to_obj = Coordinates(
            pos=[pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
            rot=np.eye(3)
        )

        co_base_to_obj = co_base_to_cam.copy()
        co_base_to_obj.transform(co_cam_to_obj)
        return co_base_to_obj

    def _solve_ik(self, target_coords, seed_av):
        self.robot_model.angle_vector(seed_av)
        result = self.robot_model.inverse_kinematics(
            target_coords=target_coords,
            move_target=self.end_coords,
            rotation_axis=True,
            # rotation_axis=["xyz"],
            # rotation_axis=["x"],
            rthre=np.deg2rad(30),
            stop=50,
            revert_if_fail=False
        )

        dist_err = np.linalg.norm(self.end_coords.worldpos() - target_coords.worldpos())
        is_success = (result is not False) and (result is not None)

        if is_success:
            return self.robot_model.angle_vector()
        elif dist_err < self.error_tolerance:
            return self.robot_model.angle_vector()
        else:
            return None

    def _check_gravity_initialized(self):
        """重力ベクトルが初期化されているか確認する。NGならエラー表示してFalseを返す"""
        if self.gravity_vector is None:
            rospy.logerr("Gravity vector is not initialized. Check IMU topic.")
            self.update_display("Err:NoIMU")
            return False
        return True

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
            self.teach_corners_vision(long_side_stroke=True)

    # ==========================================================
    #  Manual Teaching
    # ==========================================================
    def teach_corners_manual(self):
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

            temp_corners.append(self.end_coords.copy_worldcoords())
            temp_avs.append(current_av)
            rospy.loginfo(f"Captured Corner {i+1}")

        self.update_display("Manual Done\nWait")
        rospy.loginfo("Manual Teaching finished. Keeping Servo OFF.")

        self.set_corners_and_plan(temp_corners, temp_avs)

    # ==========================================================
    #  Vision Teaching
    # ==========================================================
    def teach_corners_vision(self, long_side_stroke=True):
        self.ri.servo_off()
        while not rospy.is_shutdown():
            # Seed未設定時はStart(1)を押せないように制御
            if self.manual_seed_av is None:
                self.update_display("Need Seed\n2:Set Seed")
                valid_btns = [2]  # ボタン2のみ有効
            else:
                self.update_display("Ready\n1:Start 2:ReSeed")
                valid_btns = [1, 2] # 両方有効
            btn = self.wait_for_button_press(valid_buttons=valid_btns, timeout=1.0)
            if btn == 2:
                self.manual_seed_av = self.ri.angle_vector()
                rospy.loginfo("Manual seed_av captured.")
                self.update_display("Seed Saved!")
                time.sleep(1.0)
                continue
            elif btn == 1:
                break
        rospy.loginfo("Vision Mode: Servo ON. Starting countdown.")
        self.ri.servo_on()

        for i in range(3, 0, -1):
            msg = f"Vision\n{i}..."
            self.update_display(msg)
            rospy.loginfo(f"Countdown: {i}")
            time.sleep(1.0)

        self.update_display("Vision\nCapture!")
        rospy.loginfo("Capturing now...")

        try:
            rospy.wait_for_service("/estimate_corners", timeout=5.0)
            vision_srv = rospy.ServiceProxy("/estimate_corners", VisualPose)
        except rospy.ROSException:
            rospy.logerr("Vision service is not available.")
            self.update_display("Srv Timeout\nCheck ROS")
            return

        # --- 指定されたプロンプトを使用 ---
        req = VisualPoseRequest(prompt=self.prompt, mode="corners")
        rospy.loginfo(f"Calling Vision Service... prompt={req.prompt}")
        self.update_display("Vision\nThinking...")
        rospy.loginfo("Using Manually set Seed AV.")

        self.robot_model.angle_vector(self.manual_seed_av)
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

        base_coords_list = []
        for pose_msg in res.poses:
            c = self.get_base_coords_from_camera(pose_msg)
            if c is None:
                self.update_display("TF Fail")
                return
            base_coords_list.append(c)

        if len(base_coords_list) == 4:
            raw_pos = np.array([c.worldpos() for c in base_coords_list])
            center = raw_pos.mean(axis=0)
            angles = np.arctan2(raw_pos[:, 1] - center[1], raw_pos[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_coords = [base_coords_list[i] for i in sorted_indices]

            dists = [np.linalg.norm(c.worldpos()) for c in sorted_coords]
            min_idx = np.argmin(dists)
            sorted_coords = sorted_coords[min_idx:] + sorted_coords[:min_idx]

            p0 = sorted_coords[0].worldpos()
            p1 = sorted_coords[1].worldpos()
            p3 = sorted_coords[3].worldpos()

            dist_stroke = np.linalg.norm(p1 - p0)
            dist_advance = np.linalg.norm(p3 - p0)

            should_swap = False
            if long_side_stroke:
                if dist_advance > dist_stroke:
                    should_swap = True
            else:
                if dist_stroke > dist_advance:
                    should_swap = True

            if should_swap:
                rospy.loginfo("Swapping corners to match stroke preference.")
                sorted_coords = [sorted_coords[0], sorted_coords[3], sorted_coords[2], sorted_coords[1]]

            base_coords_list = sorted_coords

        positions = np.array([c.worldpos() for c in base_coords_list])
        center_pos = np.mean(positions, axis=0)

        temp_corners = []
        temp_avs = []

        if not self._check_gravity_initialized():
            return

        gravity_compensation_vec = self.gravity_vector * -1.0 * self.gravity_compensation_offset
        rospy.loginfo(f"[Vision] Applying Gravity Compensation Vector: {gravity_compensation_vec}")

        for i, raw_coord in enumerate(base_coords_list):
            pos = raw_coord.worldpos()
            vec = pos - center_pos
            vec_len = np.linalg.norm(vec)
            if vec_len > 1e-6:
                direction = vec / vec_len
                pos = pos + direction * self.vision_area_margin

            offset_vec = gravity_compensation_vec
            pos += offset_vec
            target = Coordinates(pos=pos, rot=base_rot)

            av = self._solve_ik(target, self.manual_seed_av)
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

        if not self._check_gravity_initialized():
            self.av_seq = []
            return

        # lift_vec = 重力ベクトル * -1 (上向き) * スカラー高さ
        lift_height = 0.10  # [m]
        lift_vec = self.gravity_vector * -1.0 * lift_height
        rospy.loginfo(f"[Plan] Applying Lift Vector: {lift_vec} (based on Gravity)")

        # --- 外部から注入された軌道生成関数を使用 ---
        rospy.loginfo(f"Generating trajectory using: {self.trajectory_generator.__name__}")
        waypoints, seed_avs = self.trajectory_generator(
            self.corners,
            self.corner_avs,
            lift_offset=lift_vec
        )

        seq = self.solve_full_ik(waypoints, seed_avs)

        if seq is None:
            rospy.logerr("Planning failed.")
            self.update_display("IK Fail\n1:Retry")
            self.av_seq = []
            return

        self.av_seq = seq
        self.times = []
        # 最初の点は、開始姿勢(av_seq[0])へ移動済みとして、短い時間または0を入れる
        self.times.append(1.0)
        # 2点目以降の時間を計算
        for i in range(1, len(waypoints)):
            # 前回の座標と今回の座標の距離を計算
            pos_prev = np.array(waypoints[i-1].worldpos())
            pos_curr = np.array(waypoints[i].worldpos())
            dist = np.linalg.norm(pos_curr - pos_prev)
            # 時間 = 距離 / 速度
            # ただしゼロ割防止と、回転のみの動作考慮で最小時間を設ける
            dt = max(self.min_time_step, dist / self.target_speed)
            self.times.append(dt)

        # 合計時間をログ表示
        total_time = sum(self.times)
        rospy.loginfo(f"Plan ready. Total waypoints: {len(self.av_seq)}, Est. Duration: {total_time:.1f}s, Speed: {self.target_speed}m/s")

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

        rospy.loginfo("Saving start pose...")
        start_av = self.ri.angle_vector()

        rospy.loginfo("Executing motion...")
        self.update_display("Playing\n1:STOP")
        self.ri.servo_on()
        rospy.loginfo("Moving to trajectory start (3.0s)...")
        self.ri.angle_vector(self.av_seq[0], 3.0)
        self.ri.wait_interpolation()

        self.ri.angle_vector_sequence(self.av_seq, times=self.times)
        is_canceled = False
        # 1クリックで動作をキャンセル
        while not rospy.is_shutdown() and self.ri.is_interpolating():
            btn = self.wait_for_button_press(valid_buttons=[1], timeout=0.1)
            if btn == 1:
                self.ri.cancel_angle_vector()  # 動作をキャンセル
                is_canceled = True
                break
        if is_canceled:
            rospy.loginfo("Play interrupted by user.")
            self.update_display("STOPPED")
            time.sleep(1.0)
            return

        rospy.loginfo("Returning to start pose...")
        self.update_display("Back to\nStart")
        self.ri.angle_vector(start_av, 3.0)
        self.ri.wait_interpolation()

        rospy.loginfo("Motion Finished.")
        self.update_display("Done")

    def run(self):
        rospy.loginfo("Task Node Ready.")
        while not rospy.is_shutdown():
            self.update_display("1:Teach\n2:Play\n3:Free")
            btn = self.wait_for_button_press(timeout=1.0)

            if btn == 1:
                self.teach_mode_entry()
            elif btn == 2:
                self.execute_motion()
            elif btn == 3:
                rospy.loginfo("Servo OFF (Free Mode)")
                self.update_display("Free\nMode")
                self.ri.servo_off()
                time.sleep(1.0)


# ==========================================================
#  Setup Functions
# ==========================================================
def setup_robot(namespace):
    """ロボットモデルの読み込みとインターフェースの初期化を行う"""
    desc_param = namespace + "/robot_description" if is_ros_master_local() else namespace + "/robot_description_viz"
    
    if not is_ros_master_local():
        from kxr_models.download_urdf import download_urdf_mesh_files
        download_urdf_mesh_files(namespace)

    robot_model = RobotModel()
    from skrobot.utils.urdf import no_mesh_load_mode
    with no_mesh_load_mode():
        robot_model.load_urdf_from_robot_description(desc_param)

    ri = KXRROSRobotInterface(robot_model, namespace=namespace, controller_timeout=60.0)
    return ri, robot_model


def execute_instruction(ri, robot_model, verb, target_object):
    """指示(SVO)に基づいてパラメータを解決し、タスクを実行する"""
    try:
        # ==========================================================
        #  パラメータの自動解決 (Action Registry)
        # ==========================================================
        action_registry = {
            "clean": {
                "func": generate_zigzag_trajectory,  # ジグザグに掃除
                "margin": 0.03,  # ゴミを取りこぼさないよう、認識領域より少し広く取る
            },
            "collect": {
                "func": generate_radial_gathering_trajectory,  # 放射状に集める
                "margin": 0.08,  # ゴミを取りこぼさないよう、認識領域より少し広く取る
            },
            "paint": {
                "func": generate_zigzag_trajectory,  # ジグザグに塗る
                "margin": -0.03,  # はみ出さないように、少し狭くする
            }
        }

        # 登録されていないVerbが来た場合はエラーを出して終了
        if verb not in action_registry:
            valid_verbs = list(action_registry.keys())
            rospy.logerr(f"Unknown instruction verb: '{verb}'. Supported verbs are: {valid_verbs}")
            return

        config = action_registry[verb]
        
        rospy.loginfo(f"Task Instruction: Verb='{verb}', Object='{target_object}'")
        rospy.loginfo(f"-> Trajectory: {config['func'].__name__}, Margin: {config['margin']}")

        # ==========================================================
        #  Taskの起動
        # ==========================================================
        task = CornerTeachingTask(
            ri,
            robot_model,
            prompt=target_object,
            trajectory_generator=config["func"],
            vision_area_margin=config["margin"],
        )
        task.run()

    except Exception as e:
        rospy.logerr(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default="")
    args = parser.parse_args()

    rospy.init_node("corner_teaching_task", anonymous=True)

    # 1. ロボットセットアップ (Setup Phase)
    ri, robot_model = setup_robot(args.namespace)

    # 2. 指示 (Instruction Phase)
    # ここを変えるだけで挙動が変わる
    instruction_verb = "collect"       # 動作
    instruction_object = "coffee powder"  # 対象

    # 3. タスク実行 (Execution Phase)
    execute_instruction(ri, robot_model, instruction_verb, instruction_object)


if __name__ == "__main__":
    main()
