#!/usr/bin/env python

import argparse
import json
import math
import time

from kxr_controller.check_ros_master import is_ros_master_local
from kxr_controller.kxr_interface import KXRROSRobotInterface
from kxr_controller.msg import ServoOnOff
import numpy as np
from riberry_startup.srv import TaskInstruction
from riberry_startup.srv import TaskInstructionResponse
from riberry_startup.srv import VisualPose
from riberry_startup.srv import VisualPoseRequest
import rospy
from sensor_msgs.msg import Imu
from skrobot.coordinates import Coordinates
from skrobot.coordinates import make_cascoords
from skrobot.coordinates.math import interpolate_rotation_matrices
from skrobot.coordinates.math import quaternion2matrix
from skrobot.model import RobotModel
from std_msgs.msg import Int32
from std_msgs.msg import String
import tf


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


def generate_radial_gathering_trajectory(corners, corner_avs, step_width=0.03, gravity_vector=None, lift_height=0.1, **kwargs):
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

    # --- 持ち上げベクトルの決定 ---
    # 重力と逆方向(上)へ lift_height 分
    lift_vec = gravity_vector * -1.0 * lift_height

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
            sweep_ratio = 1.0  # 中心への移動量の割合 (0.0~1.0)
            pos_inner = pos_edge + vec_to_center * sweep_ratio
            rot_inner = rot_edge  # 回転は維持

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


def generate_spiral_stirring_trajectory(corners, corner_avs, step_width=0.02, gravity_vector=None, stir_depth=0.0, manual_seed_av=None, **kwargs):
    """
    4隅で定義される平面上で、外側から中心へ向かい、また外側へ戻る螺旋軌道を生成する。

    修正点:
    - 座標軸を強制的に直交化(Gram-Schmidt)し、歪んだコーナー配置でも真円を描けるように修正。
    """
    p = [c.worldpos() for c in corners]
    r = [c.worldrot() for c in corners]
    av = corner_avs

    # 1. 平面の定義
    center_pos = np.mean(p, axis=0)

    # --- 座標軸の直交化 (正規直交基底の生成) ---
    vec_a = p[1] - p[0] # 仮のX軸
    vec_b = p[3] - p[0] # 仮のY軸 (直交しているとは限らない)

    # 平面の法線ベクトルを算出
    normal = np.cross(vec_a, vec_b)
    norm_n = np.linalg.norm(normal)
    if norm_n < 1e-6:
        normal = np.array([0, 0, 1]) # 潰れている場合はZ軸仮定
    else:
        normal /= norm_n

    # unit_x を決定 (vec_a 方向)
    len_a = np.linalg.norm(vec_a)
    if len_a < 1e-6:
        unit_x = np.array([1, 0, 0])
    else:
        unit_x = vec_a / len_a

    # unit_y を決定 (normal と unit_x に直交するベクトル)
    # これにより unit_x ⊥ unit_y が保証される
    unit_y = np.cross(normal, unit_x)

    # 半径の基準とする長さ（元の形状のサイズ感を使用）
    # vec_b の長さそのものではなく、unit_y 方向への射影長を使うのが厳密だが、
    # ここではシンプルに元の辺の長さを使ってサイズを決定する
    len_x = np.linalg.norm(vec_a)
    len_y = np.linalg.norm(vec_b)

    # --- 設定パラメータ ---
    radius_margin = kwargs.get('radius_margin', 1.0)
    max_radius = min(len_x, len_y) / 2.0 * radius_margin
    spiral_turns = kwargs.get('turns', 2)

    # --- ログ出力 ---
    rospy.loginfo("\n" + "="*30)
    rospy.loginfo(f"[Spiral] Detected Center: {center_pos}")

    if gravity_vector is not None and abs(stir_depth) > 1e-6:
        depth_offset = gravity_vector * stir_depth
        center_pos += depth_offset
        rospy.loginfo(f"[Spiral] Applying Depth Offset: {depth_offset} (Depth={stir_depth}m)")

    rospy.loginfo(f"[Spiral] Max Radius: {max_radius:.4f} m (Margin: {radius_margin})")
    # 歪みチェック用ログ
    dot_prod = np.dot(vec_a / len_a, vec_b / len_y)
    angle_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    rospy.loginfo(f"[Spiral] Corner Angle (p1-p0-p3): {angle_deg:.1f} deg (Orthogonalized)")
    rospy.loginfo("="*30 + "\n")

    # 姿勢設定
    center_rot = interpolate_rotation_matrices(0.5, r[0], r[2])

    if manual_seed_av is not None:
        center_av = np.array(manual_seed_av)
    else:
        center_av = np.mean(av, axis=0)

    # --- 軌道生成の計算 ---
    outer_circle_len = 2 * np.pi * max_radius
    spiral_len = 0.5 * max_radius * (2 * np.pi * spiral_turns)

    n_steps_outer = max(10, int(outer_circle_len / step_width))
    n_steps_spiral = max(20, int(spiral_len / step_width))

    waypoints = []
    seed_avs = []

    # 重み付きシード計算関数
    def compute_weighted_seed(current_pos, corner_positions, corner_avs, center_pos, center_av):
        dists = [np.linalg.norm(current_pos - c_pos) for c_pos in corner_positions]
        dist_center = np.linalg.norm(current_pos - center_pos)
        weights = [1.0 / (d + 1e-4) for d in dists]
        weights.append(1.0 / (dist_center + 1e-4))
        weights = np.array(weights)
        weights /= np.sum(weights)
        target_avs = list(corner_avs) + [center_av]
        weighted_av = np.zeros_like(center_av)
        for w, av_vec in zip(weights, target_avs):
            weighted_av += w * av_vec
        return weighted_av

    def add_point(radius, angle):
        # 直交基底 unit_x, unit_y を使って移動
        offset_x = radius * math.cos(angle)
        offset_y = radius * math.sin(angle)
        pos = center_pos + (unit_x * offset_x) + (unit_y * offset_y)

        wp = Coordinates(pos=pos, rot=center_rot)
        seed = compute_weighted_seed(pos, p, av, center_pos, center_av)
        waypoints.append(wp)
        seed_avs.append(seed)

    # ==========================
    # Phase 1: 往路 (外 -> 中)
    # ==========================
    # 1-A. 外周を1周回る
    for i in range(n_steps_outer):
        ratio = float(i) / n_steps_outer
        angle = 2 * np.pi * ratio
        add_point(max_radius, angle)

    # 1-B. 外から中へ螺旋
    for i in range(n_steps_spiral + 1):
        t = float(i) / n_steps_spiral
        r_ratio = math.sqrt(1.0 - t)
        current_radius = max_radius * r_ratio
        current_angle = (2 * np.pi) + (2 * np.pi * spiral_turns * t)
        add_point(current_radius, current_angle)

    # ==========================
    # Phase 2: 復路 (中 -> 外)
    # ==========================
    end_angle_inward = (2 * np.pi) + (2 * np.pi * spiral_turns)
    for i in range(1, n_steps_spiral + 1):
        t = float(i) / n_steps_spiral
        r_ratio = math.sqrt(t)
        current_radius = max_radius * r_ratio
        current_angle = end_angle_inward + (2 * np.pi * spiral_turns * t)
        add_point(current_radius, current_angle)

    return waypoints, seed_avs


def generate_grid_pressing_trajectory(corners, corner_avs, step_width=0.05, gravity_vector=None, press_stroke=0.03, manual_seed_av=None, press_steps=5, gravity_comp_offset=0.0, base_offset=(0, 0, 0), **kwargs):
    """
    領域内をグリッド状に移動し、各点に対して垂直方向の押し込み（プレス）動作を行う。

    Args:
        base_offset (tuple): ロボットベース座標系基準のオフセット (x, y, z) [m]
                             認識位置全体をこの量だけ平行移動させる。
        press_steps (int): 押し込み動作の分割数（動作をゆっくりにするため）
    """
    # --- 1. ベース座標系オフセットの適用 ---
    # 入力されたコーナー座標全体を、ロボット基準でずらす
    offset_vec = np.array(base_offset)
    p = [c.worldpos() + offset_vec for c in corners]
    r = [c.worldrot() for c in corners]
    av = corner_avs

    rospy.loginfo(f"[Press] Applying Base Offset: {base_offset}")

    # --- 2. ベクトル計算 ---
    if gravity_vector is None:
        g_vec = np.array([0, 0, -1.0])
    else:
        g_vec = np.array(gravity_vector)

    vec_up = -1.0 * g_vec
    vec_down = g_vec

    # 進行方向(0->3)の長さを基準に「行数」を計算
    len_long = np.linalg.norm(p[3] - p[0])
    n_rows = max(1, int(len_long / step_width))

    waypoints = []
    seed_avs = []

    center_pos_global = np.mean(p, axis=0)
    center_av_global = np.array(manual_seed_av) if manual_seed_av is not None else np.mean(av, axis=0)

    def compute_weighted_seed(current_pos):
        dists = [np.linalg.norm(current_pos - c_pos) for c_pos in p]
        weights = [1.0 / (d + 1e-4) for d in dists]
        dist_center = np.linalg.norm(current_pos - center_pos_global)
        weights.append(1.0 / (dist_center + 1e-4))

        weights = np.array(weights)
        weights /= np.sum(weights)

        target_avs = list(av) + [center_av_global]
        weighted_av = np.zeros_like(center_av_global)
        for w, av_vec in zip(weights, target_avs):
            weighted_av += w * av_vec
        return weighted_av

    # グリッド走査
    for i in range(n_rows + 1):
        ratio_row = float(i) / n_rows

        p_left = p[0] + (p[3] - p[0]) * ratio_row
        p_right = p[1] + (p[2] - p[1]) * ratio_row

        r_left = interpolate_rotation_matrices(ratio_row, r[0], r[3])
        r_right = interpolate_rotation_matrices(ratio_row, r[1], r[2])

        len_lat = np.linalg.norm(p_right - p_left)
        n_cols = max(1, int(len_lat / step_width))

        col_range = range(n_cols + 1) if i % 2 == 0 else range(n_cols, -1, -1)

        for j in col_range:
            ratio_col = float(j) / n_cols

            # cornersで定義される「安全高さ(Safe Plane)」上の点
            p_safe_plane = p_left + (p_right - p_left) * ratio_col
            r_surf = interpolate_rotation_matrices(ratio_col, r_left, r_right)

            # --- 高さの補正 ---
            # 1. 基準位置 (Top)
            pos_top = p_safe_plane + (vec_down * gravity_comp_offset)
            # 2. 押し込み位置 (Bottom)
            pos_bottom = pos_top + (vec_down * press_stroke)

            # --- シーケンス生成 ---

            # 1. まず基準位置(Top)へ移動 (アプローチ)
            wp_top_approach = Coordinates(pos=pos_top, rot=r_surf)
            seed_top = compute_weighted_seed(pos_top)

            waypoints.append(wp_top_approach)
            seed_avs.append(seed_top)

            # 2. 押し込み (Top -> Bottom) - 分割してゆっくり
            steps = max(1, press_steps)
            for k in range(1, steps + 1):
                ratio = float(k) / steps
                pos_inter = pos_top + (vec_down * press_stroke * ratio)

                wp_inter = Coordinates(pos=pos_inter, rot=r_surf)
                seed_inter = compute_weighted_seed(pos_inter)

                waypoints.append(wp_inter)
                seed_avs.append(seed_inter)

            # 3. 戻り (Bottom -> Top) - 分割してゆっくり
            for k in range(1, steps + 1):
                ratio = float(k) / steps
                pos_inter = pos_bottom + (vec_up * press_stroke * ratio)

                wp_inter = Coordinates(pos=pos_inter, rot=r_surf)
                seed_inter = compute_weighted_seed(pos_inter)

                waypoints.append(wp_inter)
                seed_avs.append(seed_inter)

    return waypoints, seed_avs


# ==========================================================
#  Corner Teaching Task Class
# ==========================================================
class CornerTeachingTask:
    def __init__(self, ri, robot_model):
        self.ri = ri
        self.robot_model = robot_model

        # --- 設定ファイルの読み込み ---
        config_path = rospy.get_param("~task_config_path")
        rospy.loginfo(f"Loading task config from: {config_path}")
        try:
            with open(config_path, encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config file: {e}")
            exit(1)

        self.requested_from_service = False

        # --- 現在のタスクパラメータ (動的に変わる) ---
        self.current_task_params = None
        # --- ロボット動作の固定パラメータ (Const Params) ---
        # 変化しない設定値をここに集約します
        self.const_params = {
            "target_speed": 0.15,           # [m/s]
            "min_time_step": 0.3,           # [s]
            "error_tolerance": 0.02,        # [m] IK許容誤差
            "gravity_comp_offset": 0.20,    # [m] Vision認識時の重力補正高さ
            "lift_height": 0.10,            # [m] 移動時の持ち上げ高さ
            "stir_depth": 0.06,             # [m] かき混ぜ時の深さ
            "press_stroke": 0.18,           # 押し込み深さ (基準高さより下)

            "vision_base_offset": (0.0, 0.0, 0.0),  # [m] ベース座標系相対でのオフセット

            # "ee_offset": (-0.1, 0.0, 0.2),  # 刷毛把持用
            # "ee_offset": (-0.12, 0.0, 0.08),  # 糊用グリッパ
            # "ee_offset": (0.0, 0.0, 0.08),  # デフォルトグリッパ
            # "ee_offset": (-0.03, 0.0, 0.08),  # 布巾を持つとき（カメラから離した場所が先端になる）
            # "ee_offset": (-0.12, 0.0, 0.25),  # 箸をもつとき
            # "ee_offset": (-0.11, 0.0, 0.14),  # 押し洗い用エンドエフェクタ
            "ee_offset": (-0.12, 0.0, 0.15),    # 毛玉とるとる用エンドエフェクタ
        }

        self.func_map = {
            "zigzag": generate_zigzag_trajectory,
            "radial": generate_radial_gathering_trajectory,
            "spiral": generate_spiral_stirring_trajectory,
            "press": generate_grid_pressing_trajectory,
        }

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

        self.servo_on_states = None
        ns = self.ri.namespace if self.ri.namespace else ""
        servo_topic = ns + "fullbody_controller/servo_on_off_real_interface/state"
        rospy.Subscriber(servo_topic, ServoOnOff, self._cb_servo_on_states, queue_size=1)

        self.srv_server = rospy.Service(
            '/task_instruction',
            TaskInstruction,
            self._cb_task_instruction
        )
        rospy.loginfo("Service /task_instruction is ready.")

        # --- 保存データ ---
        self.corners = []
        self.corner_avs = []
        self.av_seq = []
        self.times = []
        self.manual_seed_av = None

        # --- IMUと重力関連 ---
        # 重力ベクトル (初期値はとりあえずZ下向きと仮定)
        self.gravity_vector = None
        rospy.Subscriber("/imu", Imu, self._cb_imu)

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
        ee_offset = self.const_params["ee_offset"]
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
            self.gravity_vector = np.array([gx / norm, gy / norm, gz / norm])

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

    def _cb_servo_on_states(self, msg):
        self.servo_on_states = msg

    def toggle_servo_on_off(self):
        """
        If one of the servos is on, turn the entire servo off.
        If all of the servos is off, turn the entire servo on.
        """
        if self.ri is None:
            rospy.logwarn("KXRROSRobotInterface instance is not created.")
            return
        if self.servo_on_states is None:
            rospy.logwarn("Servo states not received yet.")
            return

        servo_on_states = self.servo_on_states.servo_on_states
        if any(servo_on_states) is True:
            rospy.loginfo("Toggle: Servo OFF")
            self.ri.servo_off()
            self.update_display("Servo OFF")
        else:
            rospy.loginfo("Toggle: Servo ON")
            self.ri.servo_on()
            self.update_display("Servo ON")
        time.sleep(1.0)  # チャタリング防止

    # ==========================================================
    #  Service Callback (Updated)
    # ==========================================================
    def _cb_task_instruction(self, req):
        """外部からのタスク指示を受信して設定を更新し、実行フラグを立てる"""
        rospy.loginfo(f"Task Request Received: {req.action_verb} {req.target_object}")

        action_registry = self.config.get("action_registry", {})
        object_registry = self.config.get("object_registry", {})

        if req.action_verb not in action_registry:
            msg = f"Unknown verb: {req.action_verb}"
            rospy.logerr(msg)
            return TaskInstructionResponse(success=False, message=msg)

        if req.target_object not in object_registry:
            msg = f"Unknown target: {req.target_object}"
            rospy.logerr(msg)
            return TaskInstructionResponse(success=False, message=msg)

        action_data = action_registry[req.action_verb]
        traj_type = action_data["trajectory_type"]
        trajectory_func = self.func_map.get(traj_type)
        vision_strategy = object_registry[req.target_object]
        rotation_axis = action_data["rotation_axis"]

        self.current_task_params = {
            "prompt": req.target_object,
            "vision_strategy": vision_strategy,
            "vision_area_margin": action_data["margin"],
            "trajectory_generator": trajectory_func,
            "repeat_count": req.repeat_value,
            "repeat_unit": req.repeat_unit,
            "rotation_axis": rotation_axis
        }

        rospy.loginfo(
            "\n=============================================\n"
            " Task Instruction Accepted\n"
            "=============================================\n"
            f" - Target Object      : {req.target_object}\n"
            f" - Vision Strategy    : {vision_strategy}\n"
            f" - Vision Area Margin : {action_data['margin']} m\n"
            f" - Action Verb        : {req.action_verb}\n"
            f" - Trajectory Type    : {traj_type} ({trajectory_func.__name__})\n"
            f" - Rotation Axis      : {rotation_axis}\n"
            f" - Repeat             : {req.repeat_value} {req.repeat_unit}\n"
            "============================================="
        )

        self.corners = []
        self.av_seq = []
        self.requested_from_service = True

        self.update_display(f"New Task:\n{req.target_object}")
        rospy.loginfo("Task accepted. Automation sequence initiated.")

        return TaskInstructionResponse(success=True, message="Task accepted. Starting automation.")

    # ==========================================================
    #  Helper: 共通処理 (TF変換 / IK計算)
    # ==========================================================
    def _transform_pose_to_base(self, pose_msg):
        """Camera座標系のPoseMsgをBase座標系のCoordinatesに変換"""
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

        co_base_to_cam.transform(co_cam_to_obj)
        return co_base_to_cam

    def _solve_ik(self, target_coords, seed_av):
        rot_axis = True
        if self.current_task_params and "rotation_axis" in self.current_task_params:
            rot_axis = self.current_task_params["rotation_axis"]

        self.robot_model.angle_vector(seed_av)
        result = self.robot_model.inverse_kinematics(
            target_coords=target_coords,
            move_target=self.end_coords,
            rotation_axis=rot_axis,
            rthre=np.deg2rad(5),
            # rthre=np.deg2rad(45),
            stop=50,
            revert_if_fail=False
        )

        dist_err = np.linalg.norm(self.end_coords.worldpos() - target_coords.worldpos())
        is_success = (result is not False) and (result is not None)

        if is_success:
            return self.robot_model.angle_vector()
        elif dist_err < self.const_params["error_tolerance"]:
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
    def manual_mode_entry(self):
        """マニュアルモード: Teach(1) か Play(2) を選択"""
        while not rospy.is_shutdown():
            # 計画が存在する場合のみ Play を表示・選択可能にする
            if self.av_seq:
                # ユーザー要望に合わせて改行を入れる
                self.update_display("Manual\n1: Teach\n2: Play")
                valid_btns = [1, 2, 3]
            else:
                self.update_display("Manual\n1: Teach")
                valid_btns = [1, 3]

            btn = self.wait_for_button_press(valid_buttons=valid_btns, timeout=0.5)

            if btn == 1:
                rospy.loginfo("Manual: Teach Start")
                self.teach_corners_manual()
                # Teachが終わったらループを継続し、Playが表示されるようにする
            elif btn == 2:
                rospy.loginfo("Manual: Play Start")
                self.execute_motion()
                # Playが終わったらループを抜けてメインメニュー（最初の画面）に戻る
                # これにより、ボタン3でServo ON/OFFができるようになる
                rospy.loginfo("Play finished, returning to main menu.")
                break
            elif btn == 3:
                # 戻る
                rospy.loginfo("Exit Manual Mode")
                break

    # ==========================================================
    #  Manual Teaching
    # ==========================================================
    def teach_corners_manual(self):
        temp_corners = []
        temp_avs = []

        rospy.loginfo("Start Manual Teaching: Servo OFF")
        self.trajectory_generator = generate_zigzag_trajectory
        rospy.loginfo("Manual Mode: Force trajectory type to 'zigzag'")

        self.ri.servo_off()

        for i in range(4):
            self.update_display(f"Manual {i + 1}/4\n1:Set")
            rospy.loginfo(f"Waiting for Corner {i + 1}...")

            if self.wait_for_button_press(valid_buttons=[1]) is None:
                return

            current_av = self.ri.angle_vector()
            self.robot_model.angle_vector(current_av)

            temp_corners.append(self.end_coords.copy_worldcoords())
            temp_avs.append(current_av)
            rospy.loginfo(f"Captured Corner {i + 1}")

        self.update_display("Manual Done\nWait")
        rospy.loginfo("Manual Teaching finished. Keeping Servo OFF.")

        self.set_corners_and_plan(
            temp_corners,
            temp_avs,
            trajectory_generator=generate_zigzag_trajectory
        )

    # ==========================================================
    #  Vision Teaching
    # ==========================================================
    def set_vision_seed(self):
        rospy.loginfo("Setting IK Seed for Vision...")
        # 呼ばれた瞬間の姿勢をシードとして保存
        self.manual_seed_av = self.ri.angle_vector()

        rospy.loginfo("Manual seed_av captured.")
        self.update_display("Seed Saved!")
        time.sleep(1.5)

    def teach_corners_vision(self, long_side_stroke=True):
        """
        Visionによるコーナー検出とIK計算を行う。
        ボタン操作は排除し、実行可能かどうかの判定と処理のみ行う。
        """
        # シードがない場合は失敗
        if self.manual_seed_av is None:
            rospy.logerr("Seed AV is not set. Please set seed manually first.")
            return False

        if self.current_task_params is None:
            rospy.logerr("No task parameters set.")
            return False
        params = self.current_task_params  # 短い名前でアクセス

        rospy.loginfo("Vision Mode: Servo ON. Starting countdown.")
        self.ri.servo_on()

        for i in range(3, 0, -1):
            msg = f"Vision\n{i}..."
            self.update_display(msg)
            rospy.loginfo(f"Countdown: {i}")
            time.sleep(1.0)

        self.update_display("Vision\nCapture!")

        try:
            rospy.wait_for_service("/estimate_corners", timeout=5.0)
            vision_srv = rospy.ServiceProxy("/estimate_corners", VisualPose)
        except rospy.ROSException:
            rospy.logerr("Vision service is not available.")
            self.update_display("Srv Timeout\nCheck ROS")
            return

        # --- 指定されたプロンプトを使用 ---
        req = VisualPoseRequest(
            prompt=params["prompt"],
            mode="corners",
            strategy=params["vision_strategy"]
        )
        rospy.loginfo(f"Calling Vision Service... prompt={req.prompt}, strategy='{req.strategy}'")
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
            c = self._transform_pose_to_base(pose_msg)
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

        gravity_compensation_vec = self.gravity_vector * -1.0 * self.const_params["gravity_comp_offset"]
        rospy.loginfo(f"[Vision] Applying Gravity Compensation Vector: {gravity_compensation_vec}")

        for i, raw_coord in enumerate(base_coords_list):
            pos = raw_coord.worldpos()
            vec = pos - center_pos
            vec_len = np.linalg.norm(vec)
            if vec_len > 1e-6:
                direction = vec / vec_len
                pos = pos + direction * params["vision_area_margin"]

            offset_vec = gravity_compensation_vec
            pos += offset_vec
            target = Coordinates(pos=pos, rot=base_rot)

            av = self._solve_ik(target, self.manual_seed_av)
            if av is None:
                rospy.logerr(f"IK Failed for Vision Corner {i + 1}")
                self.update_display(f"IK Fail\nCorner{i + 1}")
                return

            temp_corners.append(target)
            temp_avs.append(av)

        rospy.loginfo("All vision corners processed.")
        self.update_display("Vision Done\nWait")
        self.set_corners_and_plan(
            temp_corners,
            temp_avs,
            trajectory_generator=params["trajectory_generator"]
        )

        if self.av_seq:
            return True
        else:
            return False

    # ==========================================================
    #  Planning & Execution
    # ==========================================================
    def set_corners_and_plan(self, corners, avs, trajectory_generator):
        self.corners = corners
        self.corner_avs = avs

        if len(self.corners) != 4:
            return

        if not self._check_gravity_initialized():
            self.av_seq = []
            return

        # radial用: 持ち上げ高さ
        lift_height = self.const_params["lift_height"]
        # spiral用: 鍋底への沈み込み深さ
        stir_depth = self.const_params["stir_depth"]
        press_stroke = self.const_params["press_stroke"]

        base_offset = self.const_params["vision_base_offset"]

        # --- 外部から注入された軌道生成関数を使用 ---
        rospy.loginfo(f"Generating trajectory using: {trajectory_generator.__name__}")
        waypoints, seed_avs = trajectory_generator(
            self.corners,
            self.corner_avs,
            gravity_vector=self.gravity_vector, # 共通: 重力ベクトル
            lift_height=lift_height,            # Radial用: 持ち上げ高さ
            stir_depth=stir_depth,              # Spiral用: 沈める深さ
            press_stroke=press_stroke,          # Press用
            manual_seed_av=self.manual_seed_av,
            base_offset=base_offset,
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
        target_speed = self.const_params["target_speed"]
        min_time_step = self.const_params["min_time_step"]
        # 2点目以降の時間を計算
        for i in range(1, len(waypoints)):
            # 前回の座標と今回の座標の距離を計算
            pos_prev = np.array(waypoints[i - 1].worldpos())
            pos_curr = np.array(waypoints[i].worldpos())
            dist = np.linalg.norm(pos_curr - pos_prev)
            # 時間 = 距離 / 速度
            # ただしゼロ割防止と、回転のみの動作考慮で最小時間を設ける
            dt = max(min_time_step, dist / target_speed)
            self.times.append(dt)

        # 合計時間をログ表示
        total_time = sum(self.times)
        rospy.loginfo(f"Plan ready. Total waypoints: {len(self.av_seq)}, Est. Duration: {total_time:.1f}s, Speed: {target_speed}m/s")

    def solve_full_ik(self, waypoints, seed_avs):
        rospy.loginfo("Solving IK for full trajectory...")
        self.update_display("Plan\nWait")

        av_sequence = []
        for i, (wp, seed) in enumerate(zip(waypoints, seed_avs)):
            av = self._solve_ik(wp, seed)
            if av is None:
                dist = np.linalg.norm(self.end_coords.worldpos() - wp.worldpos())
                rospy.logwarn(f"Point {i}: IK Failed. Error: {dist * 1000:.1f}mm")
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

        # --- リピート設定の取得 ---
        # 手動Playなどでparamsが無い場合はデフォルト(1回)
        params = self.current_task_params if self.current_task_params else {}
        rep_val = params.get("repeat_count", 1)
        rep_unit = params.get("repeat_unit", "times")

        # 時間指定の場合の終了時刻計算
        start_time_unix = time.time()
        time_limit = rep_val * 60.0 if rep_unit == "minutes" else 0

        rospy.loginfo(f"Execution Start: {rep_val} {rep_unit}")
        rospy.loginfo("Saving home pose...")
        home_av = self.ri.angle_vector() # 終了後に戻る場所

        self.ri.servo_on()

        loop_count = 0
        is_canceled = False

        while not rospy.is_shutdown():
            # --- 終了判定 ---
            if rep_unit == "times":
                if loop_count >= rep_val:
                    rospy.loginfo("Repeat count reached.")
                    break
                display_status = f"Rep {loop_count + 1}/{rep_val}"
            elif rep_unit == "minutes":
                elapsed = time.time() - start_time_unix
                if elapsed >= time_limit:
                    rospy.loginfo("Time limit reached.")
                    break
                remaining = int(time_limit - elapsed)
                display_status = f"Time {remaining}s"
            else:
                # 未知の単位なら1回で終了
                if loop_count >= 1: break
                display_status = "Playing"

            rospy.loginfo(f"--- Loop {loop_count + 1} Start ({display_status}) ---")
            self.update_display(f"{display_status}\n1:STOP")

            # 1. 軌道の開始点へ移動 (2回目以降もここを通ることでループがつながる)
            # ※初回や遠い場合はゆっくり(3.0s)、近くなら速く移動する制御も可能だが、一旦安全のため3.0s固定
            self.ri.angle_vector(self.av_seq[0], 3.0)
            self.ri.wait_interpolation()

            # 2. 軌道実行
            self.ri.angle_vector_sequence(self.av_seq, times=self.times)

            # 3. 実行中のキャンセル監視
            while not rospy.is_shutdown() and self.ri.is_interpolating():
                if self.current_button_state == 1:
                    rospy.loginfo("Button 1 pressed -> Canceling motion")
                    self.ri.cancel_angle_vector()
                    self.current_button_state = 0
                    is_canceled = True
                    break
                time.sleep(0.05)

            if is_canceled:
                break

            loop_count += 1

        # --- 終了処理 ---
        if is_canceled:
            rospy.loginfo("Play interrupted by user.")
            self.update_display("STOPPED")
        else:
            rospy.loginfo("All sequences finished. Returning to home pose...")
            self.update_display("Back to\nStart")
            # 最後に元の姿勢(Home)に戻る
            self.ri.angle_vector(home_av, 3.0)
            self.ri.wait_interpolation()
            self.update_display("Done")
            rospy.loginfo("Motion Finished.")

    def run(self):
        rospy.loginfo("Task Node Ready. Waiting for commands...")
        self.update_display("Wait Task...")

        while not rospy.is_shutdown():
            # ==========================================
            # 1. サービスからのリクエスト処理 (自動実行)
            # ==========================================
            if self.requested_from_service:
                self.requested_from_service = False
                rospy.loginfo(">>> Starting Service Requested Sequence <<<")

                # シードが設定されていない場合はエラーで弾く
                if self.manual_seed_av is None:
                    rospy.logerr("Cannot start auto task: Seed AV is not set.")
                    self.update_display("Err:No Seed")
                    time.sleep(1.0)
                    continue

                # Vision認識 & 計画 (成功すればTrue)
                success = self.teach_corners_vision(long_side_stroke=True)

                # 計画があれば実行
                if success and self.av_seq:
                    self.execute_motion()
                    self.update_display("Auto Task\nFinished")
                else:
                    rospy.logwarn("Auto Task Failed (Vision or IK Error)")
                    self.update_display("Task Fail")

                time.sleep(1.0)
                # 自動実行後はループ先頭に戻り、次の指示やボタン入力を待つ
                continue

            # ==========================================
            # 2. ユーザーインターフェース (ボタン待ち)
            # ==========================================
            self.update_display("Wait for task\n1:Manual\n2:Set seed\n3:Servo ON/OFF")
            btn = self.wait_for_button_press(valid_buttons=[1, 2, 3], timeout=0.5)
            if btn == 1:
                # マニュアルモード (手動教示 or 手動再生)
                self.manual_mode_entry()
            elif btn == 2:
                # Vision用 IKシード設定
                self.set_vision_seed()
            elif btn == 3:
                # サーボ ON/OFF 切り替え
                self.toggle_servo_on_off()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default="")
    args, _unknown = parser.parse_known_args()

    rospy.init_node("corner_teaching_task", anonymous=True)

    # 1. ロボットセットアップ
    ri, robot_model = setup_robot(args.namespace)

    # 2. タスククラスの起動
    # 初期状態ではタスク(prompt)は空で起動し、サービス待ちorボタン待ちになります
    task = CornerTeachingTask(ri, robot_model)
    task.run()


if __name__ == "__main__":
    main()
