#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import sys
import numpy as np
import rospy

from skrobot.model import RobotModel
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import midrot

from kxr_controller.check_ros_master import is_ros_master_local
from kxr_controller.kxr_interface import KXRROSRobotInterface


class CleaningTask(object):
    def __init__(self, ri, robot_model, target_link_name):
        self.ri = ri
        self.robot_model = robot_model

        self.corners = []      # 教示した4隅の座標 (Coordinates)
        self.corner_avs = []   # 教示した4隅の関節角度 (numpy array)
        self.av_seq = []       # 計算済みの関節角度列
        self.times = []        # 移動時間のリスト

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
            if link == self.robot_model:
                break
            if hasattr(link, 'joint') and link.joint:
                self.link_list.append(link)
            link = link.parent
        self.link_list.reverse()

        print(f"Target Link: {self.target_coords.name}")

    def teach_corners(self):
        """
        4隅を教示する関数 (関節角度も保存するように変更)
        """
        self.corners = []
        self.corner_avs = []

        print("\n=== Teaching Phase ===")
        print(f"Targeting: {self.target_coords.name}")
        print("Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")

        print("Turning Servo OFF for teaching...")
        self.ri.servo_off()
        time.sleep(1.0)

        for i in range(4):
            input(f"Move to Corner {i+1} and press Enter...")

            # 現在の実機の角度を取得
            current_av = self.ri.angle_vector()

            # モデルに反映して座標を取得
            self.robot_model.angle_vector(current_av)
            coord = self.target_coords.copy_worldcoords()

            # 座標と関節角度の両方を保存
            self.corners.append(coord)
            self.corner_avs.append(current_av)

            print(f"Captured Corner {i+1}: {coord.worldpos()}")

        print("Teaching finished. Turning Servo ON.")
        self.ri.servo_on()
        time.sleep(1.0)

        self.ri.angle_vector(self.robot_model.angle_vector())
        self.ri.wait_interpolation()

        self.compute_plan()

    def generate_zigzag_trajectory(self, corners, corner_avs, step_width=0.05):
        """
        4隅の座標と角度からジグザグ軌道（WaypointsとSeed角度）を生成する
        """
        # 座標の取り出し
        p1, p2 = corners[0].worldpos(), corners[1].worldpos()
        p3, p4 = corners[2].worldpos(), corners[3].worldpos()

        # 回転行列の取り出し
        r1, r2 = corners[0].worldrot(), corners[1].worldrot()
        r3, r4 = corners[2].worldrot(), corners[3].worldrot()

        # 関節角度(Seed用)の取り出し
        av1, av2 = corner_avs[0], corner_avs[1]
        av3, av4 = corner_avs[2], corner_avs[3]

        vec_advance = p4 - p1
        len_advance = np.linalg.norm(vec_advance)

        n_steps = int(len_advance / step_width)
        if n_steps < 1: n_steps = 1

        waypoints = []
        seed_avs = [] # IKの初期値リスト

        print(f"Generating trajectory with {n_steps} lines...")

        for i in range(n_steps + 1):
            ratio_adv = float(i) / n_steps # 進行方向(縦)の割合 0.0 -> 1.0

            # --- 1. 左端と右端の「座標」を補間 ---
            left_point = p1 + (p4 - p1) * ratio_adv
            right_point = p2 + (p3 - p2) * ratio_adv

            # --- 2. 左端と右端の「回転」を補間 (midrot使用) ---
            # 左列の回転: r1 -> r4
            left_rot = midrot(ratio_adv, r1, r4)
            # 右列の回転: r2 -> r3
            right_rot = midrot(ratio_adv, r2, r3)

            # --- 3. 左端と右端の「関節角度」を線形補間 ---
            left_av = av1 + (av4 - av1) * ratio_adv
            right_av = av2 + (av3 - av2) * ratio_adv

            # 往復動作の作成
            if i % 2 == 0: # 偶数行: 左 -> 右
                # Start (Left)
                waypoints.append(Coordinates(pos=left_point, rot=left_rot))
                seed_avs.append(left_av)

                # End (Right)
                waypoints.append(Coordinates(pos=right_point, rot=right_rot))
                seed_avs.append(right_av)
            else: # 奇数行: 右 -> 左
                # Start (Right)
                waypoints.append(Coordinates(pos=right_point, rot=right_rot))
                seed_avs.append(right_av)

                # End (Left)
                waypoints.append(Coordinates(pos=left_point, rot=left_rot))
                seed_avs.append(left_av)

        return waypoints, seed_avs

    def solve_full_ik(self, waypoints, seed_avs):
        """
        ウェイポイント列に対して、指定されたSeed角度を使ってIKを解く
        """
        print(f"Solving IK for {len(waypoints)} points...")
        av_sequence = []
        error_tolerance = 0.02

        for idx, wp in enumerate(waypoints):
            # IKを解く前に、教示点から補間した「理想的な姿勢」をロボットモデルにセットする
            target_seed = seed_avs[idx]
            self.robot_model.angle_vector(target_seed)

            # IK実行 (move_target等はinitで設定済みと仮定)
            result = self.robot_model.inverse_kinematics(
                target_coords=wp,
                link_list=self.link_list,
                move_target=self.target_coords,
                rotation_axis=True,
                stop=50, # シードが良いので計算回数は少なめで済むはず
                revert_if_fail=False
            )

            dist_err = np.linalg.norm(self.target_coords.worldpos() - wp.worldpos())
            is_success = (result is not False) and (result is not None)

            if is_success:
                av_sequence.append(self.robot_model.angle_vector())
            else:
                # 失敗時も誤差許容範囲内なら採用（またはSeedをそのまま採用する手もある）
                if dist_err < error_tolerance:
                    print(f"  Point {idx}: IK loose fit (Err: {dist_err*1000:.1f}mm)")
                    av_sequence.append(self.robot_model.angle_vector())
                else:
                    print(f"  Point {idx}: IK Failed. Error: {dist_err*1000:.1f}mm")
                    return None

        print("IK Solved successfully.")
        return av_sequence

    def compute_plan(self):
        """
        教示データをもとに軌道とIKを計算して保存する
        """
        if len(self.corners) != 4:
            print("[Error] No corners taught yet. Please Teach first.")
            return

        print("Planning trajectory...")

        # 1. 軌道生成
        waypoints, seed_avs = self.generate_zigzag_trajectory(
            self.corners,
            self.corner_avs,
            step_width=0.01
        )

        # 2. IK計算 (Seedを渡します)
        seq = self.solve_full_ik(waypoints, seed_avs)

        if seq is None:
            print("[Failed] Planning failed due to IK.")
            self.av_seq = []
            return

        self.av_seq = seq
        self.times = [1.0] * len(self.av_seq)
        print(f"[Success] Plan ready with {len(self.av_seq)} steps.")

    def execute_motion(self):
        """
        計算済みの軌道を実行する
        """
        if not self.av_seq:
            print("[Error] No plan available. Teach and Plan first.")
            return

        print("Executing motion...")
        self.ri.angle_vector_sequence(self.av_seq, times=self.times)
        self.ri.wait_interpolation()
        print("Motion Finished.")

    def interactive_run(self):
        """
        メインループ
        """
        while True:
            print("\n" + "="*40)
            print(" [t] Teach Corners (and plan)")
            print(" [p] Play / Re-execute Motion")
            print(" [s] Servo OFF (Manual Move)")
            print(" [o] Servo ON (Hold Position)")
            print(" [q] Quit")
            print("="*40)

            try:
                if sys.version_info[0] < 3:
                    cmd = raw_input("Command >> ").strip().lower()
                else:
                    cmd = input("Command >> ").strip().lower()
            except EOFError:
                break

            if cmd == 'q':
                print("Quitting...")
                break

            elif cmd == 't':
                self.teach_corners()

            elif cmd == 'p':
                self.execute_motion()

            elif cmd == 's':
                print("Servo OFF.")
                self.ri.servo_off()

            elif cmd == 'o':
                print("Servo ON.")
                self.ri.servo_on()
                # 念のため現在角度で保持
                self.ri.angle_vector(self.ri.angle_vector())
                self.ri.wait_interpolation()

            elif cmd == '':
                continue
            else:
                print("Unknown command.")


def main():
    parser = argparse.ArgumentParser(description="Cleaning Task with KXR")
    parser.add_argument("--namespace", type=str, help="Specify the ROS namespace", default="")
    args = parser.parse_args()

    rospy.init_node("kxr_cleaning_task", anonymous=True)

    # RobotModelロード
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

    # Interface準備
    ri = KXRROSRobotInterface(
        robot_model, namespace=args.namespace, controller_timeout=60.0
    )

    try:
        # クラス初期化
        task = CleaningTask(ri, robot_model, target_link_name='module5_base_link')

        # インタラクティブループ開始
        task.interactive_run()

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
