#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import sys
import numpy as np
import rospy
from std_msgs.msg import String, Int32

from skrobot.model import RobotModel
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import midrot

# KXRインターフェース
from kxr_controller.check_ros_master import is_ros_master_local
from kxr_controller.kxr_interface import KXRROSRobotInterface


class CleaningTask(object):
    def __init__(self, ri, robot_model, target_link_name):
        self.ri = ri
        self.robot_model = robot_model
        
        # --- ROS Control / Display用の変数 ---
        self.atom_mode = ""
        self.current_button_state = 0 
        
        # Atom S3への表示用Publisher
        self.pub_display = rospy.Publisher("/atom_s3_additional_info", String, queue_size=1)

        # ROSトピックの購読
        rospy.Subscriber("/atom_s3_mode", String, self._cb_atom_mode)
        rospy.Subscriber("/atom_s3_button_state", Int32, self._cb_atom_button)

        # 保存用変数の初期化
        self.corners = []      
        self.corner_avs = []   
        self.av_seq = []       
        self.times = []        

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
        """
        Atom S3のディスプレイにメッセージを送る
        読みやすさのため、最初に改行を入れる。
        """
        self.pub_display.publish('\n' + text)

    # --- コールバック関数 ---
    def _cb_atom_mode(self, msg):
        self.atom_mode = msg.data

    def _cb_atom_button(self, msg):
        # トピックの値をそのまま保存
        self.current_button_state = msg.data

    # --- ボタン入力待機関数 (シンプル版) ---
    def wait_for_button_press(self, valid_buttons=[1, 2, 3]):
        """
        指定されたボタン値が来るまでループで待つ。
        離されるのを待つ処理（リリース待ち）は行わない。
        """
        # ループ前に一度リセットして、古い入力を拾わないようにする
        self.current_button_state = 0
        
        while not rospy.is_shutdown():
            # 1. モードチェック
            if self.atom_mode == "DisplayInformationMode":
                
                # 2. ボタン値チェック
                btn = self.current_button_state
                
                if btn in valid_buttons:
                    rospy.loginfo(f"Button {btn} accepted.")
                    # 入力を一度受け取ったらリセットしてループを抜ける
                    self.current_button_state = 0 
                    return btn
            
            # CPU負荷軽減
            time.sleep(0.05)
        
        return None

    def teach_corners(self):
        self.corners = [] 
        self.corner_avs = []
        
        # 教示モード開始：サーボOFFにする
        rospy.loginfo("Start Teaching: Servo OFF")
        self.ri.servo_off()
        
        for i in range(4):
            # 画面表示更新
            # コーナー番号と、決定ボタン(1)を案内
            msg = f"Pos {i+1}/4\n1:Set"
            self.update_display(msg)
            
            rospy.loginfo(f"Waiting for Corner {i+1} (Button 1)...")
            
            # ボタン1が来るのを待つ
            btn = self.wait_for_button_press(valid_buttons=[1])
            if btn is None: return # shutdown時

            # 座標取得
            current_av = self.ri.angle_vector()
            self.robot_model.angle_vector(current_av)
            coord = self.target_coords.copy_worldcoords()
            
            self.corners.append(coord)
            self.corner_avs.append(current_av)
            rospy.loginfo(f"Captured Corner {i+1}")

        # 教示完了
        self.update_display("Done\nWait")
        rospy.loginfo("Teaching finished. Servo ON.")
        self.ri.servo_on()
        time.sleep(1.0)
        
        self.ri.angle_vector(self.robot_model.angle_vector())
        self.ri.wait_interpolation()

        self.compute_plan()

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
        
        rospy.loginfo(f"Generating trajectory: {n_steps} steps")

        waypoints = []
        seed_avs = []
        
        for i in range(n_steps + 1):
            ratio_adv = float(i) / n_steps
            left_point = p1 + (p4 - p1) * ratio_adv
            right_point = p2 + (p3 - p2) * ratio_adv
            left_rot = midrot(ratio_adv, r1, r4)
            right_rot = midrot(ratio_adv, r2, r3)
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

    def compute_plan(self):
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
        """
        メインループ
        """
        rospy.loginfo("Task Node Ready.")
        
        while not rospy.is_shutdown():
            # メインメニュー表示
            # 1: Teach, 2: Play, 3: Free
            menu_msg = "1:Teach\n2:Play\n3:Free"
            self.update_display(menu_msg)
            
            # ボタン入力を待つ (1, 2, 3 のいずれか)
            btn = self.wait_for_button_press(valid_buttons=[1, 2, 3])
            
            if btn == 1:
                # Teach Mode
                self.teach_corners()
                
            elif btn == 2:
                # Play Mode
                self.execute_motion()
                
            elif btn == 3:
                # Free Mode (Servo OFF)
                rospy.loginfo("Servo OFF (Free Mode)")
                self.update_display("Free\nMode")
                self.ri.servo_off()
                # ユーザーが次のアクションを起こすまでこの状態で待つ
                # ここでは単純に3秒待ってメニューに戻る（メニューに戻っても入力待機中はFreeのまま）
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
