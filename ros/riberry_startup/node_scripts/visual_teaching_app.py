#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import threading
import os
import json
import datetime
import re

# riberry & skrobot
from riberry.filecheck_utils import get_cache_dir
from riberry.teaching_manager import TeachingManager
from riberry.visual_perception import VisualPerception
from skrobot.coordinates import Coordinates


class VisualTeachingApp:
    def __init__(self):
        rospy.init_node("visual_teaching_app", anonymous=True)
        
        # --- モジュールの初期化 ---
        self.manager = TeachingManager()
        self.perception = VisualPerception()
        
        # --- ディレクトリ設定 ---
        cache_root = get_cache_dir()
        self.save_dir = os.path.join(cache_root, "visual_teach_data")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 再生時の一時ファイル用パス
        self.temp_play_file = os.path.join(self.save_dir, "latest_playback_temp.json")

        rospy.loginfo("Visual Teaching App Ready.")

    # =========================================================================
    #  Core Logic: Trajectory Adjustment
    # =========================================================================

    def calculate_adjusted_motion(self, original_motion, ref_coords_dict, current_coords_skrobot):
        """
        教示時の軌道と位置関係、現在の位置関係から、補正後の軌道を計算する。
        現在はパススルー（補正なし）だが、ここに入力される情報は
        「物体が何であったか」に依存せず、純粋な座標情報のみである。
        
        Args:
            original_motion (list): 教示時のモーションデータ
            ref_coords_dict (dict): 教示時の物体位置 {x, y, z} (Base frame)
            current_coords_skrobot (Coordinates): 現在の物体位置 (Base frame)

        Returns:
            list: 補正後のモーションデータ
        """
        rospy.loginfo("Calculating trajectory correction...")

        # 将来の実装イメージ:
        # T_ref = ... (from ref_coords_dict)
        # T_curr = current_coords_skrobot
        # T_diff = T_curr * T_ref^-1
        # for point in motion: point.transform(T_diff)
        
        # 現時点ではそのまま返す
        adjusted_motion = original_motion
        return adjusted_motion

    # =========================================================================
    #  File & Path Management
    # =========================================================================

    def _get_action_filepath(self, action_name):
        """動作名からファイルパスを生成 (英数字以外は_に置換)"""
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', action_name).strip('_')
        filename = f"{safe_name}.json"
        return os.path.join(self.save_dir, filename)

    def _check_action_exists(self, action_name):
        """指定された動作名のファイルが存在するか確認"""
        filepath = self._get_action_filepath(action_name)
        return os.path.exists(filepath), filepath

    def _save_temp_motion(self, motion_data, meta_data):
        """再生用に一時ファイルへ保存"""
        save_data = {
            "meta": meta_data,
            "motion": motion_data
        }
        with open(self.temp_play_file, 'w') as f:
            json.dump(save_data, f, indent=4)
        return self.temp_play_file

    def _load_json_data(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('motion', []), data.get('meta', {})

    # =========================================================================
    #  Task Flow: Teaching
    # =========================================================================

    def run_teaching_mode(self):
        print("\n=== Teaching Mode ===")
        
        # 1. 動作名 (Action Name) の決定
        #    これがファイル名になるため、重複をチェックする
        while True:
            try:
                action_name = input("Enter Action Name (unique ID, e.g. 'pick_cup'): ").strip()
                if not action_name: continue
                
                exists, filepath = self._check_action_exists(action_name)
                if exists:
                    print(f"WARNING: Action '{action_name}' already exists.")
                    overwrite = input("Overwrite? (y/n): ").strip().lower()
                    if overwrite == 'y':
                        break # 上書き許可
                    else:
                        print("Please choose another name.")
                else:
                    break # 新規作成
            except EOFError: return

        # 2. 対象物 (Target Object) の指定
        try:
            target_object = input("Enter Target Object to look for (e.g. 'cup'): ").strip()
            if not target_object: target_object = "default"
        except EOFError: return

        # 3. 認識 (教示時の基準位置を取得)
        print(f"\n[Perception] Detecting '{target_object}' for reference pose...")
        _, object_meta_info = self.perception.get_object_pose(target_object)
        
        if object_meta_info is None:
            print("Aborting: Object detection failed.")
            return

        # 4. 記録準備
        print(f"\n[Recording] Action: '{action_name}' (Target: {target_object})")
        self.manager.servo_off()
        input("Press [Enter] to START recording...")

        # 5. 記録実行
        # TeachingManagerにはファイルパスを渡す
        record_thread = threading.Thread(target=self.manager.record, args=(filepath,))
        record_thread.start()
        
        print("\n*** RECORDING... Move the robot! ***")
        input("Press [Enter] to STOP recording...")

        self.manager.stop()
        record_thread.join()

        # 6. メタデータの保存
        #    ここで action_name や 教示時の物体位置 を保存する
        #    プロンプト(target_object)はあくまで参考情報として残す
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 既存のメタ情報に追記
            object_meta_info["action_name"] = action_name
            object_meta_info["teaching_target_object"] = target_object # 記録用
            object_meta_info["created_at"] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            data['meta'] = object_meta_info
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Data saved to: {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"Error saving metadata: {e}")

    # =========================================================================
    #  Task Flow: Playback
    # =========================================================================

    def run_playback_mode(self):
        print("\n=== Playback Mode ===")
        
        # 1. 動作名 (Action Name) の指定
        try:
            action_name = input("Enter Action Name to play (e.g. 'pick_cup'): ").strip()
        except EOFError: return

        exists, filepath = self._check_action_exists(action_name)
        if not exists:
            print(f"Error: Action '{action_name}' not found.")
            return

        # 2. データ読み込み (教示データのロード)
        original_motion, meta_data = self._load_json_data(filepath)
        ref_coords_info = meta_data.get("position", None)

        if ref_coords_info is None:
            print("Error: No reference position data in this file.")
            return

        # 3. 対象物 (Target Object) の指定
        #    教示時と同じ物体である必要はない (例: 教示はcup, 再生はbottle)
        print(f"Loaded action '{action_name}'.")
        try:
            target_object = input("Enter Target Object to look for now (e.g. 'bottle'): ").strip()
            if not target_object: target_object = "default"
        except EOFError: return

        # 4. 認識 (現在の位置を取得)
        print(f"\n[Perception] Detecting '{target_object}'...")
        current_coords_skrobot, _ = self.perception.get_object_pose(target_object)
        
        if current_coords_skrobot is None:
            print("Aborting: Could not find the object.")
            return

        # 5. 軌道補正の計算
        #    入力: (軌道, 教示時の位置, 現在の位置) -> 出力: 補正後軌道
        #    ここにプロンプト名は関与しない（純粋な座標計算）
        adjusted_motion = self.calculate_adjusted_motion(
            original_motion, 
            ref_coords_info, 
            current_coords_skrobot
        )

        # 6. 一時ファイルへ保存して再生
        temp_path = self._save_temp_motion(adjusted_motion, meta_data)
        
        print(f"\n[Playback] Playing action '{action_name}' on target '{target_object}'...")
        result_msg = self.manager.play(temp_path)
        print(f"Result: {result_msg}")

    # =========================================================================
    #  Main Loop
    # =========================================================================

    def run(self):
        while not rospy.is_shutdown():
            print("\n========================================")
            print("   Visual Teaching App: Main Menu")
            print("========================================")
            print("[1] Teach (Record Action)")
            print("[2] Play  (Execute Action)")
            print("[3] Free  (Servo Off)")
            print("[q] Quit")
            
            try:
                choice = input("Select mode: ").strip()
            except EOFError: break

            if choice == '1':
                self.run_teaching_mode()
            elif choice == '2':
                self.run_playback_mode()
            elif choice == '3':
                print("Servo Off (Free Mode).")
                self.manager.servo_off()
            elif choice == 'q':
                break
            else:
                print("Invalid selection.")

if __name__ == "__main__":
    try:
        app = VisualTeachingApp()
        app.run()
    except rospy.ROSInterruptException:
        pass
