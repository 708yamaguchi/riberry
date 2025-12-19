#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import quaternion2matrix

# Service & Messages
from geometry_msgs.msg import Pose
from riberry_startup.srv import VisualPose


class VisualPerception:
    """
    視覚認識サービスとTF変換を管理し、
    ロボット基準(Base)での物体位置を提供するクラス
    """
    def __init__(self, base_frame="base_link", camera_frame="camera_color_optical_frame"):
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.tf_listener = tf.TransformListener()
        
        # TFバッファが溜まるのを待機
        rospy.sleep(1.0)
        rospy.loginfo("VisualPerception initialized.")

    def get_object_pose(self, prompt, timeout=5.0):
        """
        指定されたプロンプトの物体を検出し、Base座標系での位置姿勢を返す。

        Args:
            prompt (str): 検出対象の名前
            timeout (float): サービスのタイムアウト時間

        Returns:
            tuple: (skrobot.coordinates.Coordinates, dict)
                   - 検出された位置姿勢 (Base基準)
                   - メタデータ用辞書 (JSON保存用)
                   検出失敗時は (None, None)
        """
        service_name = "calculate_offset"
        rospy.loginfo(f"Perception: Requesting detection for '{prompt}'...")

        try:
            rospy.wait_for_service(service_name, timeout=timeout)
            service_proxy = rospy.ServiceProxy(service_name, VisualPose)
            response = service_proxy(prompt)

            if not response.success:
                rospy.logwarn(f"Perception failed: {response.message}")
                return None, None

            # 1. カメラ座標系での位置 (VisualPoseの結果)
            #    回転は取得できないため、単位行列(回転なし)とする
            pos = response.pose.position
            co_cam_to_obj = Coordinates(
                pos=[pos.x, pos.y, pos.z], 
                rot=np.eye(3)
            )

            # 2. Base -> Camera の座標変換を取得
            self.tf_listener.waitForTransform(
                self.base_frame, self.camera_frame, 
                rospy.Time(0), rospy.Duration(3.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )

            # skrobot用回転行列に変換
            # tfは[x,y,z,w], skrobot utilityへは[w,x,y,z]を渡すのが一般的だが
            # quaternion2matrixの実装に合わせてリストを作成
            rot_matrix = quaternion2matrix([rot[3], rot[0], rot[1], rot[2]])
            co_base_to_cam = Coordinates(pos=trans, rot=rot_matrix)

            # 3. 座標変換: T(Base->Obj) = T(Base->Cam) * T(Cam->Obj)
            co_base_to_obj = co_base_to_cam.copy()
            co_base_to_obj.transform(co_cam_to_obj)

            world_pos = co_base_to_obj.worldpos()
            rospy.loginfo(f"Perception Result (Base Frame): {world_pos}")

            # 4. 保存用メタデータの作成
            meta_data = {
                "prompt": prompt,
                "frame_id": self.base_frame,
                "position": {
                    "x": float(world_pos[0]),
                    "y": float(world_pos[1]),
                    "z": float(world_pos[2])
                }
            }

            return co_base_to_obj, meta_data

        except Exception as e:
            rospy.logerr(f"Perception Error: {e}")
            return None, None
