import rospy
import cv2
import numpy as np
import torch
import threading
import io
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ROS Messages & Services
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from riberry_startup.srv import VisualPose, VisualPoseResponse
from cv_bridge import CvBridge

# Sync
from message_filters import ApproximateTimeSynchronizer, Subscriber

# External Libraries
from transformers import AutoProcessor, AutoModelForCausalLM

# =============================================================================
# Helper Class: Florence-2 Segmenter (変更なし・軽量化)
# =============================================================================

class Florence2Segmenter:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        rospy.loginfo("Loading Florence-2 Model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base", 
                torch_dtype=self.torch_dtype, 
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", 
                trust_remote_code=True
            )
        except Exception as e:
            rospy.logerr(f"Failed to load Florence-2: {e}")
            raise e
        
        rospy.loginfo("Florence-2 Loaded.")

    def process_image(self, image_input, prompt):
        image = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
        task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
        text_input = task_prompt + prompt

        try:
            inputs = self.processor(text=text_input, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"], 
                max_new_tokens=1024, 
                num_beams=3, 
                do_sample=False
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            result = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.shape[1], image.shape[0])
            )
            return result
        except Exception as e:
            rospy.logerr(f"Segmentation failed: {e}")
            return None

    def create_mask(self, parsed_answer, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if not parsed_answer: return mask
        results = parsed_answer.get("<REFERRING_EXPRESSION_SEGMENTATION>", {})
        
        # 検出されたポリゴンを全てマスクとして塗りつぶす
        for polygon_group in results.get('polygons', []):
            for coords in polygon_group:
                points = np.array(coords, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], 255)
        return mask

class VisualPoseEstimator:
    def __init__(self):
        # --- Parameters ---
        self.visualize = rospy.get_param("~visualize", True)
        self.bridge = CvBridge()
        
        # Load Model
        self.segmenter = Florence2Segmenter()
        
        # --- Data Buffers ---
        self.lock = threading.Lock()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info_K = None
        self.latest_timestamp = rospy.Time(0)

        # --- Subscribers ---
        self.color_sub = Subscriber("/decompressed/camera/color/image_raw", Image)
        self.depth_sub = Subscriber("/decompressed/camera/aligned_depth_to_color/image_raw", Image)
        self.info_sub_color = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_cb)
        
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=2, 
            slop=0.2
        )
        self.sync.registerCallback(self.image_cb)

        # --- Publishers ---
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)

        # --- Services ---
        # 既存のVisualPose.srvを使用するが、実質使うのは prompt と calculate_offset のみ
        rospy.Service("calculate_offset", VisualPose, self.handle_get_position)
        
        rospy.loginfo("Simple Object Locator is READY.")

    def info_cb(self, msg):
        if self.camera_info_K is None:
            self.camera_info_K = np.array(msg.K, dtype=np.float64).reshape((3, 3))
            rospy.loginfo("Camera Info Received.")

    def image_cb(self, color_msg, depth_msg):
        with self.lock:
            try:
                self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
                self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                self.latest_timestamp = color_msg.header.stamp
            except Exception as e:
                rospy.logerr(f"Image conversion error: {e}")

    def capture_snapshot(self):
        """最新の画像を取得する"""
        with self.lock:
            if self.latest_color is not None and self.latest_depth is not None:
                return self.latest_color.copy(), self.latest_depth.copy()
        return None, None

    def handle_get_position(self, req):
        """
        プロンプトを受け取り、対象物の3次元位置(X, Y, Z)を返す。
        """
        target_prompt = req.prompt
        if not target_prompt:
            target_prompt = "object"
            
        rospy.loginfo(f"Request received for: '{target_prompt}'")
        
        # 1. 画像取得
        color, depth = self.capture_snapshot()
        if color is None or self.camera_info_K is None:
            return VisualPoseResponse(success=False, message="No image or camera info", pose=Pose())

        # 2. セグメンテーション (Florence-2)
        res = self.segmenter.process_image(color, target_prompt)
        mask = self.segmenter.create_mask(res, color.shape)
        
        # マスクが見つかったか確認
        if np.count_nonzero(mask) == 0:
             return VisualPoseResponse(success=False, message=f"Object '{target_prompt}' not found.", pose=Pose())

        # 3. 3次元座標の計算 (Depth + Intrinsics)
        position, message = self.calculate_3d_centroid(mask, depth, self.camera_info_K)
        
        if position is None:
            return VisualPoseResponse(success=False, message=message, pose=Pose())

        x, y, z = position
        
        # 4. レスポンス作成
        pose_msg = Pose()
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = z
        # 姿勢は単位クォータニオン (回転なし)
        pose_msg.orientation.w = 1.0

        msg_str = f"Found '{target_prompt}': X={x*1000:.1f}, Y={y*1000:.1f}, Z={z*1000:.1f} mm"
        rospy.loginfo(msg_str)

        # 5. 可視化
        self.publish_debug_image(color, mask, target_prompt, (x, y, z))

        return VisualPoseResponse(success=True, message=msg_str, pose=pose_msg)

    def calculate_3d_centroid(self, mask, depth_img, K):
        """
        マスク領域内のデプス情報から物体の中心位置(Camera座標系)を計算する
        """
        # マスク領域のデプス値を取得
        masked_depth = depth_img[mask > 0]
        
        # 0 (無効値) を除外
        valid_depth = masked_depth[masked_depth > 0]
        
        if len(valid_depth) < 10:
            return None, "Not enough valid depth points"

        # 外れ値の影響を避けるため、中央値(Median)を使用するのが一般的
        z_mm = np.median(valid_depth)
        z = z_mm * 0.001  # mm -> meter

        if z <= 0.0:
            return None, "Invalid depth calculated"

        # マスクの重心 (u, v) を計算
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None, "Mask moment error"
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # カメラ座標系への逆投影
        # X = (u - cx_cam) * Z / fx
        # Y = (v - cy_cam) * Z / fy
        fx, fy = K[0, 0], K[1, 1]
        cx_cam, cy_cam = K[0, 2], K[1, 2]

        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy
        
        return (x, y, z), "Success"

    def publish_debug_image(self, color, mask, prompt, pos):
        if not self.visualize:
            return

        # マスクをオーバーレイ
        vis_img = color.copy()
        
        # 緑色のマスク
        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask > 0] = [0, 255, 0]
        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

        # 重心をプロット
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis_img, (cx, cy), 10, (0, 0, 255), -1)
            
            # テキスト表示
            x, y, z = pos
            text = f"{prompt}: ({x:.2f}, {y:.2f}, {z:.2f})m"
            cv2.putText(vis_img, text, (cx - 50, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        try:
            msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            self.debug_pub.publish(msg)
        except Exception as e:
            rospy.logwarn(f"Debug pub failed: {e}")
