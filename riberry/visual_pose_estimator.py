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

# Florence2Segmenterクラスは変更なしのため省略 (そのまま使用してください)
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
        # ★変更点: サービス名を calculate_offset から estimate_cleaning_zone に変更
        rospy.Service("estimate_cleaning_zone", VisualPose, self.handle_get_position)

        rospy.loginfo("Visual Pose Estimator (Cleaning Zone) is READY.")

    def info_cb(self, msg):
        if self.camera_info_K is None:
            self.camera_info_K = np.array(msg.K, dtype=np.float64).reshape((3, 3))

    def image_cb(self, color_msg, depth_msg):
        with self.lock:
            try:
                self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
                self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            except Exception as e:
                rospy.logerr(f"Image conversion error: {e}")

    def capture_snapshot(self):
        with self.lock:
            if self.latest_color is not None and self.latest_depth is not None:
                return self.latest_color.copy(), self.latest_depth.copy()
        return None, None

    # =========================================================================
    # Service Handler
    # =========================================================================
    def handle_get_position(self, req):
        """
        req.prompt: 検出対象 ("table", "desk" etc)
        req.mode: "center", "corners"
        """
        target_prompt = req.prompt if req.prompt else "object"
        mode = req.mode if req.mode else "center"

        rospy.loginfo(f"Request: '{target_prompt}', Mode: '{mode}'")

        # 1. 画像取得
        color, depth = self.capture_snapshot()
        if color is None or self.camera_info_K is None:
            return VisualPoseResponse(success=False, message="No image/camera info", poses=[])

        # 2. セグメンテーション (マスク作成)
        res = self.segmenter.process_image(color, target_prompt)
        mask = self.segmenter.create_mask(res, color.shape)

        if np.count_nonzero(mask) == 0:
             return VisualPoseResponse(success=False, message=f"Object '{target_prompt}' not found.", poses=[])

        # 3. モード別処理
        points_3d = []  # [(x,y,z), ...]
        msg = ""
        success = False
        
        # 描画用データ保持のため
        self.last_approx_corners = None 

        if mode == "center":
            pt, msg = self.calculate_center(mask, depth)
            if pt:
                points_3d = [pt]
                success = True

        elif mode == "corners":
            # ★変更点: cornersモードの計算ロジックを刷新
            pts, approx_poly, msg = self.calculate_corners(mask, depth)
            if pts:
                points_3d = pts
                self.last_approx_corners = approx_poly # 可視化用
                success = True

        else:
            return VisualPoseResponse(success=False, message=f"Unknown mode: {mode}", poses=[])

        if not success:
             return VisualPoseResponse(success=False, message=msg, poses=[])

        # 4. レスポンス作成
        pose_list = []
        for (x, y, z) in points_3d:
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = z
            p.orientation.w = 1.0 
            pose_list.append(p)

        pts_str = ", ".join([f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in points_3d])
        rospy.loginfo(f"Result [{mode}]: {pts_str}")

        # 5. 可視化
        self.publish_debug_image(color, mask, points_3d, mode, self.last_approx_corners)

        return VisualPoseResponse(success=True, message=f"Found {len(points_3d)} points.", poses=pose_list)

    # =========================================================================
    # Calculation Logic
    # =========================================================================

    def get_representative_depth(self, mask, depth_img):
        """
        マスク領域全体の深度中央値を取得（フォールバック用）
        """
        masked_depth = depth_img[mask > 0]
        valid_depth = masked_depth[masked_depth > 0]

        if len(valid_depth) < 10:
            return None

        z_mm = np.median(valid_depth)
        return z_mm * 0.001 # mm to meters

    def project_pixel_to_3d(self, u, v, z):
        fx, fy = self.camera_info_K[0, 0], self.camera_info_K[1, 1]
        cx_cam, cy_cam = self.camera_info_K[0, 2], self.camera_info_K[1, 2]

        x = (u - cx_cam) * z / fx
        y = (v - cy_cam) * z / fy
        return (x, y, z)

    def calculate_center(self, mask, depth_img):
        z = self.get_representative_depth(mask, depth_img)
        if z is None or z <= 0:
            return None, "Invalid depth"

        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None, "Moment error"

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        point_3d = self.project_pixel_to_3d(cx, cy, z)
        return point_3d, "Success"

    def calculate_corners(self, mask, depth_img):
        """
        ★変更: 長方形近似ではなく、多角形近似(approxPolyDP)を使用。
        4点（台形形状）のそれぞれの座標で深度を取得して3次元化する。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, "No contours found"

        largest_contour = max(contours, key=cv2.contourArea)

        # 多角形近似 (周長の2%を誤差許容範囲とする)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        # approx shape is (N, 1, 2)

        # 深度のフォールバック値（コーナーの深度が抜けている場合に使用）
        fallback_z = self.get_representative_depth(mask, depth_img)
        if fallback_z is None: fallback_z = 0.0

        points_3d = []
        
        # 検出された各頂点について処理
        for point in approx:
            u, v = point[0]

            # 画像外にはみ出さないようクリップ
            v_safe = int(np.clip(v, 0, depth_img.shape[0] - 1))
            u_safe = int(np.clip(u, 0, depth_img.shape[1] - 1))

            # そのピクセルの深度を取得 (単位: mm と仮定 -> mへ)
            d_val = depth_img[v_safe, u_safe]
            z = d_val * 0.001

            # 深度が無効(0)の場合は、物体全体の代表深度(median)を使う
            if z <= 0.001:
                z = fallback_z

            if z > 0:
                pt_3d = self.project_pixel_to_3d(u, v, z)
                points_3d.append(pt_3d)
            else:
                # どうしても深度が取れない場合
                points_3d.append((0.0, 0.0, 0.0))

        return points_3d, approx, "Success"

    # =========================================================================
    # Visualization
    # =========================================================================
    def publish_debug_image(self, color, mask, points_3d, mode, approx_poly=None):
        if not self.visualize:
            return

        vis_img = color.copy()

        # マスクの半透明表示
        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask > 0] = [0, 255, 0]
        vis_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)

        if mode == "center" and points_3d:
             M = cv2.moments(mask)
             cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
             cv2.circle(vis_img, (cx, cy), 8, (0, 0, 255), -1)
             text = f"({points_3d[0][0]:.2f}, {points_3d[0][1]:.2f}, {points_3d[0][2]:.2f})m"
             cv2.putText(vis_img, text, (cx-60, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        elif mode == "corners" and points_3d and approx_poly is not None:
            # ★変更: 多角形を描画
            cv2.drawContours(vis_img, [approx_poly], -1, (0, 0, 255), 2)

            # 各頂点に3次元座標を表示
            for i, point in enumerate(approx_poly):
                u, v = point[0]
                if i < len(points_3d):
                    x, y, z = points_3d[i]
                    
                    # 点を描画
                    cv2.circle(vis_img, (u, v), 6, (255, 0, 0), -1) # 青丸
                    
                    # テキスト生成
                    label = f"P{i}: ({x:.2f}, {y:.2f}, {z:.2f})"
                    
                    # 文字が見やすいように縁取りと本体を描画
                    cv2.putText(vis_img, label, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                    cv2.putText(vis_img, label, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        try:
            msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            self.debug_pub.publish(msg)
        except Exception as e:
            rospy.logwarn(f"Debug pub failed: {e}")

if __name__ == "__main__":
    rospy.init_node("visual_pose_estimator")
    node = VisualPoseEstimator()
    rospy.spin()
