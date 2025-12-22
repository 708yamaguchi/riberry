import time
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
        rospy.Service("estimate_corners", VisualPose, self.handle_get_position)

        rospy.loginfo("Visual Pose Estimator is READY.")

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

    def capture_snapshot(self, timeout=3.0):
        # 1. 古いキャッシュを破棄する (これにより、次のコールバックが来るまでNoneになる)
        with self.lock:
            self.latest_color = None
            self.latest_depth = None

        # 2. 新しい画像が来るのを待つ
        start_time = time.time()
        rate = rospy.Rate(10) # 10Hzでチェック

        while (time.time() - start_time) < timeout:
            with self.lock:
                # 画像がセットされたか確認
                if self.latest_color is not None and self.latest_depth is not None:
                    # 確実にコピーして返す
                    return self.latest_color.copy(), self.latest_depth.copy()
            
            # まだ来てなければ待つ（ロックを開放してからsleepすることが重要）
            rate.sleep()

        # 3. タイムアウトした場合
        rospy.logerr("Capture timed out: No new image received.")
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
        回転外接矩形(minAreaRect)を使用し、矩形らしさを判定。
        「拭き掃除」に適した、矩形度が高い領域のみ4点を返す。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, "No contours found"

        # 最大の領域を取得
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        # 1. 回転外接矩形を計算 (中心(x,y), (幅,高さ), 角度)
        rect = cv2.minAreaRect(largest_contour)
        (center, (w, h), angle) = rect
        box_area = w * h

        # 2. 基準判定: 「掃除する価値がある矩形か？」
        
        # A. 面積ゼロ除算回避
        if box_area <= 1e-5:
            return None, None, "Area too small"

        # B. 矩形度 (Rectangularity) チェック
        # マスクの面積が、外接矩形の面積の何割を占めるか。
        # 長方形に近い物体(テーブル等)なら0.8~0.9以上になります。
        # 0.6未満などは、L字型や複雑な凹凸形状である可能性が高く、単純な長方形スイープ動作には不向きです。
        rectangularity = contour_area / box_area
        if rectangularity < 0.6:  # 閾値は環境に合わせて調整 (0.6 ~ 0.7 推奨)
            return None, None, f"Shape not rectangular enough ({rectangularity:.2f})"

        # C. 最小サイズチェック
        # 短辺が極端に短い（細長い棒など）場合は除外
        min_side = min(w, h)
        if min_side < 20: # ピクセル単位。状況に応じて調整してください
             return None, None, "Object too thin"

        # 3. 4点の画像座標を取得
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points) # [[x,y], [x,y], [x,y], [x,y]]

        # 4. 3次元座標へ変換
        # コーナーの深度が抜けている場合に備え、物体全体の深度中央値を計算
        fallback_z = self.get_representative_depth(mask, depth_img)
        if fallback_z is None: fallback_z = 0.0

        points_3d = []
        
        for point in box_points:
            u, v = point

            # 画像外にはみ出さないようクリップ
            v_safe = int(np.clip(v, 0, depth_img.shape[0] - 1))
            u_safe = int(np.clip(u, 0, depth_img.shape[1] - 1))

            # そのピクセルの深度を取得
            d_val = depth_img[v_safe, u_safe]
            z = d_val * 0.001

            # 外接矩形のコーナーは、実際の物体の外側（背景）にある可能性が高いため、
            # 深度が急激に遠い、または0の場合は、物体の代表深度(fallback_z)で代用する処理を入れると安定します。
            # ここでは簡易的に「0なら代用」としていますが、
            # 「fallback_zとの差が大きすぎたら代用」とするとよりロバストです。
            if z <= 0.001 or (fallback_z > 0 and abs(z - fallback_z) > 0.5):
                z = fallback_z

            if z > 0:
                pt_3d = self.project_pixel_to_3d(u, v, z)
                points_3d.append(pt_3d)
            else:
                # どうしても深度が決まらない場合
                points_3d.append((0.0, 0.0, 0.0))

        # box_pointsは描画用(approx_poly)としてもそのまま使えます
        # points_3dは必ず4点になります
        return points_3d, box_points, "Success"

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
            cv2.drawContours(vis_img, [approx_poly], -1, (0, 0, 255), 2)
            # 各頂点に3次元座標を表示
            for i, point in enumerate(approx_poly):
                # u, v = point[0]
                u, v = point.flatten()
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
