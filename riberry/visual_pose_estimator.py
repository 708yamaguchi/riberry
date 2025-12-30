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
            # --- 1. VQA用モデル ---
            # 存在確認(Yes/No)が得意なFine-Tunedモデル
            rospy.loginfo("Loading VQA Model ...")
            vqa_model_name = "microsoft/Florence-2-large-ft"
            self.model_vqa = AutoModelForCausalLM.from_pretrained(
                vqa_model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            self.processor_vqa = AutoProcessor.from_pretrained(
                vqa_model_name,
                trust_remote_code=True
            )
            # --- 2. Segmentation用モデル ---
            # 座標検出(Grounding)が得意なPre-Trainedモデル
            rospy.loginfo("Loading Segmentation Model ...")
            # seg_model_name = "microsoft/Florence-2-base"
            seg_model_name = "microsoft/Florence-2-large"
            self.model_seg = AutoModelForCausalLM.from_pretrained(
                seg_model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            self.processor_seg = AutoProcessor.from_pretrained(
                seg_model_name,
                trust_remote_code=True
            )
        except Exception as e:
            rospy.logerr(f"Failed to load Florence-2: {e}")
            raise e
        rospy.loginfo("Florence-2 Loaded.")

    def check_existence(self, image_input, target_name):
        """
        VQAタスクを使って物体が存在するかどうかを Yes/No で判定する
        """
        image = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)

        # VQA用のプロンプトを作成
        # 例: "Is there a screw?" "Are there screws?"
        task_prompt = "<VQA>"
        question = f"Is there a {target_name} in this image?"
        text_input = task_prompt + question

        try:
            inputs = self.processor_vqa(text=text_input, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

            generated_ids = self.model_vqa.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )

            generated_text = self.processor_vqa.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # 結果のパース
            result = self.processor_vqa.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.shape[1], image.shape[0])
            )

            # Florence-2のVQAは辞書形式ではなく、直接文字列を返すことが多いですが
            # post_processの結果に合わせて調整してください。
            # 通常は result['<VQA>'] に "Yes" や "No" が入ります。
            answer = result.get("<VQA>", "").lower()

            rospy.loginfo(f"VQA Check '{question}' -> '{answer}'")

            # "yes" が含まれていれば存在する
            return "yes" in answer

        except Exception as e:
            rospy.logerr(f"VQA failed: {e}")
            return False

    def process_image(self, image_input, prompt, task_prompt):
        """
        画像処理のメイン関数
        Args:
            task_prompt: "<OPEN_VOCABULARY_DETECTION>" or "<REFERRING_EXPRESSION_SEGMENTATION>"
        """
        image = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
        text_input = task_prompt + prompt

        try:
            inputs = self.processor_seg(text=text_input, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

            generated_ids = self.model_seg.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

            generated_text = self.processor_seg.batch_decode(generated_ids, skip_special_tokens=False)[0]

            result = self.processor_seg.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.shape[1], image.shape[0])
            )
            return result
        except Exception as e:
            rospy.logerr(f"Segmentation failed: {e}")
            return None

    def create_mask(self, parsed_answer, task_prompt, image_shape):
        """
        タスクの種類に応じて結果を解析し、マスクを作成する
        Args:
            parsed_answer: process_imageの戻り値 (辞書型)
            task_prompt: 実行したタスクのプロンプト (キーとして使用)
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if not parsed_answer: 
            return mask
        
        # 結果辞書から該当タスクのデータを取り出す
        results = parsed_answer.get(task_prompt, {})

        # --- Detection (OD) の場合 ---
        if task_prompt == "<OPEN_VOCABULARY_DETECTION>":
            bboxes = results.get('bboxes', [])
            for bbox in bboxes:
                # bbox: [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # --- Segmentation の場合 ---
        elif task_prompt == "<REFERRING_EXPRESSION_SEGMENTATION>":
            polygons = results.get('polygons', [])
            for polygon_group in polygons:
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
        self.latest_header = None
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
                self.latest_header = color_msg.header
            except Exception as e:
                rospy.logerr(f"Image conversion error: {e}")

    def capture_snapshot(self, timeout=3.0):
        # 1. 古いキャッシュを破棄する (これにより、次のコールバックが来るまでNoneになる)
        with self.lock:
            self.latest_color = None
            self.latest_depth = None
            self.latest_header = None

        # 2. 新しい画像が来るのを待つ
        start_time = time.time()
        rate = rospy.Rate(10) # 10Hzでチェック

        while (time.time() - start_time) < timeout:
            with self.lock:
                # 画像がセットされたか確認
                if self.latest_color is not None and self.latest_depth is not None:
                    # 確実にコピーして返す
                    return self.latest_color.copy(), self.latest_depth.copy(), self.latest_header

            # まだ来てなければ待つ（ロックを開放してからsleepすることが重要）
            rate.sleep()

        # 3. タイムアウトした場合
        rospy.logerr("Capture timed out: No new image received.")
        return None, None, None

# =========================================================================
    # Service Handler
    # =========================================================================
    def handle_get_position(self, req):
        """
        req.prompt: 検出対象 ("table", "desk" etc)
        req.mode: "center", "corners"
        """
        target_prompt = req.prompt if req.prompt else "object"
        mode = req.mode if req.mode else "corners"
        strategy = req.strategy if req.strategy else "segmentation" # デフォルト
        rospy.loginfo(f"Request: '{target_prompt}', Mode: '{mode}', Strategy: '{strategy}'")

        if "detect" in strategy.lower() or "od" in strategy.lower():
            task_prompt = "<OPEN_VOCABULARY_DETECTION>"
            task_display_name = "OD"
        else:
            task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
            task_display_name = "SEG"

        # 1. 画像取得
        color, depth, header = self.capture_snapshot()
        if color is None or self.camera_info_K is None or header is None:
            return VisualPoseResponse(success=False, message="No image/camera info", poses=[])

        # --- 1. まずVQAで存在確認 ---
        exists = self.segmenter.check_existence(color, target_prompt)
        if not exists:
             msg_str = f"Object '{target_prompt}' not found (VQA check)."
             rospy.logwarn(msg_str)
             empty_mask = np.zeros(color.shape[:2], dtype=np.uint8)
             self.publish_debug_image(
                 color, empty_mask, [], mode, header,
                 prompt_text=target_prompt, 
                 status_msg="Not Found (VQA)"
             )
             return VisualPoseResponse(success=False, message=msg_str, poses=[])

         # 4. モデル実行 (process_image に task_prompt を渡す)
        rospy.loginfo(f"Executing task: {task_display_name} ({task_prompt})")
        result = self.segmenter.process_image(color, target_prompt, task_prompt)
        mask = self.segmenter.create_mask(result, task_prompt, color.shape)

        if np.count_nonzero(mask) == 0:
             msg_str = f"Object '{target_prompt}' not found (Empty mask)."
             
             # 空のマスクですが画像を表示し、メッセージを表示してPublish
             self.publish_debug_image(
                 color, mask, [], mode, header,
                 prompt_text=target_prompt, 
                 status_msg="Not Found (Empty Mask)"
             )
             
             return VisualPoseResponse(success=False, message=msg_str, poses=[])

        # 3. モード別処理
        points_3d = []
        msg = ""
        success = False
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
                self.last_approx_corners = approx_poly
                success = True

        else:
            return VisualPoseResponse(success=False, message=f"Unknown mode: {mode}", poses=[])

        if not success:
             # 計算失敗時も画像を出したい場合はここにも追加可能
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

        # 5. 可視化 (成功時: status_msgは空または"Found"など)
        self.publish_debug_image(
            color, mask, points_3d, mode, header,
            approx_poly=self.last_approx_corners, 
            prompt_text=target_prompt,
            status_msg="Found (VQA)"
        )

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
        変更点: minAreaRect(長方形強制)をやめ、approxPolyDP(多角形近似)を使用して
        パースのついた台形や一般四角形として頂点を取得する。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, "No contours found"

        # 最大の領域を取得
        largest_contour = max(contours, key=cv2.contourArea)

        # --- 変更ここから ---

        # 1. 形状をきれいにするために凸包(Convex Hull)を取得
        # これにより、凹みノイズを除去し、綺麗な多角形にしやすくします
        hull = cv2.convexHull(largest_contour)

        # 2. 多角形近似を行い、頂点が4つになるパラメータを探す
        box_points = None

        # 許容誤差(epsilon)を少しずつ大きくしながら、ちょうど4点になる場所を探す
        # 周長の1%から始めて、最大10%まで試行
        for factor in np.linspace(0.01, 0.1, 10):
            epsilon = factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                # (N, 1, 2) -> (N, 2) に形状変換
                box_points = approx.reshape(-1, 2)
                break

        # もしうまく4点が見つからなかった場合(円形に近い、ノイズが多い等)のフォールバック
        if box_points is None:
            # 仕方がないので従来のminAreaRectを使う（あるいはエラーにする）
            rect = cv2.minAreaRect(largest_contour)
            box_points = np.int0(cv2.boxPoints(rect))

        # 頂点の順序を整列させる（オプション）
        # 左上、右上、右下、左下 の順などに揃えたい場合はここでソート処理を入れると良いです
        # 今回は検出順のまま進めます

        # --- 変更ここまで ---

        # 4. 3次元座標へ変換 (以降は元のコードと同じですが、box_pointsを使います)
        fallback_z = self.get_representative_depth(mask, depth_img)
        if fallback_z is None: fallback_z = 0.0

        points_3d = []

        for point in box_points:
            u, v = point

            # 画像外にはみ出さないようクリップ
            v_safe = int(np.clip(v, 0, depth_img.shape[0] - 1))
            u_safe = int(np.clip(u, 0, depth_img.shape[1] - 1))

            d_val = depth_img[v_safe, u_safe]
            z = d_val * 0.001

            if z <= 0.001 or (fallback_z > 0 and abs(z - fallback_z) > 0.5):
                z = fallback_z

            if z > 0:
                pt_3d = self.project_pixel_to_3d(u, v, z)
                points_3d.append(pt_3d)
            else:
                points_3d.append((0.0, 0.0, 0.0))

        # 描画用に box_points を返す (approx_polyとして使用)
        # box_points は numpy array なので、リスト構造などに直す必要はなくそのまま使えます
        return points_3d, box_points, "Success"

    # =========================================================================
    # Visualization
    # =========================================================================
    def publish_debug_image(self, color, mask, points_3d, mode, header_info, approx_poly=None, prompt_text="", status_msg=""):
        if not self.visualize:
            return

        vis_img = color.copy()

        # マスクの半透明表示（マスクがある場合のみ）
        if np.count_nonzero(mask) > 0:
            colored_mask = np.zeros_like(vis_img)
            colored_mask[mask > 0] = [0, 255, 0]
            vis_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)

        # --- テキスト描画 (プロンプト) ---
        text_str = f"Prompt: {prompt_text}"
        cv2.putText(vis_img, text_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(vis_img, text_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- テキスト描画 (ステータスメッセージ: Not Foundなど) ---
        if status_msg:
            # プロンプトの下(y=70あたり)に赤色で表示
            cv2.putText(vis_img, status_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(vis_img, status_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 赤

        # --- 成功時の描画 (座標など) ---
        if mode == "center" and points_3d:
             M = cv2.moments(mask)
             if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.circle(vis_img, (cx, cy), 8, (0, 0, 255), -1)
                text = f"({points_3d[0][0]:.2f}, {points_3d[0][1]:.2f}, {points_3d[0][2]:.2f})m"
                cv2.putText(vis_img, text, (cx-60, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        elif mode == "corners" and points_3d and approx_poly is not None:
            cv2.drawContours(vis_img, [approx_poly], -1, (0, 0, 255), 2)
            for i, point in enumerate(approx_poly):
                u, v = point.flatten()
                if i < len(points_3d):
                    x, y, z = points_3d[i]
                    cv2.circle(vis_img, (u, v), 6, (255, 0, 0), -1)
                    label = f"P{i}: ({x:.2f}, {y:.2f}, {z:.2f})"
                    cv2.putText(vis_img, label, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                    cv2.putText(vis_img, label, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        try:
            msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            msg.header = header_info
            self.debug_pub.publish(msg)
        except Exception as e:
            rospy.logwarn(f"Debug pub failed: {e}")


if __name__ == "__main__":
    rospy.init_node("visual_pose_estimator")
    node = VisualPoseEstimator()
    rospy.spin()
