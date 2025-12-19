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
                trust_remote_code=True,
                attn_implementation="eager"
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
                trust_remote_code=True,
                attn_implementation="eager"
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
        self.color_sub = Subscriber("decompressed/camera/color/image_raw", Image)
        self.depth_sub = Subscriber("decompressed/camera/depth/image_rect_raw", Image)
        self.info_sub_color = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.info_cb)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.2
        )
        self.sync.registerCallback(self.image_cb)

        # --- Publishers ---
        self.debug_pub = rospy.Publisher("visual_pose_estimator/debug_image", Image, queue_size=1)

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
        マスク領域全体の深度代表値を取得（フォールバック用）
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

    # def calculate_corners(self, mask, depth_img):
    #     """
    #     修正版: エッジノイズ対策（Inset & Median）を追加
    #     """
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return None, None, "No contours found"

    #     # 最大の領域を取得
    #     largest_contour = max(contours, key=cv2.contourArea)

    #     # 重心(Centroid)を計算 (Inset方向決定のため)
    #     M = cv2.moments(largest_contour)
    #     if M["m00"] == 0:
    #         cx, cy = 0, 0
    #     else:
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])
    #     center_arr = np.array([cx, cy])

    #     # --- 形状概略化 (元のコードのまま) ---
    #     hull = cv2.convexHull(largest_contour)
    #     box_points = None
    #     for factor in np.linspace(0.01, 0.1, 10):
    #         epsilon = factor * cv2.arcLength(hull, True)
    #         approx = cv2.approxPolyDP(hull, epsilon, True)
    #         if len(approx) == 4:
    #             box_points = approx.reshape(-1, 2)
    #             break

    #     if box_points is None:
    #         rect = cv2.minAreaRect(largest_contour)
    #         box_points = np.int0(cv2.boxPoints(rect))

    #     # --- 3次元座標変換 (修正箇所) ---

    #     # マスク全体の深度代表値 (フォールバック用)
    #     fallback_z = self.get_representative_depth(mask, depth_img)
    #     if fallback_z is None: fallback_z = 0.0

    #     points_3d = []

    #     # パラメータ設定
    #     # inset_dist = 5.0  # 重心方向に何ピクセル内側を見るか
    #     # kernel_r = 2      # 参照半径 (2なら5x5領域を見る)
    #     inset_dist = 1.0
    #     kernel_r = 5

    #     for point in box_points:
    #         u, v = point

    #         # 1. Inset処理: 重心方向へ座標をずらす
    #         # ベクトル計算
    #         vec = center_arr - point
    #         norm = np.linalg.norm(vec)

    #         if norm > 0:
    #             # 重心方向へ inset_dist 分だけ移動
    #             direction = vec / norm
    #             new_pt = point + direction * inset_dist
    #             u_in, v_in = int(new_pt[0]), int(new_pt[1])
    #         else:
    #             u_in, v_in = u, v

    #         # 画像範囲外チェック
    #         v_in = int(np.clip(v_in, 0, depth_img.shape[0] - 1))
    #         u_in = int(np.clip(u_in, 0, depth_img.shape[1] - 1))

    #         # 2. Area Sampling: 周辺領域の代表値を取得
    #         v_min = max(0, v_in - kernel_r)
    #         v_max = min(depth_img.shape[0], v_in + kernel_r + 1)
    #         u_min = max(0, u_in - kernel_r)
    #         u_max = min(depth_img.shape[1], u_in + kernel_r + 1)

    #         roi = depth_img[v_min:v_max, u_min:u_max]
    #         valid_depths = roi[roi > 0] # 0(欠損)を除外

    #         if len(valid_depths) > 0:
    #             # 小さい順（手前順）にソートして、ノイズを除いた「最も手前」に近い値を取る
    #             # 5パーセンタイル（下位5%の位置にある値）を採用
    #             d_val = np.percentile(valid_depths, 5)
    #         else:
    #             d_val = 0

    #         z = d_val * 0.001

    #         # 3. 外れ値除去 (閾値を0.5 -> 0.2に厳格化)
    #         # 角の深度が、物体全体の平均深度と20cm以上ずれていたら、平均深度を採用する
    #         if z <= 0.001 or (fallback_z > 0 and abs(z - fallback_z) > 0.2):
    #             z = fallback_z

    #         # 4. 投影 (X,Yは「元の角のピクセル(u,v)」を使い、Zは「内側で測った深度」を使う)
    #         if z > 0:
    #             pt_3d = self.project_pixel_to_3d(u, v, z)
    #             points_3d.append(pt_3d)
    #         else:
    #             points_3d.append((0.0, 0.0, 0.0))

    #     return points_3d, box_points, "Success"

    def calculate_corners(self, mask, depth_img):
        """
        修正版: NumPyによる軽量RANSACを用いて、外れ値に強い机の3次元平面推定を行う。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, "No contours found"

        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            cx, cy = 0, 0
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        center_arr = np.array([cx, cy])

        hull = cv2.convexHull(largest_contour)
        box_points = None
        for factor in np.linspace(0.01, 0.1, 10):
            epsilon = factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                box_points = approx.reshape(-1, 2)
                break

        if box_points is None:
            rect = cv2.minAreaRect(largest_contour)
            box_points = np.int0(cv2.boxPoints(rect))

        fallback_z = self.get_representative_depth(mask, depth_img)
        if fallback_z is None: fallback_z = 0.0

        # =========================================================
        # 机の3次元平面推定 (RANSAC Plane Fitting)
        # =========================================================
        table_plane = None  # (normal_vector, centroid)

        kernel_large = np.ones((50, 50), np.uint8)
        kernel_small = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel_large, iterations=1)
        inner_mask = cv2.dilate(mask, kernel_small, iterations=1)
        surround_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(inner_mask))

        ys, xs = np.where((surround_mask > 0) & (depth_img > 0))

        if len(ys) > 100:
            if len(ys) > 1000:
                indices = np.random.choice(len(ys), 1000, replace=False)
                ys, xs = ys[indices], xs[indices]

            zs = depth_img[ys, xs] * 0.001

            fx, fy = self.camera_info_K[0, 0], self.camera_info_K[1, 1]
            cx_cam, cy_cam = self.camera_info_K[0, 2], self.camera_info_K[1, 2]

            X = (xs - cx_cam) * zs / fx
            Y = (ys - cy_cam) * zs / fy
            points_3d_surround = np.column_stack((X, Y, zs))

            # --- RANSAC パラメータ ---
            num_iterations = 100     # 試行回数
            distance_threshold = 0.02 # インライアと判定する平面からの距離 (2cm)

            best_inliers_count = 0
            best_inliers_idx = None

            for _ in range(num_iterations):
                # 1. ランダムに3点を選択
                idx = np.random.choice(len(points_3d_surround), 3, replace=False)
                p1, p2, p3 = points_3d_surround[idx]

                # 2. 3点から法線ベクトルを計算
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)

                if norm < 1e-6: # 3点が一直線上にある場合はスキップ
                    continue
                normal = normal / norm

                # 3. 全点と平面の距離を計算: d = |(P - p1)・normal|
                vecs = points_3d_surround - p1
                distances = np.abs(np.dot(vecs, normal))

                # 4. 閾値内の点（インライア）をカウント
                inliers = distances < distance_threshold
                num_inliers = np.sum(inliers)

                if num_inliers > best_inliers_count:
                    best_inliers_count = num_inliers
                    best_inliers_idx = inliers

            # --- 最良のインライア群を用いてSVDで最終的な平面を再計算 ---
            if best_inliers_count > 50: # 最低50点はインライアが必要と定義
                inlier_points = points_3d_surround[best_inliers_idx]
                centroid = np.mean(inlier_points, axis=0)
                centered_points = inlier_points - centroid

                _, _, Vh = np.linalg.svd(centered_points)
                final_normal = Vh[2, :]

                # 法線がカメラ方向(-Z)を向くように調整
                if final_normal[2] > 0:
                    final_normal = -final_normal

                table_plane = (final_normal, centroid)
                plane_distance = np.abs(np.dot(final_normal, centroid))

                print("-" * 30)
                print(f"【机の平面検出: 成功】")
                print(f"  法線ベクトル(向き): x={final_normal[0]:.3f}, y={final_normal[1]:.3f}, z={final_normal[2]:.3f}")
                print(f"  カメラから平面までの最短距離: {plane_distance:.3f} m")
                print(f"  支持点数(Inliers): {best_inliers_count}点")
            else:
                print("【机の平面検出: 失敗】支持点不足によりフォールバックします。")

        # =========================================================
        # 3次元座標変換 (Ray-Plane Intersection)
        # =========================================================
        points_3d = []
        inset_dist = 1.0
        kernel_r = 5

        fx = self.camera_info_K[0, 0]
        fy = self.camera_info_K[1, 1]
        cx_cam = self.camera_info_K[0, 2]
        cy_cam = self.camera_info_K[1, 2]

        for point in box_points:
            u, v = point

            if table_plane is not None:
                normal, centroid = table_plane

                ray_x = (u - cx_cam) / fx
                ray_y = (v - cy_cam) / fy
                ray = np.array([ray_x, ray_y, 1.0])

                dot_product = np.dot(normal, ray)

                if abs(dot_product) > 1e-6:
                    z = np.dot(normal, centroid) / dot_product
                    if z > 0:
                        pt_3d = (ray[0] * z, ray[1] * z, z)
                        points_3d.append(pt_3d)
                        continue

            # RANSAC失敗時 or 交点計算失敗時のフォールバック処理
            vec = center_arr - point
            norm = np.linalg.norm(vec)

            if norm > 0:
                direction = vec / norm
                new_pt = point + direction * inset_dist
                u_in, v_in = int(new_pt[0]), int(new_pt[1])
            else:
                u_in, v_in = u, v

            v_in = int(np.clip(v_in, 0, depth_img.shape[0] - 1))
            u_in = int(np.clip(u_in, 0, depth_img.shape[1] - 1))

            v_min = max(0, v_in - kernel_r)
            v_max = min(depth_img.shape[0], v_in + kernel_r + 1)
            u_min = max(0, u_in - kernel_r)
            u_max = min(depth_img.shape[1], u_in + kernel_r + 1)

            roi = depth_img[v_min:v_max, u_min:u_max]
            valid_depths = roi[roi > 0]

            if len(valid_depths) > 0:
                d_val = np.percentile(valid_depths, 5)
            else:
                d_val = 0

            z = d_val * 0.001

            if z <= 0.001 or (fallback_z > 0 and abs(z - fallback_z) > 0.2):
                z = fallback_z

            if z > 0:
                pt_3d = self.project_pixel_to_3d(u, v, z)
                points_3d.append(pt_3d)
            else:
                points_3d.append((0.0, 0.0, 0.0))

        return points_3d, box_points, "Success"

    # =========================================================================
    # Visualization
    # =========================================================================
    def publish_debug_image(self, color, mask, points_3d, mode, header_info, approx_poly=None, prompt_text="", status_msg=""):
        if not self.visualize:
            return

        vis_img = color.copy()

        # 物体が見つかった場合：対象を緑色に強調し、それ以外を暗くする（既存の処理）
        if np.count_nonzero(mask) > 0:
            colored_mask = np.zeros_like(vis_img)
            colored_mask[mask > 0] = [0, 255, 0]
            highlighted_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)
            dark_bg = (vis_img * 1.0).astype(np.uint8)  # 今は背景を暗くしない
            # mask[:, :, None] で次元を合わせるのがポイントです
            vis_img = np.where(mask[:, :, None] > 0, highlighted_img, dark_bg)
        else:
            vis_img = (vis_img * 1.0).astype(np.uint8)  # 今は背景を暗くしない
            # 失敗時のみ、テキストを描画
            # --- テキスト描画 (プロンプト) ---
            if prompt_text != "":
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
                # text = f"({points_3d[0][0]:.2f}, {points_3d[0][1]:.2f}, {points_3d[0][2]:.2f})m"
                # cv2.putText(vis_img, text, (cx-60, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        elif mode == "corners" and points_3d and approx_poly is not None:
            cv2.drawContours(vis_img, [approx_poly], -1, (0, 0, 255), 2)
            for i, point in enumerate(approx_poly):
                u, v = point.flatten()
                if i < len(points_3d):
                    x, y, z = points_3d[i]
                    cv2.circle(vis_img, (u, v), 6, (255, 0, 0), -1)
                    # label = f"P{i}: ({x:.2f}, {y:.2f}, {z:.2f})"
                    # cv2.putText(vis_img, label, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                    # cv2.putText(vis_img, label, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

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
