#!/usr/bin/env python

from requests.exceptions import ChunkedEncodingError
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from dataclasses import dataclass
from kornia_moons.viz import draw_LAF_matches

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber


@dataclass
class MatchingResult:
    img1: torch.Tensor
    img2: torch.Tensor
    kps1: torch.Tensor
    kps2: torch.Tensor
    idxs: torch.Tensor
    inliers: np.ndarray


class FeatureMatcher:
    def __init__(self, feature_type="disk"):
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        # self.device = "cpu"  # "cuda:0"
        print(f"Using device: {self.device}")
        self.lg_matcher = KF.LightGlueMatcher(feature_type).eval().to(self.device)
        self.disk = KF.DISK.from_pretrained("depth").to(self.device)

    def process_images(self, img1_array, img2_array, num_features: int = 2048) -> MatchingResult:
        # Load RGB images as tensors
        img1 = K.image_to_tensor(img1_array, False).float() / 255.0  # (3, H, W)
        img2 = K.image_to_tensor(img2_array, False).float() / 255.0
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        hw1 = torch.tensor(img1.shape[2:], device=self.device)
        hw2 = torch.tensor(img2.shape[2:], device=self.device)
        
        with torch.inference_mode():
            inp = torch.cat([img1, img2], dim=0)
            features1, features2 = self.disk(inp, num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors
            lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=self.device))
            lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=self.device))
            dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        print(f"{idxs.shape[0]} tentative matches with DISK LightGlue")

        # findFundamentalMat は最低8点（推奨）必要です。0点だとクラッシュします。
        if idxs.shape[0] < 8:
            print("Not enough matches found. Skipping Fundamental Matrix estimation.")
            # 空の結果を返して、処理を継続できるようにする
            return MatchingResult(
                img1, img2, kps1, kps2, idxs, 
                np.zeros(idxs.shape[0], dtype=bool) # 全てFalse (inlierなし) とする
            )

        mkpts1, mkpts2 = kps1[idxs[:, 0]], kps2[idxs[:, 1]]

        # Estimate the fundamental matrix using RANSAC
        Fm, inliers = cv2.findFundamentalMat(
            mkpts1.detach().cpu().numpy(),
            mkpts2.detach().cpu().numpy(),
            cv2.USAC_MAGSAC,
            1.0,
            0.999,
            100000,
        )
        inliers = inliers.astype(bool)
        print(f"{inliers.sum()} inliers with DISK")

        return MatchingResult(img1, img2, kps1, kps2, idxs, inliers)

    def get_inlier_keypoints(self, result: MatchingResult):
        # Filter only inlier matches
        inliers_1d = result.inliers[:, 0].astype(bool)
        inlier_mkpts1 = result.kps1[result.idxs[:, 0][inliers_1d]]
        inlier_mkpts2 = result.kps2[result.idxs[:, 1][inliers_1d]]
        return inlier_mkpts1, inlier_mkpts2


class PoseEstimator:
    def __init__(self, rgb_camera_matrix: np.ndarray, depth_camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None):
        self.rgb_camera_matrix = rgb_camera_matrix
        self.depth_camera_matrix = depth_camera_matrix
        self.dist_coeffs = dist_coeffs

    def estimate_pose(self, inlier_mkpts1: torch.Tensor, inlier_mkpts2: torch.Tensor, depth1: np.ndarray):
        """
        PnP (SolvePnP) を使用して、3D点(Frame1)と2D点(Frame2)のマッチングからポーズを推定する
        """
        # テンソルをnumpyに変換
        pts1 = inlier_mkpts1.detach().cpu().numpy()
        pts2 = inlier_mkpts2.detach().cpu().numpy()

        # Depthカメラの内部パラメータ
        fx, fy = self.depth_camera_matrix[0, 0], self.depth_camera_matrix[1, 1]
        cx, cy = self.depth_camera_matrix[0, 2], self.depth_camera_matrix[1, 2]

        points_3d_frame1 = []
        points_2d_frame2 = []

        # Frame1のキーポイントをDepthを使って3D座標に変換
        h, w = depth1.shape
        for i, (u, v) in enumerate(pts1):
            ui, vi = int(round(u)), int(round(v))
            
            # 画面外参照のガード
            if not (0 <= ui < w and 0 <= vi < h):
                continue
                
            # 深度値の取得 (mm -> meter)
            z = depth1[vi, ui] * 0.001 
            
            # 有効な深度値のみを使用
            if z > 0.1:  # 0.1m以上のみ（ノイズ除去）
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d_frame1.append([x, y, z])
                points_2d_frame2.append(pts2[i])

        points_3d_frame1 = np.array(points_3d_frame1, dtype=np.float32)
        points_2d_frame2 = np.array(points_2d_frame2, dtype=np.float32)

        if len(points_3d_frame1) < 4:
            print("Not enough valid 3D points for PnP.")
            return np.eye(3), np.zeros(3)

        # SolvePnPRansac でポーズ推定 (単位はメートル)
        # 戻り値は Frame1 から Frame2 への変換
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d_frame1,
            points_2d_frame2,
            self.rgb_camera_matrix,
            self.dist_coeffs,
            iterationsCount=1000,
            reprojectionError=2.0,  # 許容誤差ピクセル数
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("PnP failed.")
            return np.eye(3), np.zeros(3)

        # 回転ベクトルを回転行列に変換
        R, _ = cv2.Rodrigues(rvec)
        
        return R, tvec

    def compute_translation_scale(
        self,
        inlier_mkpts1: torch.Tensor,
        inlier_mkpts2: torch.Tensor,
        depth1: np.ndarray,
        depth2: np.ndarray,
        R: np.ndarray,
        t_unit: np.ndarray
    ) -> float:
        # Intrinsic parameters from depth camera
        fx, fy = self.depth_camera_matrix[0, 0], self.depth_camera_matrix[1, 1]
        cx, cy = self.depth_camera_matrix[0, 2], self.depth_camera_matrix[1, 2]

        pts1 = inlier_mkpts1.detach().cpu().numpy()
        pts2 = inlier_mkpts2.detach().cpu().numpy()

        points3d_1 = []
        points3d_2 = []

        for (u1, v1), (u2, v2) in zip(pts1, pts2):
            u1i, v1i = int(round(u1)), int(round(v1))
            u2i, v2i = int(round(u2)), int(round(v2))
            if (
                0 <= v1i < depth1.shape[0] and 0 <= u1i < depth1.shape[1] and
                0 <= v2i < depth2.shape[0] and 0 <= u2i < depth2.shape[1]
            ):
                z1 = depth1[v1i, u1i] * 0.001  # convert from mm to meters
                z2 = depth2[v2i, u2i] * 0.001
                if z1 <= 0.0 or z2 <= 0.0:
                    continue
                x1 = (u1 - cx) * z1 / fx
                y1 = (v1 - cy) * z1 / fy
                x2 = (u2 - cx) * z2 / fx
                y2 = (v2 - cy) * z2 / fy
                points3d_1.append(np.array([x1, y1, z1]))
                points3d_2.append(np.array([x2, y2, z2]))

        if len(points3d_1) < 2:
            raise ValueError("Not enough valid 3D points for triangulation.")

        scales = []
        for p1, p2 in zip(points3d_1, points3d_2):
            p1_proj = R @ p1  # project p1 into the second camera's frame
            delta = p2 - p1_proj
            scale = np.dot(delta, t_unit.reshape(-1))  # project delta onto unit translation direction
            scales.append(scale)

        return np.median(scales)


def visualize_matches(result: MatchingResult, raw_img1=None, raw_img2=None, mask1=None, mask2=None):
    kps1 = result.kps1.cpu().numpy() if hasattr(result.kps1, 'cpu') else result.kps1
    kps2 = result.kps2.cpu().numpy() if hasattr(result.kps2, 'cpu') else result.kps2
    idxs = result.idxs.cpu().numpy() if hasattr(result.idxs, 'cpu') else result.idxs
    inliers = result.inliers.cpu().numpy() if hasattr(result.inliers, 'cpu') else result.inliers

    img1_masked_np = K.tensor_to_image(result.img1.cpu())
    img2_masked_np = K.tensor_to_image(result.img2.cpu())

    def blend_image_with_mask(masked_img_rgb, raw_img_bgr, mask, alpha=0.3):
        """
        masked_img_rgb: マスク処理済みのRGB画像 [0.0-1.0]
        raw_img_bgr: 元のBGR画像 [0-255] (OpenCV形式)
        mask: マスク画像 [0 or 255]
        alpha: マスク外の薄さ (0.0=非表示, 1.0=そのまま)
        """
        if raw_img_bgr is None or mask is None:
            return masked_img_rgb
            
        # 生画像を [0, 1] の RGB float に変換
        raw_img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # マスクを3チャンネルのブールマスクに変換 [H, W, 1]
        mask_3d = (mask[..., None] > 0)
        
        # 合成: マスク内は「マスク済み画像」、マスク外は「薄くした生画像」
        # np.where(条件, Trueの時の値, Falseの時の値)
        display_img = np.where(mask_3d, masked_img_rgb, raw_img_rgb * alpha)
        
        return np.clip(display_img, 0.0, 1.0)

    # 表示用の合成画像を作成
    display_img1 = blend_image_with_mask(img1_masked_np, raw_img1, mask1)
    display_img2 = blend_image_with_mask(img2_masked_np, raw_img2, mask2)

    # マスクフィルタリング
    if mask1 is not None and mask2 is not None and inliers is not None:
        h1, w1 = mask1.shape[:2]
        h2, w2 = mask2.shape[:2]
        valid_kps1 = {i for i, (x, y) in enumerate(kps1) if 0 <= x < w1 and 0 <= y < h1 and mask1[int(y), int(x)] != 0}
        valid_kps2 = {i for i, (x, y) in enumerate(kps2) if 0 <= x < w2 and 0 <= y < h2 and mask2[int(y), int(x)] != 0}
        valid_matches = [i for i, (idx1, idx2) in enumerate(idxs) if idx1 in valid_kps1 and idx2 in valid_kps2]
        filtered_inliers = np.zeros_like(inliers, dtype=bool)
        filtered_inliers[valid_matches] = inliers[valid_matches]
        
        if hasattr(result.inliers, 'copy_'):
            result.inliers = torch.from_numpy(filtered_inliers).to(result.kps1.device)
        else:
            result.inliers = filtered_inliers
    else:
        filtered_inliers = inliers

    # 前回のウィンドウが残っていると邪魔なので閉じる
    plt.close('all')

    # Figureを作成
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    
    # 2行2列のレイアウト
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

    # --- 上段: 元画像を表示 ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(display_img1)
    ax1.set_title("Input Image 1 (Raw)")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(display_img2)
    ax2.set_title("Input Image 2 (Raw)")
    ax2.axis('off')

    # --- 下段: マッチング結果を表示 ---
    ax_match = fig.add_subplot(gs[1, :])
    ax_match.set_title("Matches")

    # 描画設定
    draw_dict = {
        "inlier_color": (0.2, 1, 0.2),
        "tentative_color": (1, 1, 0.2, 0.3),
        "feature_color": None,
        "vertical": False,
    }

    try:
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(kps1[None])),
            KF.laf_from_center_scale_ori(torch.from_numpy(kps2[None])),
            torch.from_numpy(idxs),
            img1_masked_np,
            img2_masked_np,
            torch.from_numpy(filtered_inliers) if filtered_inliers is not None else None,
            draw_dict=draw_dict,
            ax=ax_match  # ここで描画先を指定
        )
    except TypeError:
        # 古いkornia_moonsで ax 引数がない場合のフォールバック
        # 一旦画像を連結して手動で表示してから描画コンテキストを合わせる
        plt.sca(ax_match)
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(kps1[None])),
            KF.laf_from_center_scale_ori(torch.from_numpy(kps2[None])),
            torch.from_numpy(idxs),
            img1_masked_np,
            img2_masked_np,
            torch.from_numpy(filtered_inliers) if filtered_inliers is not None else None,
            draw_dict=draw_dict,
        )

    # 軸をオフにする（draw_LAF_matchesがオンにする場合があるため最後に呼ぶ）
    ax_match.axis('off')

    plt.show()


def parse_camera_info(camera_info_msg):
    K = np.array(camera_info_msg.K, dtype=np.float64).reshape((3, 3))
    D = np.array(camera_info_msg.D, dtype=np.float64)
    return K, D


class CameraPoseEstimator:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_image1 = None
        self.depth_image1 = None
        self.color_image2 = None
        self.depth_image2 = None

        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.color_info_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.depth_info_callback)
        self.rgb_camera_matrix = None
        self.depth_camera_matrix = None
        self.dist_coeffs = None

        self.feature_matcher = FeatureMatcher()

        self.color_image_sub = Subscriber("/decompressed/camera/color/image_raw", Image)
        self.depth_image_sub = Subscriber("/decompressed/camera/aligned_depth_to_color/image_raw", Image)
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_timestamp = rospy.Time(0)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub],
            queue_size=3,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

    def color_info_callback(self, msg):
        if self.rgb_camera_matrix is None:
            self.rgb_camera_matrix, self.dist_coeffs = parse_camera_info(msg)
            rospy.loginfo("Received RGB Camera Intrinsics")

    def depth_info_callback(self, msg):
        if self.depth_camera_matrix is None:
            self.depth_camera_matrix, _ = parse_camera_info(msg)
            rospy.loginfo("Received Depth Camera Intrinsics")

    def image_callback(self, color_img_msg, depth_img_msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(color_img_msg, color_img_msg.encoding)
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(depth_img_msg, depth_img_msg.encoding)
            self.latest_timestamp = color_img_msg.header.stamp
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def wait_for_images(self, timeout):
        self.latest_color_image = None
        self.latest_depth_image = None
        timeout = rospy.Duration(timeout)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = rospy.Time.now() - start_time
            if elapsed > timeout:
                rospy.logwarn("Timeout while waiting for image")
                return False
            if self.latest_color_image is not None and self.latest_depth_image is not None:
                return True
            rospy.sleep(0.1)
        return False

    def capture_new_image(self, timeout=10.0):
        rospy.loginfo("Waiting for a FRESH image...")
        
        # 1. まず、現在の最新画像のタイムスタンプを基準として保存
        # (まだ画像が来ていない場合は 0 になっているので、とにかく何か来るまで待つ挙動になる)
        base_timestamp = self.latest_timestamp
        
        rate = rospy.Rate(20) # チェック頻度
        wait_start = rospy.Time.now()
        
        while not rospy.is_shutdown():
            # タイムアウト判定
            if (rospy.Time.now() - wait_start).to_sec() > timeout:
                rospy.logwarn("Timeout while waiting for new image.")
                return None, None
            # 「PCの現在時刻」ではなく、「さっきまでの画像の時刻(base_timestamp)」より
            # 新しいものが来たら採用する。これなら時計がズレていてもOK。
            if self.latest_timestamp > base_timestamp:
                if self.latest_color_image is not None and self.latest_depth_image is not None:
                    # 念のため、画像が空でないかチェック
                    if self.latest_color_image.size > 0:
                        return self.latest_color_image.copy(), self.latest_depth_image.copy()
            
            rate.sleep()
            
        return None, None
    
    def process_images(self, mask1=None, mask2=None):
        if any(x is None for x in (self.color_image1, self.color_image2, self.depth_image1, self.depth_image2)):
            rospy.logwarn("Missing images for processing.")
            return

        pose_estimator = PoseEstimator(self.rgb_camera_matrix, self.depth_camera_matrix, self.dist_coeffs)
        raw_color1 = self.color_image1.copy()
        raw_color2 = self.color_image2.copy()

        # Mask images
        color1 = self.color_image1.copy()
        color2 = self.color_image2.copy()
        depth1 = self.depth_image1.copy()
        depth2 = self.depth_image2.copy()
        if mask1 is not None:
            color1[mask1 == 0] = 0
            depth1[mask1 == 0] = 0
        if mask2 is not None:
            color2[mask2 == 0] = 0
            depth2[mask2 == 0] = 0
        
        result = self.feature_matcher.process_images(color1, color2)
        inlier_mkpts1, inlier_mkpts2 = self.feature_matcher.get_inlier_keypoints(result)
        # print(inlier_mkpts1)

        # マスク内にあるキーポイントのみを選択
        if mask1 is not None and mask2 is not None:
            kpts1_int = inlier_mkpts1.round().long().cpu().numpy()
            kpts2_int = inlier_mkpts2.round().long().cpu().numpy()
            h1, w1 = mask1.shape[:2]
            h2, w2 = mask2.shape[:2]
            valid_idx1 = [
                i for i, (x, y) in enumerate(kpts1_int)
                if 0 <= x < w1 and 0 <= y < h1 and mask1[y, x] != 0
            ]
            valid_idx2 = [
                i for i, (x, y) in enumerate(kpts2_int)
                if 0 <= x < w2 and 0 <= y < h2 and mask2[y, x] != 0
            ]
            valid_idx = list(set(valid_idx1) & set(valid_idx2))
            filtered_mkpts1 = inlier_mkpts1[valid_idx]
            filtered_mkpts2 = inlier_mkpts2[valid_idx]
            print(f"Filtered keypoints: {len(filtered_mkpts1)}/{len(inlier_mkpts1)}")
        else:
            filtered_mkpts1 = inlier_mkpts1
            filtered_mkpts2 = inlier_mkpts2

        if len(filtered_mkpts1) >= 4:
            # depth1 を渡すのが重要
            R, t = pose_estimator.estimate_pose(filtered_mkpts1, filtered_mkpts2, depth1)
            
            print("Rotation Matrix (R):\n", R)
            print("Translation Vector (t) [meters]:\n", t)
            
            # ノルム（移動距離）を計算
            distance = np.linalg.norm(t)
            print(f"Total Movement: {distance * 1000:.2f} [mm]")
        else:
            print("Not enough inliers for pose estimation")

        visualize_matches(result, raw_img1=raw_color1, raw_img2=raw_color2, mask1=mask1, mask2=mask2)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Florence2Segmenter:
    def __init__(self, max_retries=3, retry_delay=2):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._initialize_model()

    def _initialize_model(self):
        for attempt in range(self.max_retries):
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
                break
            except ChunkedEncodingError:
                if attempt == self.max_retries - 1:
                    raise
                rospy.sleep(self.retry_delay)

    def process_image(self, image_input, prompt):
        image = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
        for attempt in range(self.max_retries):
            try:
                inputs = self.processor(
                    text=f"<REFERRING_EXPRESSION_SEGMENTATION>{prompt}",
                    images=image,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=4096,
                    num_beams=3,
                    do_sample=False
                )

                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                return self.processor.post_process_generation(
                    generated_text,
                    task="<REFERRING_EXPRESSION_SEGMENTATION>",
                    image_size=(image.shape[1], image.shape[0])  # width, height
                ), image
                
            except ChunkedEncodingError:
                if attempt == self.max_retries - 1:
                    raise
                rospy.sleep(self.retry_delay)

    def create_mask(self, parsed_answer, image_shape):
        """セグメンテーション領域を示すマスクを生成 (領域内=255, 領域外=0)"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        task_key = "<REFERRING_EXPRESSION_SEGMENTATION>"

        if task_key in parsed_answer:
            results = parsed_answer[task_key]
            for polygon_group in results.get('polygons', []):
                for coords in polygon_group:
                    if len(coords) >= 6 and len(coords) % 2 == 0:
                        points = np.array([(coords[j], coords[j+1]) for j in range(0, len(coords), 2)], dtype=np.int32)
                        cv2.fillPoly(mask, [points], color=255)
        return mask


if __name__ == "__main__":
    rospy.init_node("camera_pose_estimator", anonymous=True)

    prompt_param = rospy.get_param("~prompt", "")
    if prompt_param and prompt_param.strip():
        mask_prompt = prompt_param
        rospy.loginfo(f"Segmentation enabled. Prompt: '{mask_prompt}'")
    else:
        mask_prompt = None
        rospy.loginfo("Segmentation disabled (no prompt provided).")
    
    try:
        t = threading.Thread(target=rospy.spin, daemon=True)
        t.start()
        camera_pose_estimator = CameraPoseEstimator()
        if mask_prompt is not None:
            segmenter = Florence2Segmenter()
        # Capture images
        capture_stage = 0
        while not rospy.is_shutdown():
            if capture_stage == 0:
                rospy.loginfo("Waiting for camera connection...")
                while not camera_pose_estimator.wait_for_images(5.0):
                    rospy.loginfo("Still waiting for the first image pair...")
                rospy.loginfo("Camera is ready!")
                # Capture image
                input("Press ENTER to capture FIRST image (Move camera to pos 1)...")
                color1, depth1 = camera_pose_estimator.capture_new_image()
                
                if color1 is not None:
                    # Capture用の変数を更新
                    camera_pose_estimator.color_image1 = color1
                    camera_pose_estimator.depth_image1 = depth1
                    capture_stage = 1
                else:
                    rospy.logerr("Failed to capture 1st image. Retrying...")

            elif capture_stage == 1:
                input("Press ENTER to capture SECOND image (Move camera to pos 2)...")
                color2, depth2 = camera_pose_estimator.capture_new_image()
                
                if color2 is not None:
                    # Capture用の変数を更新
                    camera_pose_estimator.color_image2 = color2
                    camera_pose_estimator.depth_image2 = depth2
                    capture_stage = 2
                else:
                    rospy.logerr("Failed to capture 2nd image. Retrying...")
            else:
                break
        # Apply mask
        if mask_prompt is not None:
            rospy.loginfo(f"Calculating mask of {mask_prompt}")
            results1, image1 = segmenter.process_image(color1, mask_prompt)
            mask1 = segmenter.create_mask(results1, image1.shape)
            results2, image2 = segmenter.process_image(color2, mask_prompt)
            mask2 = segmenter.create_mask(results2, image2.shape)
        else:
            mask1 = None
            mask2 = None
        # Pose estimation
        rospy.loginfo("Pose estimation")
        camera_pose_estimator.process_images(mask1=mask1, mask2=mask2)
        rospy.signal_shutdown("Finished capturing images")
        t.join()
    except rospy.ROSInterruptException:
        pass
