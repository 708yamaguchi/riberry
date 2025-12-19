#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import torch
import threading
import os
import io
import matplotlib.pyplot as plt

# ROS Messages & Services
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
import tf.transformations as tft

# Sync
from message_filters import ApproximateTimeSynchronizer, Subscriber

# External Libraries
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
from transformers import AutoProcessor, AutoModelForCausalLM
from requests.exceptions import ChunkedEncodingError
from dataclasses import dataclass

# --- 定数: データ保存場所 (決め打ち) ---
SAVE_DIR = "/tmp/visual_pose_estimator"
PATH_RGB = os.path.join(SAVE_DIR, "ref_rgb.png")
PATH_DEPTH = os.path.join(SAVE_DIR, "ref_depth.png") # 16bit PNG
PATH_INFO = os.path.join(SAVE_DIR, "ref_info.yaml")
PATH_PROMPT = os.path.join(SAVE_DIR, "ref_prompt.txt")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

@dataclass
class MatchingResult:
    img1: torch.Tensor
    img2: torch.Tensor
    kps1: torch.Tensor
    kps2: torch.Tensor
    idxs: torch.Tensor
    inliers: np.ndarray

# =============================================================================
# Helper Classes
# =============================================================================

class FeatureMatcher:
    def __init__(self, feature_type="disk"):
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        rospy.loginfo(f"Loading LightGlue ({feature_type}) on {self.device}...")
        
        # モデルを即時ロード
        self.lg_matcher = KF.LightGlueMatcher(feature_type).eval().to(self.device)
        self.disk = KF.DISK.from_pretrained("depth").to(self.device)
        
        # ウォームアップ
        self.warmup()
        rospy.loginfo("LightGlue Loaded & Warmed up.")

    def warmup(self):
        try:
            dummy = torch.rand(1, 3, 256, 256).to(self.device)
            with torch.inference_mode():
                _ = self.disk(dummy, 100, pad_if_not_divisible=True)
        except Exception as e:
            rospy.logwarn(f"FeatureMatcher warmup failed (non-fatal): {e}")

    def process_images(self, img1_array, img2_array, num_features=2048):
        img1 = K.image_to_tensor(img1_array, False).float() / 255.0
        img2 = K.image_to_tensor(img2_array, False).float() / 255.0
        
        if img1.shape[2] < 16 or img1.shape[3] < 16:
            return None

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

        if idxs.shape[0] < 8:
            return None

        mkpts1, mkpts2 = kps1[idxs[:, 0]], kps2[idxs[:, 1]]

        Fm, inliers = cv2.findFundamentalMat(
            mkpts1.detach().cpu().numpy(),
            mkpts2.detach().cpu().numpy(),
            cv2.USAC_MAGSAC,
            1.0, 0.999, 100000
        )
        if inliers is None:
             inliers = np.zeros(idxs.shape[0], dtype=bool)
        else:
             inliers = inliers.astype(bool)

        return MatchingResult(img1, img2, kps1, kps2, idxs, inliers)

class Florence2Segmenter:
    def __init__(self, max_retries=3, retry_delay=2):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        rospy.loginfo("Loading Florence-2 Model... (This may take a while)")
        self._load_model()
        
        rospy.loginfo("Warming up Florence-2...")
        self.warmup()
        rospy.loginfo("Florence-2 Loaded & Warmed up.")

    def _load_model(self):
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

    def warmup(self):
        try:
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            self.process_image(dummy_img, "object")
        except Exception as e:
            rospy.logwarn(f"Florence-2 warmup warning: {e}")

    def process_image(self, image_input, prompt):
        image = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
        
        for attempt in range(self.max_retries):
            try:
                inputs = self.processor(text=f"<REFERRING_EXPRESSION_SEGMENTATION>{prompt}", images=image, return_tensors="pt").to(self.device, self.torch_dtype)
                
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
                    task="<REFERRING_EXPRESSION_SEGMENTATION>", 
                    image_size=(image.shape[1], image.shape[0])
                )
                return result, image

            except Exception as e:
                if attempt == self.max_retries - 1: 
                    rospy.logerr(f"Segmentation failed after retries: {e}")
                    return None, image
                rospy.sleep(self.retry_delay)

    def create_mask(self, parsed_answer, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if not parsed_answer: return mask
        results = parsed_answer.get("<REFERRING_EXPRESSION_SEGMENTATION>", {})
        for polygon_group in results.get('polygons', []):
            for coords in polygon_group:
                points = np.array(coords, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], 255)
        return mask

# =============================================================================
# Main ROS Node Class
# =============================================================================

class VisualPoseEstimatorNode:
    def __init__(self):
        rospy.init_node("visual_pose_estimator")
        
        # --- Parameters ---
        self.visualize = rospy.get_param("~visualize", True)
        self.prompt_param = rospy.get_param("~prompt", "") 
        
        self.bridge = CvBridge()
        
        # Load Models
        self.feature_matcher = FeatureMatcher()
        self.segmenter = Florence2Segmenter()
        
        # --- Data Buffers ---
        self.lock = threading.Lock()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info_K = None
        self.camera_info_D = None
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
        rospy.Service("save_reference", Trigger, self.handle_save_reference)
        rospy.Service("calculate_offset", Trigger, self.handle_calculate_offset)

        rospy.loginfo("==============================================")
        rospy.loginfo("   Visual Pose Estimator Node is READY !!     ")
        rospy.loginfo("   Models loaded. Waiting for services...     ")
        rospy.loginfo("==============================================")

    def info_cb(self, msg):
        if self.camera_info_K is None:
            self.camera_info_K = np.array(msg.K, dtype=np.float64).reshape((3, 3))
            self.camera_info_D = np.array(msg.D, dtype=np.float64)
            rospy.loginfo("Camera Info Received.")

    def image_cb(self, color_msg, depth_msg):
        with self.lock:
            try:
                self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
                self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                self.latest_timestamp = color_msg.header.stamp
            except Exception as e:
                rospy.logerr(f"Image conversion error: {e}")

    def capture_snapshot(self, timeout=5.0):
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        with self.lock:
             base_stamp = self.latest_timestamp

        while (rospy.Time.now() - start).to_sec() < timeout:
            with self.lock:
                if self.latest_timestamp > base_stamp and self.latest_color is not None and self.latest_depth is not None:
                     return self.latest_color.copy(), self.latest_depth.copy()
            rate.sleep()
        return None, None

    # -------------------------------------------------------------------------
    # Service Handlers
    # -------------------------------------------------------------------------

    def handle_save_reference(self, req):
        """Save current image as reference"""
        rospy.loginfo("Service: save_reference called.")
        color, depth = self.capture_snapshot()
        
        if color is None:
            return TriggerResponse(success=False, message="Timeout: No image received.")

        cv2.imwrite(PATH_RGB, color)
        cv2.imwrite(PATH_DEPTH, depth)

        prompt_to_save = self.prompt_param
        with open(PATH_PROMPT, 'w') as f:
            f.write(prompt_to_save)
        
        if self.camera_info_K is not None:
            fs = cv2.FileStorage(PATH_INFO, cv2.FILE_STORAGE_WRITE)
            fs.write("K", self.camera_info_K)
            fs.write("D", self.camera_info_D)
            fs.release()
        
        rospy.loginfo(f"Reference saved to {SAVE_DIR} (Prompt: '{prompt_to_save}')")
        return TriggerResponse(success=True, message=f"Saved to {SAVE_DIR}")

    def handle_calculate_offset(self, req):
        """Calculate pose offset"""
        rospy.loginfo("Service: calculate_offset called.")
        
        if not os.path.exists(PATH_RGB):
            return TriggerResponse(success=False, message="Reference files not found. Call save_reference first.")
        
        ref_color = cv2.imread(PATH_RGB)
        ref_depth = cv2.imread(PATH_DEPTH, cv2.IMREAD_UNCHANGED)
        
        ref_prompt = ""
        if os.path.exists(PATH_PROMPT):
            with open(PATH_PROMPT, 'r') as f:
                ref_prompt = f.read().strip()

        K_mat = self.camera_info_K
        D_mat = self.camera_info_D
        if os.path.exists(PATH_INFO):
             fs = cv2.FileStorage(PATH_INFO, cv2.FILE_STORAGE_READ)
             K_node = fs.getNode("K")
             D_node = fs.getNode("D")
             if not K_node.empty(): K_mat = K_node.mat()
             if not D_node.empty(): D_mat = D_node.mat()
             fs.release()
        
        if K_mat is None:
             return TriggerResponse(success=False, message="Camera Info not available.")

        curr_color, curr_depth = self.capture_snapshot()
        if curr_color is None:
            return TriggerResponse(success=False, message="Timeout: No current image.")

        # Segmentation
        mask_ref = None
        mask_curr = None
        if ref_prompt:
            rospy.loginfo(f"Applying segmentation with prompt: '{ref_prompt}'")
            res_ref, _ = self.segmenter.process_image(ref_color, ref_prompt)
            mask_ref = self.segmenter.create_mask(res_ref, ref_color.shape)
            res_curr, _ = self.segmenter.process_image(curr_color, ref_prompt)
            mask_curr = self.segmenter.create_mask(res_curr, curr_color.shape)

        # Matching
        img1_match = ref_color.copy()
        img2_match = curr_color.copy()
        if mask_ref is not None: img1_match[mask_ref == 0] = 0
        if mask_curr is not None: img2_match[mask_curr == 0] = 0

        res = self.feature_matcher.process_images(img1_match, img2_match)
        if res is None:
            return TriggerResponse(success=False, message="Not enough matches.")

        # --- 【修正】Tensorに変換してからフィルタリング ---
        # res.inliers は numpy 配列。これをGPU上のTensorのインデックスとして使うとエラーになる場合があるため
        # 明示的に bool Tensor に変換して device を合わせる
        inliers_bool_np = res.inliers[:,0].astype(bool)
        inliers_bool_tensor = torch.from_numpy(inliers_bool_np).to(res.idxs.device)
        
        inlier_mkpts1 = res.kps1[res.idxs[:, 0][inliers_bool_tensor]].cpu().numpy()
        inlier_mkpts2 = res.kps2[res.idxs[:, 1][inliers_bool_tensor]].cpu().numpy()

        # Pose Estimation
        success, R, t = self.estimate_pose_pnp(inlier_mkpts1, inlier_mkpts2, ref_depth, K_mat, D_mat)
        
        if not success:
            return TriggerResponse(success=False, message="PnP Failed.")

        # Format Result
        T_mat = np.eye(4)
        T_mat[:3, :3] = R
        q = tft.quaternion_from_matrix(T_mat)

        x_mm = t[0][0] * 1000
        y_mm = t[1][0] * 1000
        z_mm = t[2][0] * 1000
        total_mm = np.linalg.norm(t) * 1000
        rospy.loginfo(f"Camera Move (Camera Frame): X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm | Total={total_mm:.1f}mm")
        
        res_str = f"{t[0][0]:.6f},{t[1][0]:.6f},{t[2][0]:.6f},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}"
        rospy.loginfo(f"Pose Calculated: {res_str}")

        # Visualization
        self.publish_debug_image(res, ref_color, curr_color, mask_ref, mask_curr)
        
        return TriggerResponse(success=True, message=res_str)

    def estimate_pose_pnp(self, pts1, pts2, depth1, K, D):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        points_3d = []
        points_2d = []
        
        h, w = depth1.shape
        for i, (u, v) in enumerate(pts1):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                z = depth1[vi, ui] * 0.001 
                if z > 0.1:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points_3d.append([x, y, z])
                    points_2d.append(pts2[i])
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        if len(points_3d) < 4:
            rospy.logwarn("PnP: Not enough 3D points.")
            return False, None, None

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            points_3d, points_2d, K, D,
            iterationsCount=1000, reprojectionError=3.0, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return False, None, None
            
        R, _ = cv2.Rodrigues(rvec)
        return True, R, tvec

    def publish_debug_image(self, result, raw1, raw2, mask1, mask2):
        if not self.visualize:
            try:
                msg = self.bridge.cv2_to_imgmsg(raw2, encoding="bgr8")
                self.debug_pub.publish(msg)
            except Exception as e:
                rospy.logwarn(f"Failed to publish raw debug image: {e}")
            return
        
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        
        def blend(img, mask):
            if mask is None: return img
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask_3d = (mask[..., None] > 0)
            return np.clip(np.where(mask_3d, rgb, rgb * 0.3), 0, 1)

        disp1 = blend(raw1, mask1)
        disp2 = blend(raw2, mask2)
        
        # Tensorのまま取得
        kps1 = result.kps1.cpu()
        kps2 = result.kps2.cpu()
        idxs = result.idxs.cpu()
        
        # inliers を取得 (Boolean Mask for Matches)
        # result.inliers は (N, 1) などの形状の可能性があるため ravel() で1次元にする
        if hasattr(result.inliers, 'cpu'):
            inliers = result.inliers.cpu().numpy().ravel().astype(bool)
        else:
            inliers = result.inliers.ravel().astype(bool)
        
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(disp1); ax1.axis('off'); ax1.set_title("Reference")
        ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(disp2); ax2.axis('off'); ax2.set_title("Current")
        
        ax_match = fig.add_subplot(gs[1, :])
        
        t_img1 = K.image_to_tensor(raw1, False).float() / 255.0
        t_img2 = K.image_to_tensor(raw2, False).float() / 255.0
        
        try:
            draw_LAF_matches(
                KF.laf_from_center_scale_ori(kps1[None]),
                KF.laf_from_center_scale_ori(kps2[None]),
                idxs,
                t_img1, t_img2,
                inliers,
                draw_dict={"inlier_color": (0.2, 1, 0.2), "vertical": False},
                ax=ax_match
            )
        except Exception as e:
            rospy.logwarn(f"Viz failed: {e}")
            pass
        ax_match.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.show()  # 計算が止まるので、注意
        plt.close(fig)

        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        decoded = cv2.imdecode(arr, 1)
        msg = self.bridge.cv2_to_imgmsg(decoded, encoding="bgr8")
        self.debug_pub.publish(msg)



if __name__ == "__main__":
    try:
        VisualPoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

