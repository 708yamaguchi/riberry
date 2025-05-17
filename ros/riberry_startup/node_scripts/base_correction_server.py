#!/usr/bin/env python

import os
import threading

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from skrobot.coordinates.base import Coordinates
from skrobot.coordinates.base import transform_coords
from skrobot.interfaces.ros.tf_utils import coords_to_tf_pose
from skrobot.interfaces.ros.tf_utils import tf_pose_to_coords
from skrobot.interfaces.ros.transform_listener import TransformListener
from std_msgs.msg import String
from tf2_ros import TransformException
import torch

from riberry.feature_matcher import FeatureMatcher
from riberry.feature_matcher import save_matches
from riberry.filecheck_utils import get_cache_dir
from riberry.florence2_segmenter import Florence2Segmenter

# Avoid CUDA out of memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

class CameraPoseEstimator:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_camera_matrix = None
        self.depth_camera_matrix = None
        self.color1_image = None
        self.depth1_image = None
        self.color2_image = None
        self.depth2_image = None
        self.subs = [Subscriber("base_correction_mode/color/1", Image),
                     Subscriber("base_correction_mode/depth/1", Image),
                     Subscriber("base_correction_mode/color/2", Image),
                     Subscriber("base_correction_mode/depth/2", Image),
                     Subscriber("base_correction_mode/color/camera_info", CameraInfo),
                     Subscriber("base_correction_mode/depth/camera_info", CameraInfo)]
        self.mask_prompt = None
        rospy.Subscriber("base_correction_mode/mask_prompt", String, self.mask_prompt_cb)
        self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=30, slop=0.1)
        self.sync.registerCallback(self.image_callback)

    def mask_prompt_cb(self, msg):
        self.mask_prompt = msg.data
        rospy.loginfo(f"Mask prompt received: {self.mask_prompt}")

    def image_callback(self, color1_img, depth1_img, color2_img, depth2_img, color_info, depth_info):
        # Get images
        self.color1_image = self.bridge.imgmsg_to_cv2(
            color1_img, color1_img.encoding)
        self.depth1_image = self.bridge.imgmsg_to_cv2(
            depth1_img, depth1_img.encoding)
        self.color2_image = self.bridge.imgmsg_to_cv2(
            color2_img, color2_img.encoding)
        self.depth2_image = self.bridge.imgmsg_to_cv2(
            depth2_img, depth2_img.encoding)
        # Get camera infos
        def parse_camera_info(camera_info_msg):
            K = np.array(camera_info_msg.K, dtype=np.float64).reshape((3, 3))
            D = np.array(camera_info_msg.D, dtype=np.float64)
            return K, D
        if self.color_camera_matrix is None:
            self.color_camera_matrix, _ = parse_camera_info(color_info)
            self.depth_camera_matrix, _ = parse_camera_info(depth_info)

    def are_images_received(self):
        return (self.color1_image is not None and
                self.depth1_image is not None and
                self.color2_image is not None and
                self.depth2_image is not None and
                self.color_camera_matrix is not None and
                self.depth_camera_matrix is not None)

    def clear_images(self):
        self.color1_image = None
        self.depth1_image = None
        self.color2_image = None
        self.depth2_image = None
        self.color_camera_matrix = None
        self.depth_camera_matrix = None

    def estimate_pose(self, inlier_mkpts1: torch.Tensor, inlier_mkpts2: torch.Tensor):
        # Convert tensors to numpy arrays for OpenCV
        pts1 = inlier_mkpts1.detach().cpu().numpy().astype(np.float32)
        pts2 = inlier_mkpts2.detach().cpu().numpy().astype(np.float32)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=self.color_camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None or E.shape != (3, 3):
            raise ValueError("Essential matrix estimation failed or is degenerate.")

        # Recover relative pose (R, t) from essential matrix
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=self.color_camera_matrix)
        return R, t

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

        rospy.loginfo(f"{len(pts1) - len(points3d_1)} in {len(pts1)} points are ignored because of 0mm depth")
        if len(points3d_1) < 2:
            raise ValueError("Not enough valid 3D points for triangulation.")

        scales = []
        for p1, p2 in zip(points3d_1, points3d_2):
            p1_proj = R @ p1  # project p1 into the second camera's frame
            delta = p2 - p1_proj
            scale = np.dot(delta, t_unit.reshape(-1))  # project delta onto unit translation direction
            scales.append(scale)

        return np.median(scales)

    def depth_pose_estimation(self, matching_result):
        # Filter inlier mkpts (matched keypoints)
        inliers_1d = matching_result.inliers[:, 0].astype(bool)
        inlier_mkpts1 = matching_result.kps1[matching_result.idxs[:, 0][inliers_1d]]
        inlier_mkpts2 = matching_result.kps2[matching_result.idxs[:, 1][inliers_1d]]
        # Pose estimation
        if len(inlier_mkpts1) >= 4:
            R, t = self.estimate_pose(inlier_mkpts1, inlier_mkpts2)
            rospy.loginfo(f"Rotation Matrix (R):\n{R}")
            if t is not None:
                t_unit = t / np.linalg.norm(t)
                try:
                    scale = self.compute_translation_scale(
                        inlier_mkpts1, inlier_mkpts2, self.depth1_image, self.depth2_image, R, t_unit
                    )
                    t_scaled = t_unit * scale
                    rospy.loginfo(f"Translation Vector (t):\n{t_scaled}")
                    rospy.loginfo(f"Translation: {scale * 1000:.2f} [mm]")
                    return (R, t_scaled)
                except ValueError as e:
                    rospy.logerr(e)
                    return None
            else:
                rospy.loginfo("Pose estimation failed - insufficient inliers")
        else:
            rospy.loginfo("Not enough inliers for pose estimation (need at least 4)")

    def wait_for_new_images(self):
        rate = rospy.Rate(1)
        rospy.loginfo("Wait for images to be received...")
        while not self.are_images_received() and not rospy.is_shutdown():
            rate.sleep()
        rospy.loginfo("Images are received.")

    def wait_for_mask_prompt(self, timeout):
        rate = rospy.Rate(1)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("Timeout while waiting for mask_prompt")
                return False
            if self.mask_prompt is not None:
                return True
            rate.sleep()
        return False

def convert_16uc1_to_8uc1(img_16uc1):
    if img_16uc1.dtype != np.uint16:
        print("入力画像は16UC1形式ではありません")
        return
    min_val = np.min(img_16uc1)
    max_val = np.max(img_16uc1)
    if max_val > min_val:
        img_8uc1 = ((img_16uc1.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        img_8uc1 = np.zeros_like(img_16uc1, dtype=np.uint8) if min_val == 0 else np.full_like(img_16uc1, 255, dtype=np.uint8)
    return img_8uc1


def run_camera_pose_estimation(
        feature_matcher, camera_pose_estimator, segmenter,
        visualize=False, save_images=True):
    # Get camera images
    color1 = camera_pose_estimator.color1_image
    depth1 = camera_pose_estimator.depth1_image
    color2 = camera_pose_estimator.color2_image
    depth2 = camera_pose_estimator.depth2_image
    # Save raw images
    if save_images:
        target_dir = os.path.join(get_cache_dir(), "base_correction")
        rospy.loginfo(f"Save images under {target_dir}")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        file_names = ["color1.png", "depth1.png", "depth1_8uc1.png",
                      "color2.png", "depth2.png", "depth2_8uc1.png"]
        imgs = [cv2.cvtColor(color1, cv2.COLOR_RGB2BGR), depth1, convert_16uc1_to_8uc1(depth1),
                cv2.cvtColor(color2, cv2.COLOR_RGB2BGR), depth2, convert_16uc1_to_8uc1(depth2)]
        for file_name, img in zip(file_names ,imgs):
            cv2.imwrite(os.path.join(target_dir, file_name), img)
    if segmenter:
        for i, (color, depth) in enumerate([(color1, depth1), (color2, depth2)], 1):
            results, img = segmenter.process_image(color)
            mask = segmenter.create_mask(results, img.shape)
            color[mask == 0] = 0
            depth[mask == 0] = 0
            if i == 1:
                color1, depth1 = color, depth
            else:
                color2, depth2 = color, depth
        # Save maksed images
        if save_images:
            file_names = ["color1_masked.png", "depth1_masked.png", "depth1_8uc1_masked.png",
                          "color2_masked.png", "depth2_masked.png", "depth2_8uc1_masked.png"]
            imgs = [cv2.cvtColor(color1, cv2.COLOR_RGB2BGR), depth1, convert_16uc1_to_8uc1(depth1),
                    cv2.cvtColor(color2, cv2.COLOR_RGB2BGR), depth2, convert_16uc1_to_8uc1(depth2)]
            for file_name, img in zip(file_names ,imgs):
                cv2.imwrite(os.path.join(target_dir, file_name), img)
    rospy.loginfo("Pose estimation")
    try:
        match = feature_matcher.process_images(color1, color2)
    except cv2.error:
        rospy.logerr("Failed to calculate match")
        match = None
    if match is None:
        return None
    else:
        result = camera_pose_estimator.depth_pose_estimation(match)
        save_matches(match, os.path.join(target_dir, "match.png"),
                     visualize=visualize)
        return result


def run_base_pose_correction(R, t):
    # カメラ相対での移動量を、ベースの移動量に変換
    transform_in_camera_coords = Coordinates(pos=t, rot=R)
    rospy.loginfo(f"Transform in camera coords:\n{transform_in_camera_coords}")
    # Base to camera coords
    tfl = TransformListener(use_tf2=False)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            tfl.wait_for_transform("base_link", "camera_color_optical_frame", rospy.Time(0))
            base_to_camera_tf = tfl.lookup_transform("base_link", "camera_color_optical_frame")
            base_to_camera_coords = tf_pose_to_coords(base_to_camera_tf)
            rospy.loginfo(f"base_to_camera_coords:\n{base_to_camera_coords}")
            break
        except TransformException as e:
            rospy.logwarn(f"試行 {attempt + 1}/{max_retries}: TF変換に失敗 - {e!s}")
            if attempt == max_retries - 1:  # 最終試行で失敗
                rospy.logerr("最大リトライ回数に達しました。TF変換を取得できませんでした。")
            rospy.sleep(1.0)
    # Base transform
    # 注意: カメラ画像を取得するときのベースとカメラの相対姿勢(base_to_camera_coords)は一定として計算している
    base_transform = transform_coords(
        transform_coords(base_to_camera_coords, transform_in_camera_coords),
        base_to_camera_coords.inverse_transformation())
    rospy.loginfo(f"base_transform:\n{base_transform}")

    return base_transform.rotation, base_transform.translation


if __name__ == "__main__":
    rospy.init_node("camera_pose_estimator", anonymous=True)
    threading.Thread(target=rospy.spin, daemon=True).start()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")
    pub = rospy.Publisher('base_correction_mode/transform', Transform, queue_size=1)

    # Create main processors
    feature_matcher = FeatureMatcher(device=device)
    camera_pose_estimator = CameraPoseEstimator()
    camera_pose_estimator.wait_for_new_images()

    # Wait for mask prompt until timeout
    camera_pose_estimator.wait_for_mask_prompt(5)
    mask_prompt = camera_pose_estimator.mask_prompt
    if mask_prompt:
        rospy.loginfo(f"Receive mask prompt {mask_prompt}")
        segmenter = Florence2Segmenter(device=device)
        segmenter.set_mask_prompt(mask_prompt)
        rospy.loginfo(f"Calculating mask of {mask_prompt}")
        camera_pose_estimator.mask_prompt = None
    else:
        segmenter = None

    # Loop for "Capture image -> Base pose correction"
    while not rospy.is_shutdown():
        result = run_camera_pose_estimation(
            feature_matcher=feature_matcher,
            camera_pose_estimator=camera_pose_estimator,
            segmenter=segmenter,
            # visualize=False)
            visualize=True)
        if result is not None:
            R_in_camera_coords, t_in_camera_coords = result
            R_in_base_coords, t_in_base_coords = run_base_pose_correction(
                R_in_camera_coords, t_in_camera_coords)
            base_transform = Coordinates(pos=t_in_base_coords, rot=R_in_base_coords)
            base_transform_msg = coords_to_tf_pose(base_transform)
            pub.publish(base_transform_msg)
            rospy.loginfo(f"Publish /base_transform\n{base_transform_msg}")
        else:
            pub.publish(Transform())
            rospy.loginfo("Failed to match, publish Empty /base_transform topic")
        camera_pose_estimator.clear_images()
        camera_pose_estimator.wait_for_new_images()
        rospy.sleep(1)  # Wait for mask prompt
        segmenter.set_mask_prompt(camera_pose_estimator.mask_prompt)

    rospy.signal_shutdown("Finished pose estimation")
