from dataclasses import dataclass

import cv2
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class MatchingResult:
    img1: torch.Tensor
    img2: torch.Tensor
    kps1: torch.Tensor
    kps2: torch.Tensor
    idxs: torch.Tensor
    inliers: np.ndarray


class FeatureMatcher:
    def __init__(self, feature_type="disk", device="cpu"):
        self.device = device
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


def save_matches(result: MatchingResult, img_path, visualize=False):
    kps1 = result.kps1.cpu().numpy() if hasattr(result.kps1, 'cpu') else result.kps1
    kps2 = result.kps2.cpu().numpy() if hasattr(result.kps2, 'cpu') else result.kps2
    idxs = result.idxs.cpu().numpy() if hasattr(result.idxs, 'cpu') else result.idxs
    inliers = result.inliers.cpu().numpy() if hasattr(result.inliers, 'cpu') else result.inliers

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(kps1[None])),
        KF.laf_from_center_scale_ori(torch.from_numpy(kps2[None])),
        torch.from_numpy(idxs),
        K.tensor_to_image(result.img1.cpu()),
        K.tensor_to_image(result.img2.cpu()),
        torch.from_numpy(inliers) if inliers is not None else None,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": (1, 1, 0.2, 0.3),
            "feature_color": None,
            "vertical": False,
        },
    )
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    if visualize:
        plt.show()
