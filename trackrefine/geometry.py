from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def box_iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), np.float32)
    lt = np.maximum(first[:, None, :2], second[None, :, :2])
    rb = np.minimum(first[:, None, 2:], second[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.prod(np.maximum(first[:, 2:] - first[:, :2], 0), axis=1)
    area2 = np.prod(np.maximum(second[:, 2:] - second[:, :2], 0), axis=1)
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-6)


def transform_box(box: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    corners = np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
        np.float32,
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    return np.array(
        [mapped[:, 0].min(), mapped[:, 1].min(),
         mapped[:, 0].max(), mapped[:, 1].max()],
        np.float32,
    )


class CameraMotion:
    """ORB/RANSAC camera motion compensation with cumulative homographies."""

    def __init__(self, image_paths: List[Path], cache: Path | None = None):
        self.paths = image_paths
        self.cache = cache
        self.cumulative: List[np.ndarray] = []

    def compute(self) -> List[np.ndarray]:
        if self.cache and self.cache.exists():
            values = np.load(self.cache)["homographies"]
            self.cumulative = [value.astype(np.float32) for value in values]
            return self.cumulative
        orb = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cumulative = np.eye(3, dtype=np.float32)
        self.cumulative = [cumulative.copy()]
        previous, scale = self._gray(self.paths[0])
        kp_prev, des_prev = orb.detectAndCompute(previous, None)
        for path in self.paths[1:]:
            current, current_scale = self._gray(path)
            kp_curr, des_curr = orb.detectAndCompute(current, None)
            step = np.eye(3, dtype=np.float32)
            if des_prev is not None and des_curr is not None:
                matches = sorted(matcher.match(des_prev, des_curr), key=lambda m: m.distance)
                matches = matches[: min(500, len(matches))]
                if len(matches) >= 12:
                    src = np.float32([kp_prev[m.queryIdx].pt for m in matches])
                    dst = np.float32([kp_curr[m.trainIdx].pt for m in matches])
                    affine, inliers = cv2.estimateAffinePartial2D(
                        src, dst, method=cv2.RANSAC,
                        ransacReprojThreshold=3.0, maxIters=2000,
                        confidence=0.995,
                    )
                    if affine is not None and inliers is not None and inliers.sum() >= 8:
                        # Keypoints were measured on a uniformly downscaled
                        # image; convert translation back to native pixels.
                        affine[:, 2] /= max(current_scale, 1e-12)
                        step[:2] = affine
            cumulative = step @ cumulative
            self.cumulative.append(cumulative.copy())
            previous, kp_prev, des_prev = current, kp_curr, des_curr
        if self.cache:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache, homographies=np.stack(self.cumulative))
        return self.cumulative

    @staticmethod
    def _gray(path: Path) -> tuple[np.ndarray, float]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        scale = min(1.0, 720.0 / image.shape[1])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image, scale

    def between(self, source_index: int, target_index: int) -> np.ndarray:
        return self.cumulative[target_index] @ np.linalg.inv(self.cumulative[source_index])
