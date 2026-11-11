from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def _center(box):
    return (np.asarray(box[:2]) + np.asarray(box[2:])) / 2


def link_features(first, second) -> np.ndarray:
    """Sequence-agnostic tracklet-pair features used by train and inference."""
    gap = second.observations[0].frame - first.last.frame
    tail = first.observations[-min(8, len(first.observations)):]
    times = np.asarray([item.frame for item in tail], np.float32)
    centers = np.asarray([_center(item.box) for item in tail], np.float32)
    if len(tail) >= 2:
        centered = times - times.mean()
        velocity = (centered[:, None] * (centers - centers.mean(0))).sum(0)
        velocity /= (centered ** 2).sum() + 1e-6
    else:
        velocity = np.zeros(2, np.float32)
    end_center = _center(first.last.box)
    start_center = _center(second.observations[0].box)
    displacement = start_center - end_center
    height = max(
        first.last.box[3] - first.last.box[1],
        second.observations[0].box[3] - second.observations[0].box[1], 25,
    )
    direct = np.linalg.norm(displacement) / height
    predicted = np.linalg.norm(end_center + velocity * gap - start_center) / height
    direction = float(
        np.dot(velocity, displacement)
        / ((np.linalg.norm(velocity) + 1e-6) * (np.linalg.norm(displacement) + 1e-6))
    )
    area1 = max(np.prod(first.last.box[2:] - first.last.box[:2]), 1)
    area2 = max(np.prod(second.observations[0].box[2:] - second.observations[0].box[:2]), 1)
    h1 = max(first.last.box[3] - first.last.box[1], 1)
    h2 = max(second.observations[0].box[3] - second.observations[0].box[1], 1)
    first_tail_feature = np.mean(
        [item.feature for item in first.observations[-min(8, len(first.observations)):]], axis=0
    )
    second_head_feature = np.mean(
        [item.feature for item in second.observations[:min(8, len(second.observations))]], axis=0
    )
    first_tail_feature /= np.linalg.norm(first_tail_feature) + 1e-12
    second_head_feature /= np.linalg.norm(second_head_feature) + 1e-12
    endpoint_similarity = float(first_tail_feature @ second_head_feature)
    return np.asarray([
        float(first.feature @ second.feature),
        endpoint_similarity,
        np.log1p(gap) / np.log(61),
        min(float(direct), 12) / 12,
        min(float(predicted), 12) / 12,
        direction,
        min(abs(float(np.log(area1 / area2))), 3) / 3,
        min(abs(float(np.log(h1 / h2))), 2) / 2,
        min(len(first.observations), 100) / 100,
        min(len(second.observations), 100) / 100,
        float(displacement[0] / height) / 8,
        float(displacement[1] / height) / 8,
    ], np.float32)


def _fresh_boundary(observations, from_end: bool) -> list:
    boundary = (
        observations[-min(12, len(observations)):]
        if from_end
        else observations[:min(12, len(observations))]
    )
    fresh = [
        observation
        for observation in boundary
        if getattr(observation, "feature_is_fresh", False)
        and observation.feature is not None
    ]
    return fresh or [
        observation
        for observation in boundary
        if observation.feature is not None
    ]


def link_features_v2(first, second) -> np.ndarray:
    """Freshness-aware appearance and motion features for V9 stitching."""
    gap = second.observations[0].frame - first.last.frame
    tail_motion = first.observations[-min(8, len(first.observations)):]
    times = np.asarray([item.frame for item in tail_motion], np.float32)
    centers = np.asarray([_center(item.box) for item in tail_motion], np.float32)
    if len(tail_motion) >= 2:
        centered = times - times.mean()
        velocity = (centered[:, None] * (centers - centers.mean(0))).sum(0)
        velocity /= (centered ** 2).sum() + 1e-6
    else:
        velocity = np.zeros(2, np.float32)

    end_center = _center(first.last.box)
    start_center = _center(second.observations[0].box)
    displacement = start_center - end_center
    height = max(
        first.last.box[3] - first.last.box[1],
        second.observations[0].box[3] - second.observations[0].box[1],
        25,
    )
    direct = np.linalg.norm(displacement) / height
    predicted = np.linalg.norm(
        end_center + velocity * gap - start_center
    ) / height
    direction = float(
        np.dot(velocity, displacement)
        / (
            (np.linalg.norm(velocity) + 1e-6)
            * (np.linalg.norm(displacement) + 1e-6)
        )
    )
    area1 = max(np.prod(first.last.box[2:] - first.last.box[:2]), 1)
    area2 = max(
        np.prod(
            second.observations[0].box[2:]
            - second.observations[0].box[:2]
        ),
        1,
    )
    height1 = max(first.last.box[3] - first.last.box[1], 1)
    height2 = max(
        second.observations[0].box[3]
        - second.observations[0].box[1],
        1,
    )

    tail = _fresh_boundary(first.observations, from_end=True)
    head = _fresh_boundary(second.observations, from_end=False)
    tail_features = np.stack([item.feature for item in tail])
    head_features = np.stack([item.feature for item in head])
    tail_mean = tail_features.mean(axis=0)
    head_mean = head_features.mean(axis=0)
    tail_mean /= np.linalg.norm(tail_mean) + 1e-12
    head_mean /= np.linalg.norm(head_mean) + 1e-12
    cross = tail_features @ head_features.T
    endpoint_similarity = float(tail_mean @ head_mean)
    prototype_similarity = float(first.feature @ second.feature)
    tail_consistency = float(np.mean(tail_features @ first.feature))
    head_consistency = float(np.mean(head_features @ second.feature))
    tail_fresh_ratio = sum(
        getattr(item, "feature_is_fresh", False) for item in tail
    ) / max(len(tail), 1)
    head_fresh_ratio = sum(
        getattr(item, "feature_is_fresh", False) for item in head
    ) / max(len(head), 1)

    return np.asarray(
        [
            prototype_similarity,
            endpoint_similarity,
            float(np.max(cross)),
            float(np.median(cross)),
            tail_consistency,
            head_consistency,
            np.log1p(gap) / np.log(91),
            min(float(direct), 12) / 12,
            min(float(predicted), 12) / 12,
            direction,
            min(abs(float(np.log(area1 / area2))), 3) / 3,
            min(abs(float(np.log(height1 / height2))), 2) / 2,
            min(len(first.observations), 100) / 100,
            min(len(second.observations), 100) / 100,
            float(displacement[0] / height) / 8,
            float(displacement[1] / height) / 8,
            float(tail_fresh_ratio),
            float(head_fresh_ratio),
            float(np.mean([item.score for item in tail])),
            float(np.mean([item.score for item in head])),
        ],
        np.float32,
    )


def _normalized_silhouette(detection, height: int = 64, width: int = 32):
    """Predicted-mask geometry normalized for translation and box scale."""
    cached = getattr(detection, "_v78_silhouette", None)
    if cached is not None and cached[0].shape == (height, width):
        return cached
    mask = mask_utils.decode(detection.rle).astype(np.uint8)
    x1, y1, x2, y2 = detection.box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2 = min(mask.shape[1], int(np.ceil(x2)))
    y2 = min(mask.shape[0], int(np.ceil(y2)))
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        normalized = np.zeros((height, width), np.float32)
        box_area = 1.0
    else:
        normalized = cv2.resize(
            crop.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
        )
        box_area = float(crop.size)
    area = float(mask.sum())
    result = (normalized, area, area / max(box_area, 1.0))
    detection._v78_silhouette = result
    return result


def mask_geometry_features(first, second) -> np.ndarray:
    """Eight low-cost predicted-mask features for unified V78 linking."""
    tail = [
        _normalized_silhouette(item)
        for item in first.observations[-min(8, len(first.observations)):]
    ]
    head = [
        _normalized_silhouette(item)
        for item in second.observations[:min(8, len(second.observations))]
    ]
    end_shape, end_area, end_ratio = tail[-1]
    start_shape, start_area, start_ratio = head[0]

    def similarity(left, right):
        a, b = left.reshape(-1), right.reshape(-1)
        return float(
            a @ b
            / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))
        )

    union = np.maximum(end_shape, start_shape).sum()
    soft_iou = float(
        np.minimum(end_shape, start_shape).sum() / max(union, 1e-12)
    )
    cross = np.asarray(
        [similarity(left[0], right[0]) for left in tail for right in head],
        np.float32,
    )
    tail_ratios = np.asarray([item[2] for item in tail], np.float32)
    head_ratios = np.asarray([item[2] for item in head], np.float32)
    area_change = min(
        abs(float(np.log(max(end_area, 1.0) / max(start_area, 1.0)))), 3.0
    ) / 3.0
    return np.asarray(
        [
            soft_iou,
            similarity(end_shape, start_shape),
            float(cross.mean()),
            float(cross.max()),
            area_change,
            abs(end_ratio - start_ratio),
            abs(float(tail_ratios.mean() - head_ratios.mean())),
            min(float(tail_ratios.mean()), float(head_ratios.mean())),
        ],
        np.float32,
    )


class LinkMLP:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.architecture = str(payload.get("architecture", "20-32-1 ReLU MLP"))
        self.feature_version = str(payload.get("feature_version", "v1"))
        self.feature_columns = (
            np.asarray(payload["feature_columns"], np.int64)
            if "feature_columns" in payload
            else None
        )
        self.mean = np.asarray(payload["mean"], np.float32)
        self.std = np.asarray(payload["std"], np.float32)
        self.linear_weight = (
            np.asarray(payload["linear_weight"], np.float32)
            if "linear_weight" in payload
            else None
        )
        self.linear_bias = (
            float(payload["linear_bias"])
            if "linear_bias" in payload
            else None
        )
        self.w1 = np.asarray(payload["w1"], np.float32) if "w1" in payload else None
        self.b1 = np.asarray(payload["b1"], np.float32) if "b1" in payload else None
        self.w2 = np.asarray(payload["w2"], np.float32) if "w2" in payload else None
        self.b2 = float(payload["b2"]) if "b2" in payload else None
        self.threshold = float(payload["threshold"])
        self.rank_w2 = (
            np.asarray(payload["rank_w2"], np.float32)
            if "rank_w2" in payload
            else None
        )
        self.rank_b2 = (
            float(payload["rank_b2"])
            if "rank_b2" in payload
            else None
        )
        self.part_rank_mean = np.asarray(payload["part_rank_mean"], np.float32) if "part_rank_mean" in payload else None
        self.part_rank_std = np.asarray(payload["part_rank_std"], np.float32) if "part_rank_std" in payload else None
        self.part_rank_weight = np.asarray(payload["part_rank_weight"], np.float32) if "part_rank_weight" in payload else None
        self.part_rank_bias = float(payload["part_rank_bias"]) if "part_rank_bias" in payload else None

    def features(self, first, second) -> np.ndarray:
        if self.feature_version == "exp3_v4_32":
            from .link_features32 import features32
            return features32(first, second)
        if self.feature_version == "v3_mask_geometry":
            return np.concatenate(
                (link_features_v2(first, second), mask_geometry_features(first, second))
            )
        if self.feature_version == "v2":
            return link_features_v2(first, second)
        return link_features(first, second)

    def score(self, features: np.ndarray) -> float:
        """Frozen gate-head probability used for candidate admission."""
        values = np.asarray(features, np.float32)
        if self.feature_columns is not None:
            values = values[self.feature_columns]
        x = (values - self.mean) / self.std
        if self.linear_weight is not None:
            logit = float(x @ self.linear_weight + self.linear_bias)
            return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        hidden = np.maximum(x @ self.w1 + self.b1, 0)
        logit = float(hidden @ self.w2 + self.b2)
        return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))

    def hidden(self, features: np.ndarray) -> np.ndarray:
        """Frozen gate representation, exposed only to optional rank heads."""
        if self.linear_weight is not None:
            raise ValueError("hidden representation is unavailable for a linear link model")
        values = np.asarray(features, np.float32)
        if self.feature_columns is not None:
            values = values[self.feature_columns]
        x = (values - self.mean) / self.std
        return np.maximum(x @ self.w1 + self.b1, 0)

    @property
    def has_rank_head(self) -> bool:
        return self.rank_w2 is not None and self.rank_b2 is not None

    @property
    def has_part_rank_head(self) -> bool:
        return all(value is not None for value in (self.part_rank_mean, self.part_rank_std, self.part_rank_weight, self.part_rank_bias))

    def part_rank_score(self, features: np.ndarray, part_features: np.ndarray) -> float:
        """V71 conflict-only score.  With no visible part, exactly use gate."""
        if not self.has_part_rank_head:
            raise ValueError("part_rank_score requested without a V71 part rank head")
        parts = np.asarray(part_features, np.float32)
        if len(parts) < 4 or parts[3] <= 0:
            return self.score(features)
        values = np.concatenate((self.hidden(features), [self.score(features)], parts))
        x = (values - self.part_rank_mean) / self.part_rank_std
        logit = float(x @ self.part_rank_weight + self.part_rank_bias)
        return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))

    def rank_score(self, features: np.ndarray) -> float:
        """Optional V62 rank-head score; it never changes gate admission."""
        if not self.has_rank_head:
            raise ValueError("rank_score requested from a single-head LinkMLP")
        x = (features - self.mean) / self.std
        hidden = np.maximum(x @ self.w1 + self.b1, 0)
        logit = float(hidden @ self.rank_w2 + self.rank_b2)
        return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
