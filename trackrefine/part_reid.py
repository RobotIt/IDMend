"""Mask-guided vertical part descriptors, activated only by V70 config."""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_utils


PARTS = ((0.0, .33), (.33, .70), (.70, 1.0))


def attach_vertical_part_features(images, frames, encoder, batch_size: int = 128, minimum_visible_ratio: float = .08):
    """Attach ``[(feature, visible), ...]`` to detections in place.

    This is deliberately separate from normal ReID.  V70 invokes it only for
    the screened run; baseline pipelines pay no cost and see no new feature.
    """
    crops, owners, visible = [], [], []
    for frame in sorted(frames):
        image = cv2.imread(str(images[frame]))
        if image is None:
            raise FileNotFoundError(images[frame])
        for det in frames[frame]:
            mask = mask_utils.decode(det.rle).astype(bool)
            x1, y1, x2, y2 = det.box.round().astype(int)
            y1, y2 = max(0, y1), min(mask.shape[0], y2)
            area = max(int(mask.sum()), 1)
            for index, (low, high) in enumerate(PARTS):
                begin = y1 + round((y2 - y1) * low)
                end = y1 + round((y2 - y1) * high)
                part = np.zeros_like(mask)
                part[begin:end, :] = mask[begin:end, :]
                owners.append((det, index))
                visible.append(part.sum() / area >= minimum_visible_ratio)
                crops.append(encoder.crop_from_mask(image, det, part))
    features = []
    for start in range(0, len(crops), batch_size):
        features.extend(encoder.encode_crops(crops[start:start + batch_size]))
    for det in (item for frame in sorted(frames) for item in frames[frame]):
        det.part_features = [None, None, None]
    for (det, index), feature, is_visible in zip(owners, features, visible):
        det.part_features[index] = (feature.astype(np.float32), bool(is_visible))
    return {"part_reid_instances": len(owners) // 3, "part_reid_crops": len(owners)}


def tracklet_part_features(first, second):
    """Same four statistics used by the V70 offline audit."""
    tail = first.observations[-min(8, len(first.observations)):]
    head = second.observations[:min(8, len(second.observations))]
    similarities = []
    for index in range(3):
        left = [x.part_features[index][0] for x in tail if getattr(x, "part_features", None) and x.part_features[index][1]]
        right = [x.part_features[index][0] for x in head if getattr(x, "part_features", None) and x.part_features[index][1]]
        if not left or not right:
            continue
        first_mean, second_mean = np.mean(left, axis=0), np.mean(right, axis=0)
        first_mean /= np.linalg.norm(first_mean) + 1e-12
        second_mean /= np.linalg.norm(second_mean) + 1e-12
        similarities.append(float(first_mean @ second_mean))
    return np.asarray([
        float(np.mean(similarities)) if similarities else 0.0,
        float(np.min(similarities)) if similarities else 0.0,
        float(np.std(similarities)) if similarities else 1.0,
        len(similarities) / 3.0,
    ], np.float32)


def attach_vertical_part_features_for_observations(
    images, observations, encoder, batch_size: int = 128,
    minimum_visible_ratio: float = .08,
):
    """Attach part features only to unique observations requested by a graph."""
    unique = {id(observation): observation for observation in observations}
    selected_observations = sorted(
        unique.values(), key=lambda item: (item.frame, id(item))
    )
    crops, owners, visible = [], [], []
    image_cache = {}
    for det in selected_observations:
        image = image_cache.get(det.frame)
        if image is None:
            image = cv2.imread(str(images[det.frame]))
            if image is None:
                raise FileNotFoundError(images[det.frame])
            image_cache[det.frame] = image
        mask = mask_utils.decode(det.rle).astype(bool)
        _x1, y1, _x2, y2 = det.box.round().astype(int)
        y1, y2 = max(0, y1), min(mask.shape[0], y2)
        area = max(int(mask.sum()), 1)
        for index, (low, high) in enumerate(PARTS):
            begin = y1 + round((y2 - y1) * low)
            end = y1 + round((y2 - y1) * high)
            part = np.zeros_like(mask)
            part[begin:end, :] = mask[begin:end, :]
            owners.append((det, index))
            visible.append(part.sum() / area >= minimum_visible_ratio)
            crops.append(encoder.crop_from_mask(image, det, part))
    features = []
    for start in range(0, len(crops), batch_size):
        features.extend(encoder.encode_crops(crops[start:start + batch_size]))
    for det in selected_observations:
        det.part_features = [None, None, None]
    for (det, index), feature, is_visible in zip(owners, features, visible):
        det.part_features[index] = (feature.astype(np.float32), bool(is_visible))
    return {
        "part_reid_instances": len(selected_observations),
        "part_reid_crops": len(crops),
    }
