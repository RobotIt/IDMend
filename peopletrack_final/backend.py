"""Frozen split-and-link inference, independent of experiment directories.

The original association implementation owns tracklet splitting, candidate
admission, local competition ranking and greedy identity assignment.  This
adapter supplies the deployed class-specific appearance and ResNetLink models.
"""
from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from trackrefine.association import GlobalTracker
from trackrefine.geometry import transform_box
from trackrefine.link_model import link_features_v2
from trackrefine.mots_io import read_mots, write_mots
from trackrefine.reid import AppearanceEncoder, _decode_rle_crop

from .model import ResNet18Link


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_set(path: Path) -> Counter:
    return Counter(
        tuple(parts[index] for index in (0, 2, 3, 4, 5))
        for line in path.read_text(encoding="ascii").splitlines()
        if (parts := line.split(" ", 5))
    )


def _crop(image: np.ndarray, detection, crop_h_w: tuple[int, int]) -> np.ndarray:
    height, width = crop_h_w
    x1, y1, x2, y2 = detection.box.round().astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.full((height, width, 3), 114, np.uint8)
    pixels = np.where(
        _decode_rle_crop(detection, x1, y1, x2, y2)[:, :, None],
        image[y1:y2, x1:x2],
        114,
    ).astype(np.uint8)
    return cv2.cvtColor(
        cv2.resize(pixels, (width, height)), cv2.COLOR_BGR2RGB
    )


class _ClassEncoder(AppearanceEncoder):
    def __init__(self, weights: Path, crop_h_w, device: str, batch_size: int):
        self.crop_h_w = tuple(crop_h_w)
        super().__init__(
            device=device,
            batch_size=batch_size,
            weights=weights,
            crop_rle_decode=True,
        )

    def _crop(self, image, detection):
        return _crop(image, detection, self.crop_h_w)


class _CameraTracker(GlobalTracker):
    """Original KITTI background-stabilised prediction, or the base predictor."""

    camera_mode = "none"

    def _prediction(self, track, frame):
        if self.camera_mode == "none":
            return super()._prediction(track, frame)
        reference_frame = track.last.frame
        reference = self.h[reference_frame - self.frame_offset]
        tail = track.observations[-min(8, len(track.observations)):]
        boxes = np.stack([
            transform_box(
                detection.box,
                reference @ np.linalg.inv(self.h[detection.frame - self.frame_offset]),
            )
            for detection in tail
        ])
        times = np.asarray([detection.frame for detection in tail], np.float32)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        centered = times - times.mean()
        velocity = (centered[:, None] * (centers - centers.mean(0))).sum(0)
        velocity /= np.square(centered).sum() + 1e-6
        predicted = track.last.box.copy()
        predicted[:2] += velocity * (frame - reference_frame)
        predicted[2:] += velocity * (frame - reference_frame)
        return transform_box(
            predicted, self.h[frame - self.frame_offset] @ np.linalg.inv(reference)
        )


def _raw_features(first, second, tracker):
    if tracker.camera_mode == "none":
        return link_features_v2(first, second)
    reference = tracker.h[second.observations[0].frame - tracker.frame_offset]

    def stabilise(track):
        result = copy(track)
        result.observations = []
        for detection in track.observations:
            observation = copy(detection)
            observation.box = transform_box(
                detection.box,
                reference @ np.linalg.inv(tracker.h[detection.frame - tracker.frame_offset]),
            )
            result.observations.append(observation)
        return result

    return link_features_v2(stabilise(first), stabilise(second))


class _PrecomputedLinker:
    has_rank_head = False
    has_part_rank_head = False

    def __init__(self, rows, scores, threshold, tracker):
        self.threshold = float(threshold)
        self.tracker = tracker
        self.table = {
            (row["first"], row["second"]): (row["raw"], float(score))
            for row, score in zip(rows, scores)
        }
        if len(self.table) != len(rows) or len(scores) != len(rows):
            raise ValueError("Duplicate or missing precomputed candidate scores")

    def features(self, first, second):
        return _raw_features(first, second, self.tracker)

    def score_pair(self, first, second, raw):
        expected, score = self.table[first.track_id, second.track_id]
        if not np.allclose(expected, raw, rtol=1e-5, atol=1e-6):
            raise ValueError("Rebuilt candidate differs from precomputed features")
        return score


def _candidates(tracker, tracks):
    rows = []
    for first in tracks:
        for second in tracks:
            gap = second.observations[0].frame - first.last.frame
            if not 0 < gap <= 90:
                continue
            predicted = tracker._prediction(first, second.observations[0].frame)
            target = second.observations[0].box
            distance = np.linalg.norm(
                (predicted[:2] + predicted[2:] - target[:2] - target[2:]) / 2
            ) / max(predicted[3] - predicted[1], target[3] - target[1], 25)
            if distance <= 2.5 + 0.06 * gap:
                rows.append({
                    "first": first.track_id,
                    "second": second.track_id,
                    "raw": _raw_features(first, second, tracker).tolist(),
                })
    return rows


class AssociationBackend:
    """One deployed class; image and weight paths are supplied by the caller.

    Feature caches contain float32 vectors without an extra renormalisation,
    matching the original final inference adapters.  Existing historical
    caches can be supplied explicitly through ``feature_cache_input``.
    """

    def __init__(self, tracking: dict, reid: dict, class_config: dict, device="cuda"):
        self.tracking = deepcopy(tracking)
        self.reid = deepcopy(reid)
        self.config = deepcopy(class_config)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_id = int(self.config["class_id"])
        self.window = int(self.config["window"])
        self.crop_h_w = tuple(int(value) for value in self.config["crop_h_w"])
        self.threshold = float(self.config["threshold"])
        self.camera_mode = self.config.get("camera_mode", "none")
        if self.camera_mode == "identity":
            self.camera_mode = "none"
        if self.class_id not in (1, 2) or self.window < 1:
            raise ValueError("Expected class_id 1/2 and a positive boundary window")
        if len(self.crop_h_w) != 2 or min(self.crop_h_w) <= 0:
            raise ValueError("crop_h_w must contain two positive integers")
        if self.camera_mode not in ("none", "background_lk"):
            raise ValueError(f"Unknown external camera mode: {self.camera_mode}")
        if int(self.tracking.get("global_max_gap", 90)) != 90:
            raise ValueError("The deployed candidate protocol requires global_max_gap=90")
        self.tracking["global_max_gap"] = 90
        self.tracking["link_threshold"] = self.threshold
        self.tracking["link_model"] = None
        self.tracking.setdefault("local_ranker", {})["model"] = str(self.config["ranker"])
        self.encoder = None
        self.model = None

    def _visual_model(self):
        if self.model is None:
            checkpoint = torch.load(
                Path(self.config["resnet"]), map_location="cpu", weights_only=False
            )
            self.model = ResNet18Link()
            self.model.load_state_dict(checkpoint["state"])
            self.model.to(self.device).eval()
        return self.model

    def _attach_features(self, frames, images, target, supplied):
        ordered = [detection for frame in sorted(frames) for detection in frames[frame]]
        cached = supplied if supplied is not None else (target if target.exists() else None)
        if cached is not None:
            cached = Path(cached)
            with np.load(cached) as payload:
                features = payload["features"]
                selected = payload["selected"]
                reasons = payload["reasons"].astype(str)
                if features.dtype != np.float32 or features.shape != (len(ordered), 512):
                    raise ValueError(f"Expected an aligned float32 OSNet cache: {cached}")
                if len(selected) != len(ordered) or len(reasons) != len(ordered):
                    raise ValueError(f"Feature metadata length differs: {cached}")
                if not np.isfinite(features).all():
                    raise ValueError(f"Nonfinite appearance vectors: {cached}")
                for detection, feature, fresh, reason in zip(ordered, features, selected, reasons):
                    detection.feature = feature.copy()
                    detection.feature_is_fresh = bool(fresh)
                    detection.feature_reason = str(reason)
            stats = {"reid_cache_hit": True, "reid_cache_source": str(cached)}
        else:
            if self.encoder is None:
                self.encoder = _ClassEncoder(
                    Path(self.config["osnet"]),
                    self.crop_h_w,
                    str(self.device),
                    int(self.reid.get("batch_size", 32)),
                )
            offset = min(frames)
            identity = np.tile(
                np.eye(3, dtype=np.float32), (max(images) + 1 - offset, 1, 1)
            )
            # Deliberately do not use AppearanceEncoder's default fp16 cache.
            stats = self.encoder.encode_sequence(
                images,
                frames,
                policy=self.reid.get("policy", "budget_aware"),
                event_config=self.reid.get("event", {}),
                homographies=identity,
            )
        if cached is None or Path(cached).resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                features=np.stack([detection.feature for detection in ordered]).astype(np.float32),
                selected=np.asarray([detection.feature_is_fresh for detection in ordered]),
                reasons=np.asarray([detection.feature_reason for detection in ordered]),
            )
        return stats

    @torch.inference_mode()
    def _score(self, tracks, rows, images):
        if not rows:
            return np.empty(0, np.float32), {"crops": 0, "window_lengths": {}}
        model = self._visual_model()
        by_id = {track.track_id: track for track in tracks}
        wanted = sorted({row[key] for row in rows for key in ("first", "second")})
        requests, slots = [], {}
        for identity in wanted:
            track = by_id[identity]
            for side, observations in (
                ("left", track.observations[-self.window:]),
                ("right", track.observations[:self.window]),
            ):
                slots[identity, side] = list(range(len(requests), len(requests) + len(observations)))
                requests.extend(observations)
        vectors = []
        for start in range(0, len(requests), 64):
            pixels = []
            for detection in requests[start:start + 64]:
                image = cv2.imread(str(images[detection.frame]))
                if image is None:
                    raise FileNotFoundError(images[detection.frame])
                pixels.append(_crop(image, detection, self.crop_h_w))
            vectors.append(model.encode(torch.from_numpy(np.stack(pixels)).to(self.device)).cpu().numpy())
        vectors = np.concatenate(vectors)
        means = {}
        for key, indices in slots.items():
            value = vectors[indices].mean(axis=0)
            if self.window > 1:
                value /= max(np.linalg.norm(value), 1e-8)
            means[key] = value
        result = []
        for start in range(0, len(rows), 512):
            chunk = rows[start:start + 512]
            left = np.stack([means[row["first"], "left"] for row in chunk])
            right = np.stack([means[row["second"], "right"] for row in chunk])
            if self.window > 1:
                left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-8)
                right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-8)
            raw = np.asarray([row["raw"] for row in chunk], np.float32)
            result.extend(model.pair_logits(
                torch.from_numpy(left).to(self.device),
                torch.from_numpy(right).to(self.device),
                torch.from_numpy(raw).to(self.device),
            ).sigmoid().cpu().tolist())
        values = np.asarray(result, np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite ResNetLink scores")
        return values, {
            "crops": len(requests),
            "window_lengths": dict(Counter(len(indices) for indices in slots.values())),
        }

    def process(
        self,
        native_path: Path,
        image_paths: list[Path],
        output: Path,
        feature_cache: Path,
        camera_matrices: np.ndarray | None = None,
        feature_cache_input: Path | None = None,
    ) -> dict:
        start = time.perf_counter()
        native_path, output, feature_cache = map(Path, (native_path, output, feature_cache))
        images = {int(Path(path).stem): Path(path) for path in image_paths}
        if not images or len(images) != len(image_paths):
            raise ValueError("Expected nonempty images with unique numeric frame names")
        frames = read_mots(native_path, target_class=self.class_id)
        if not frames:
            write_mots(output, [])
            return {"class_id": self.class_id, "rows": 0, "tracklets": 0,
                    "candidates": 0, "mask_preserved": True,
                    "seconds": time.perf_counter() - start}
        missing = set(frames) - set(images)
        if missing:
            raise FileNotFoundError(f"Missing image frames: {sorted(missing)[:8]}")
        scores_path = native_path.with_suffix(".scores.npz")
        if not scores_path.exists():
            raise FileNotFoundError(
                f"Native detection scores are required to preserve the deployed features: {scores_path}"
            )
        reid_stats = self._attach_features(frames, images, feature_cache, feature_cache_input)
        offset = min(frames)
        if self.camera_mode == "background_lk":
            if camera_matrices is None:
                raise ValueError("background_lk requires cumulative camera_matrices indexed by frame")
            matrices = np.asarray(camera_matrices)
            if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or len(matrices) <= max(frames):
                raise ValueError("Camera matrix shape must cover native frame indices")
            if not np.isfinite(matrices).all():
                raise ValueError("Camera matrices contain nonfinite values")
        else:
            matrices = np.tile(np.eye(3, dtype=np.float32), (max(images) + 1, 1, 1))
        tracker = _CameraTracker(list(matrices[offset:]), deepcopy(self.tracking))
        tracker.camera_mode = self.camera_mode
        tracks = tracker.prepare_refine_tracklets(frames)
        if tracker.frame_offset != offset:
            raise AssertionError("Camera matrices and tracklet frame offset differ")
        rows = _candidates(tracker, tracks)
        probabilities, boundary = self._score(tracks, rows, images)
        tracker.linker_override = _PrecomputedLinker(rows, probabilities, self.threshold, tracker)
        tracker.link_prepared_tracklets(tracks)
        detections = [detection for frame in frames.values() for detection in frame]
        write_mots(output, detections)
        if _mask_set(output) != _mask_set(native_path):
            raise AssertionError("Association changed native detections or masks")
        np.save(output.with_suffix(".link_scores.npy"), probabilities)
        return {
            "class_id": self.class_id,
            "window": self.window,
            "threshold": self.threshold,
            "camera_mode": self.camera_mode,
            "rows": len(detections),
            "tracklets": len(tracks),
            "candidates": len(rows),
            "admitted": int((probabilities >= self.threshold).sum()),
            "boundary": boundary,
            "reid": reid_stats,
            "tracking": tracker.stats,
            "mask_preserved": True,
            "native_sha256": _sha(native_path),
            "output_sha256": _sha(output),
            "seconds": time.perf_counter() - start,
        }
