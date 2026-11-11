from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from tqdm import tqdm

from .geometry import box_iou_matrix, transform_box
from .graph_budget import select_graph_budget
from .mots_io import Detection


def _compressed_rle_counts(value: str | bytes) -> np.ndarray:
    """Decode COCO's compressed run counts without expanding the image."""
    text = value.decode("ascii") if isinstance(value, bytes) else value
    counts: list[int] = []
    position = 0
    run_index = 0
    while position < len(text):
        decoded = 0
        shift = 0
        more = True
        while more:
            code = ord(text[position]) - 48
            position += 1
            decoded |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                decoded |= -1 << (5 * (shift + 1))
            shift += 1
        if run_index > 2:
            decoded += counts[run_index - 2]
        counts.append(decoded)
        run_index += 1
    return np.asarray(counts, dtype=np.int64)


def _decode_rle_crop(
    detection: Detection,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Decode exactly the requested RLE rectangle in column-major order.

    ``pycocotools.decode`` materialises a full HxW mask for every selected
    person even though ReID consumes only the bounding-box crop. Walking the
    foreground runs that intersect each crop column preserves identical mask
    pixels while avoiding the full-frame allocation.
    """
    crop_height = max(y2 - y1, 0)
    crop_width = max(x2 - x1, 0)
    crop = np.zeros((crop_height, crop_width), dtype=bool)
    if crop_height == 0 or crop_width == 0:
        return crop
    counts = _compressed_rle_counts(detection.counts)
    run_ends = np.cumsum(counts)
    run_starts = np.concatenate((np.zeros(1, np.int64), run_ends[:-1]))
    foreground_starts = run_starts[1::2]
    foreground_ends = run_ends[1::2]
    image_height = int(detection.height)
    # Expand foreground positions only inside the crop's column span. This is
    # proportional to person-mask area rather than full image area and keeps
    # the hot loop inside NumPy instead of iterating over pixels in Python.
    span_start = x1 * image_height
    span_end = x2 * image_height
    starts = np.maximum(foreground_starts, span_start)
    ends = np.minimum(foreground_ends, span_end)
    lengths = np.maximum(ends - starts, 0)
    keep_runs = lengths > 0
    starts = starts[keep_runs]
    lengths = lengths[keep_runs]
    foreground_area = int(lengths.sum())
    if foreground_area == 0:
        return crop
    run_offsets = np.concatenate((
        np.zeros(1, np.int64), np.cumsum(lengths)[:-1]
    ))
    flat_indices = (
        np.arange(foreground_area, dtype=np.int64)
        - np.repeat(run_offsets, lengths)
        + np.repeat(starts, lengths)
    )
    columns = flat_indices // image_height
    rows = flat_indices - columns * image_height
    inside = (rows >= y1) & (rows < y2)
    crop[rows[inside] - y1, columns[inside] - x1] = True
    return crop


class CausalBudgetSelector:
    """Stateful, frame-by-frame form of the V20 ReID budget selector."""

    def __init__(
        self,
        config: dict,
        homographies: list[np.ndarray] | None = None,
        frame_offset: int = 0,
    ) -> None:
        self.max_gap = max(int(config.get("max_gap", 1)), 1)
        self.initial_samples = max(int(config.get("initial_samples", 2)), 1)
        self.refresh_interval = max(int(config.get("refresh_interval", 18)), 1)
        self.top_k = max(int(config.get("top_k", 3)), 1)
        self.min_iou = float(config.get("minimum_predicted_iou", 0.25))
        self.max_center = float(config.get("maximum_center_distance", 0.60))
        self.max_area_ratio = float(config.get("maximum_area_ratio", 1.8))
        self.overlap_iou = float(config.get("overlap_iou", 0.30))
        self.ambiguity_min_score = float(config.get("ambiguity_min_score", 0.42))
        self.ambiguity_margin = float(config.get("ambiguity_margin", 0.10))
        self.mask_weight = float(config.get("mask_weight", 0.15))
        self.mask_pre_margin = float(config.get("mask_pre_margin", 0.18))
        self.budget_ratio = min(max(float(config.get("budget_ratio", 1.0)), 0.0), 1.0)
        self.mandatory_reasons = set(config.get("mandatory_reasons", ["new_identity"]))
        self.homographies = homographies
        self.frame_offset = int(frame_offset)
        self.last_by_source: dict[int, Detection] = {}
        self.observations_by_source: Counter = Counter()
        self.last_fresh_observation: dict[int, int] = {}
        self.selected: list[bool] = []
        self.reasons: list[str] = []
        self.candidate_pairs = 0
        self.retained_pairs = 0
        self.mask_pairs = 0
        self.ambiguous_observations = 0
        self.budget_slots = 0
        self.budget_used = 0
        self.budget_suppressed = 0

    def _predict(self, previous: Detection, frame: int) -> np.ndarray:
        if self.homographies is None:
            return previous.box
        source_index = previous.frame - self.frame_offset
        target_index = frame - self.frame_offset
        if not (
            0 <= source_index < len(self.homographies)
            and 0 <= target_index < len(self.homographies)
        ):
            return previous.box
        matrix = (
            self.homographies[target_index]
            @ np.linalg.inv(self.homographies[source_index])
        )
        return transform_box(previous.box, matrix)

    def select_frame(
        self, frame: int, detections: list[Detection]
    ) -> tuple[np.ndarray, list[str]]:
        """Select fresh embeddings using only current and past observations."""
        frame_choices: list[tuple[Detection, int, int, str, float, bool]] = []
        active_before = {
            identity: detection
            for identity, detection in self.last_by_source.items()
            if 0 < frame - detection.frame <= self.max_gap
        }
        boxes = (
            np.stack([detection.box for detection in detections])
            if detections else np.empty((0, 4), np.float32)
        )
        if detections:
            overlap_matrix = box_iou_matrix(boxes, boxes)
            np.fill_diagonal(overlap_matrix, 0.0)
            overlap_flags = np.any(overlap_matrix >= self.overlap_iou, axis=1)
        else:
            overlap_flags = np.empty(0, dtype=bool)
        active_ids = list(active_before)
        active_detections = [active_before[identity] for identity in active_ids]
        if active_detections and detections:
            predicted = np.stack([
                self._predict(item, frame) for item in active_detections
            ])
            iou_matrix = box_iou_matrix(predicted, boxes)
            predicted_centers = (predicted[:, :2] + predicted[:, 2:]) / 2
            detection_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            heights = np.maximum(
                np.maximum(
                    (predicted[:, 3] - predicted[:, 1])[:, None],
                    (boxes[:, 3] - boxes[:, 1])[None, :],
                ),
                25.0,
            )
            center_matrix = np.linalg.norm(
                predicted_centers[:, None] - detection_centers[None, :], axis=2
            ) / heights
            predicted_areas = (
                np.maximum(predicted[:, 2] - predicted[:, 0], 0.0)
                * np.maximum(predicted[:, 3] - predicted[:, 1], 0.0)
            )
            detection_areas = (
                np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
                * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
            )
            area_ratio_matrix = np.maximum(
                predicted_areas[:, None], detection_areas[None, :]
            ) / np.maximum(
                np.minimum(predicted_areas[:, None], detection_areas[None, :]), 1.0
            )
            geometric_matrix = 0.62 * iou_matrix + 0.38 * np.exp(-center_matrix)
        else:
            iou_matrix = center_matrix = area_ratio_matrix = geometric_matrix = (
                np.empty((len(active_ids), len(detections)), np.float32)
            )

        for detection_index, detection in enumerate(detections):
            source_id = detection.source_id
            previous = self.last_by_source.get(source_id)
            observation_number = self.observations_by_source[source_id]
            reason = "reuse"
            if previous is None:
                reason = "new_identity"
            elif frame - previous.frame > self.max_gap:
                reason = "reappearance"
            elif observation_number < self.initial_samples:
                reason = "identity_warmup"
            elif overlap_flags[detection_index]:
                reason = "current_overlap"
            else:
                self.candidate_pairs += len(active_ids)
                order = np.argsort(
                    -geometric_matrix[:, detection_index], kind="stable"
                )[: self.top_k]
                sparse = [
                    (
                        float(geometric_matrix[index, detection_index]),
                        active_ids[index],
                        active_detections[index],
                        float(iou_matrix[index, detection_index]),
                        float(center_matrix[index, detection_index]),
                        float(area_ratio_matrix[index, detection_index]),
                    )
                    for index in order
                ]
                if source_id in active_before and not any(
                    item[1] == source_id for item in sparse
                ):
                    own_index = active_ids.index(source_id)
                    own_item = (
                        float(geometric_matrix[own_index, detection_index]),
                        source_id,
                        active_detections[own_index],
                        float(iou_matrix[own_index, detection_index]),
                        float(center_matrix[own_index, detection_index]),
                        float(area_ratio_matrix[own_index, detection_index]),
                    )
                    if len(sparse) >= self.top_k:
                        sparse[-1] = own_item
                    else:
                        sparse.append(own_item)
                self.retained_pairs += len(sparse)
                geometric_own = next(
                    (item[0] for item in sparse if item[1] == source_id), None
                )
                geometric_alternative = max(
                    (item[0] for item in sparse if item[1] != source_id),
                    default=0.0,
                )
                inspect_masks = (
                    self.mask_weight > 0
                    and geometric_own is not None
                    and geometric_alternative
                    >= self.ambiguity_min_score - self.mask_pre_margin
                    and geometric_alternative
                    >= geometric_own - self.ambiguity_margin - self.mask_pre_margin
                )
                temporal_mask_ious = (
                    mask_utils.iou(
                        [item[2].rle for item in sparse],
                        [detection.rle],
                        [0],
                    )[:, 0]
                    if inspect_masks
                    else np.zeros(len(sparse), np.float32)
                )
                if inspect_masks:
                    self.mask_pairs += len(sparse)
                scored = []
                for (
                    geometric_score,
                    identity,
                    _candidate,
                    iou,
                    center,
                    area_ratio,
                ), temporal_mask_iou in zip(sparse, temporal_mask_ious):
                    score = geometric_score
                    if inspect_masks:
                        score = (
                            (1.0 - self.mask_weight) * geometric_score
                            + self.mask_weight * float(temporal_mask_iou)
                        )
                    scored.append((score, identity, iou, center, area_ratio))
                own = next(
                    (item for item in scored if item[1] == source_id), None
                )
                alternatives = [item for item in scored if item[1] != source_id]
                best_alternative = max(
                    (item[0] for item in alternatives), default=0.0
                )
                if own is None:
                    reason = "missing_predecessor"
                elif (
                    best_alternative >= self.ambiguity_min_score
                    and best_alternative >= own[0] - self.ambiguity_margin
                ):
                    reason = "ambiguous_sparse_graph"
                    self.ambiguous_observations += 1
                elif (
                    own[2] < self.min_iou
                    or own[3] > self.max_center
                    or own[4] > self.max_area_ratio
                ):
                    reason = "geometry_change"
                elif (
                    observation_number
                    - self.last_fresh_observation.get(source_id, -1)
                    >= self.refresh_interval
                ):
                    reason = "periodic_refresh"
            priority = {
                "ambiguous_sparse_graph": 1.00,
                "current_overlap": 0.90,
                "reappearance": 0.85,
                "missing_predecessor": 0.85,
                "geometry_change": 0.80,
                "identity_warmup": 0.65,
                "periodic_refresh": 0.50,
            }.get(reason, 0.0)
            frame_choices.append(
                (
                    detection,
                    source_id,
                    observation_number,
                    reason,
                    priority,
                    reason in self.mandatory_reasons,
                )
            )

        proposed = [
            index for index, choice in enumerate(frame_choices)
            if choice[3] != "reuse"
        ]
        mandatory = {index for index in proposed if frame_choices[index][5]}
        if self.budget_ratio >= 1.0:
            frame_limit = len(detections)
        else:
            frame_limit = int(math.ceil(self.budget_ratio * len(detections)))
        optional_limit = max(frame_limit - len(mandatory), 0)
        optional = sorted(
            (index for index in proposed if index not in mandatory),
            key=lambda index: (-frame_choices[index][4], index),
        )
        accepted = mandatory | set(optional[:optional_limit])
        self.budget_slots += frame_limit
        self.budget_used += len(accepted)
        self.budget_suppressed += len(proposed) - len(accepted)

        frame_selected: list[bool] = []
        frame_reasons: list[str] = []
        for index, (
            detection,
            source_id,
            observation_number,
            reason,
            _priority,
            _mandatory,
        ) in enumerate(frame_choices):
            is_fresh = index in accepted
            final_reason = reason if is_fresh or reason == "reuse" else "budget_reuse"
            frame_selected.append(is_fresh)
            frame_reasons.append(final_reason)
            if is_fresh:
                self.last_fresh_observation[source_id] = observation_number
            self.last_by_source[source_id] = detection
            self.observations_by_source[source_id] += 1
        self.selected.extend(frame_selected)
        self.reasons.extend(frame_reasons)
        return np.asarray(frame_selected, dtype=bool), frame_reasons

    def summary(self) -> dict[str, int | float]:
        total = len(self.selected)
        return {
            "reid_candidate_pairs": self.candidate_pairs,
            "reid_retained_pairs": self.retained_pairs,
            "reid_mask_pairs": self.mask_pairs,
            "reid_candidate_keep_ratio": self.retained_pairs / max(self.candidate_pairs, 1),
            "reid_ambiguous_observations": self.ambiguous_observations,
            "reid_budget_ratio": self.budget_ratio,
            "reid_budget_slots": self.budget_slots,
            "reid_budget_used": self.budget_used,
            "reid_budget_suppressed": self.budget_suppressed,
            "reid_budget_selected_ratio": sum(self.selected) / max(total, 1),
        }


class AppearanceEncoder:
    """Person ReID encoder with OSNet primary and ResNet-50 fallback."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 128,
        backbone: str = "osnet_x1_0",
        weights: Path | None = None,
        mask_background: bool = True,
        crop_rle_decode: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.yolo_reid = None
        if backbone.startswith("osnet"):
            import torchreid
            model = torchreid.models.build_model(
                name=backbone, num_classes=1, pretrained=False,
                use_gpu=self.device.type == "cuda",
            )
            if weights is None or not weights.exists():
                raise FileNotFoundError(f"OSNet weights not found: {weights}")
            torchreid.utils.load_pretrained_weights(model, str(weights))
            self.model = model.to(self.device).eval()
            self.output_dim = 512
        elif backbone == "resnet18_imagenet":
            if weights is not None and weights.exists():
                model = resnet18(weights=None)
                model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            else:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Identity()
            self.model = model.to(self.device).eval()
            self.output_dim = 512
        elif backbone == "yolo26n_cls":
            if weights is None or not weights.exists():
                raise FileNotFoundError(f"YOLO classification weights not found: {weights}")
            from ultralytics.trackers.utils.reid import ReID
            self.yolo_reid = ReID(str(weights), device=self.device)
            self.model = None
            self.output_dim = 256
        elif backbone in {"yolo26n_reid", "yolo26m_reid", "yolo26x_reid"}:
            if weights is None or not weights.exists():
                raise FileNotFoundError(f"YOLO ReID weights not found: {weights}")
            from ultralytics.trackers.utils.reid import ReID
            # AutoBackend passes the CUDA device index to ONNX Runtime.  A
            # bare torch.device("cuda") has index=None and silently falls back
            # to CPU, so make the first device explicit for this path.
            reid_device = self.device
            if reid_device.type == "cuda" and reid_device.index is None:
                reid_device = torch.device("cuda:0")
            self.yolo_reid = ReID(
                str(weights), imgsz=448, device=reid_device,
            )
            self.model = None
            self.output_dim = 512
        else:
            # Official torchvision checkpoint provides a network-independent
            # fallback for non-person categories.
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
            self.output_dim = 2048
        self.batch_size = batch_size
        self.mask_background = bool(mask_background)
        self.crop_rle_decode = bool(crop_rle_decode)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    @torch.inference_mode()
    def encode_sequence(
        self,
        image_by_frame: Dict[int, Path],
        frames: Dict[int, List[Detection]],
        cache: Path | None = None,
        policy: str = "all",
        event_config: dict | None = None,
        homographies: list[np.ndarray] | None = None,
    ) -> dict:
        ordered = [det for frame in sorted(frames) for det in frames[frame]]
        selection_stats: dict[str, int | float] = {}
        if cache and cache.exists():
            cached = np.load(cache)
            features = cached["features"]
            if len(features) == len(ordered):
                if "selected" in cached.files:
                    selected = cached["selected"].astype(bool)
                    reasons = (
                        cached["reasons"].astype(str).tolist()
                        if "reasons" in cached.files
                        else [
                            "cached_fresh" if fresh else "cached_reuse"
                            for fresh in selected
                        ]
                    )
                elif policy == "event_driven":
                    selected, reasons = self._event_selection(
                        frames, event_config or {}, homographies
                    )
                elif policy in (
                    "budget_aware", "all_fresh_budget_metadata",
                    "uniform_periodic_budget_metadata",
                ):
                    selected, reasons, selection_stats = (
                        self._budget_selection(
                            frames, event_config or {}, homographies
                        )
                    )
                    if policy == "all_fresh_budget_metadata":
                        selection_stats["reid_metadata_budget_selected_instances"] = int(selected.sum())
                        selected = np.ones(len(ordered), dtype=bool)
                    elif policy == "uniform_periodic_budget_metadata":
                        selection_stats["reid_metadata_budget_selected_instances"] = int(selected.sum())
                        selected = self._uniform_periodic_selection(
                            frames, int((event_config or {}).get("uniform_interval", 4))
                        )
                elif policy == "event_driven_vectorized":
                    selected, reasons, selection_stats = (
                        self._event_selection_vectorized(
                            frames, event_config or {}, homographies
                        )
                    )
                elif policy == "graph_budget":
                    selected, reasons, selection_stats = select_graph_budget(
                        frames, event_config or {}, homographies
                    )
                else:
                    selected = np.ones(len(ordered), dtype=bool)
                    reasons = ["all"] * len(ordered)
                for det, feature, fresh, reason in zip(
                    ordered, features, selected, reasons
                ):
                    det.feature = feature.astype(np.float32)
                    if not (
                        "normalized" in cached.files
                        and bool(cached["normalized"][0])
                    ):
                        det.feature /= np.linalg.norm(det.feature) + 1e-12
                    det.feature_is_fresh = bool(fresh)
                    det.feature_reason = str(reason)
                encoded = int(
                    cached["encoded_instances"][0]
                    if "encoded_instances" in cached.files
                    else selected.sum()
                )
                result = {
                    "reid_policy": policy,
                    "reid_total_instances": len(ordered),
                    "reid_encoded_instances": encoded,
                    "reid_skipped_instances": len(ordered) - encoded,
                    "reid_encoded_ratio": encoded / max(len(ordered), 1),
                    "reid_seconds": 0.0,
                    "reid_selection_reasons": {"cache": encoded},
                    "reid_cache_hit": True,
                }
                result.update(selection_stats)
                return result
        start = time.perf_counter()
        selection_start = start
        if policy == "all":
            selected = np.ones(len(ordered), dtype=bool)
            reasons = ["all"] * len(ordered)
        elif policy == "event_driven":
            selected, reasons = self._event_selection(
                frames,
                event_config or {},
                homographies,
            )
        elif policy in (
            "budget_aware", "all_fresh_budget_metadata",
            "uniform_periodic_budget_metadata",
        ):
            selected, reasons, selection_stats = self._budget_selection(
                frames,
                event_config or {},
                homographies,
            )
            if policy == "all_fresh_budget_metadata":
                # Preserve V42-B1's event labels for ambiguity splitting while
                # making the actual OSNet encoding decision fresh for every
                # detection instance.  This keeps the ablation from silently
                # disabling event-triggered tracklet splitting.
                selection_stats["reid_metadata_budget_selected_instances"] = int(selected.sum())
                selected = np.ones(len(ordered), dtype=bool)
            elif policy == "uniform_periodic_budget_metadata":
                selection_stats["reid_metadata_budget_selected_instances"] = int(selected.sum())
                selected = self._uniform_periodic_selection(
                    frames, int((event_config or {}).get("uniform_interval", 4))
                )
        elif policy == "event_driven_vectorized":
            selected, reasons, selection_stats = (
                self._event_selection_vectorized(
                    frames,
                    event_config or {},
                    homographies,
                )
            )
        elif policy == "graph_budget":
            selected, reasons, selection_stats = select_graph_budget(
                frames,
                event_config or {},
                homographies,
            )
        else:
            raise ValueError(f"Unknown ReID policy: {policy}")

        selection_seconds = time.perf_counter() - selection_start

        crops, owners = [], []
        output = np.empty((len(ordered), self.output_dim), np.float32)
        cursor = 0
        image_read_seconds = 0.0
        crop_seconds = 0.0
        model_seconds = 0.0
        model_batches = 0
        for frame in tqdm(sorted(frames), desc="ReID crops"):
            image_start = time.perf_counter()
            image = cv2.imread(str(image_by_frame[frame]))
            image_read_seconds += time.perf_counter() - image_start
            if image is None:
                raise FileNotFoundError(image_by_frame[frame])
            for det in frames[frame]:
                if selected[cursor]:
                    crop_start = time.perf_counter()
                    crops.append(self._crop(image, det))
                    crop_seconds += time.perf_counter() - crop_start
                    owners.append(cursor)
                cursor += 1
                if len(crops) >= self.batch_size:
                    model_start = time.perf_counter()
                    self._flush(crops, owners, output)
                    model_seconds += time.perf_counter() - model_start
                    model_batches += 1
        if crops:
            model_start = time.perf_counter()
            self._flush(crops, owners, output)
            model_seconds += time.perf_counter() - model_start
            model_batches += 1
        assignment_start = time.perf_counter()
        last_feature: dict[int, np.ndarray] = {}
        for index, det in enumerate(ordered):
            reuse_key = int(getattr(det, "reid_key", det.source_id))
            if selected[index]:
                feature = output[index]
                last_feature[reuse_key] = feature
            else:
                feature = last_feature.get(reuse_key)
                if feature is None:
                    raise RuntimeError(
                        "ReID policy skipped an identity before its first "
                        f"encoding: {reuse_key}"
                    )
                output[index] = feature
            det.feature = output[index]
            det.feature_is_fresh = bool(selected[index])
            det.feature_reason = str(reasons[index])
        assignment_seconds = time.perf_counter() - assignment_start
        encoded = int(selected.sum())
        cache_write_seconds = 0.0
        if cache:
            cache_start = time.perf_counter()
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                features=output.astype(np.float16),
                encoded_instances=np.asarray([encoded], np.int32),
                selected=selected.astype(np.uint8),
                reasons=np.asarray(reasons, dtype="<U32"),
            )
            cache_write_seconds = time.perf_counter() - cache_start
        total_seconds = time.perf_counter() - start
        result = {
            "reid_policy": policy,
            "reid_total_instances": len(ordered),
            "reid_encoded_instances": encoded,
            "reid_skipped_instances": len(ordered) - encoded,
            "reid_encoded_ratio": encoded / max(len(ordered), 1),
            "reid_seconds": total_seconds,
            "reid_selection_seconds": selection_seconds,
            "reid_image_read_seconds": image_read_seconds,
            "reid_crop_seconds": crop_seconds,
            "reid_model_seconds": model_seconds,
            "reid_model_batches": model_batches,
            "reid_assignment_seconds": assignment_seconds,
            "reid_cache_write_seconds": cache_write_seconds,
            "reid_profile_overhead_seconds": max(
                total_seconds
                - selection_seconds
                - image_read_seconds
                - crop_seconds
                - model_seconds
                - assignment_seconds
                - cache_write_seconds,
                0.0,
            ),
            "reid_selection_reasons": dict(Counter(reasons)),
            "reid_cache_hit": False,
        }
        result.update(selection_stats)
        return result

    @staticmethod
    def _uniform_periodic_selection(
        frames: Dict[int, List[Detection]], interval: int,
    ) -> np.ndarray:
        """Select the first and every Nth observation of each upstream ID."""
        interval = max(int(interval), 1)
        seen: Counter = Counter()
        selected = []
        for frame in sorted(frames):
            for detection in frames[frame]:
                index = seen[detection.source_id]
                selected.append(index % interval == 0)
                seen[detection.source_id] += 1
        return np.asarray(selected, dtype=bool)

    @staticmethod
    def _event_selection_vectorized(
        frames: Dict[int, List[Detection]],
        config: dict,
        homographies: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, list[str], dict[str, int | float]]:
        """Vectorized implementation of the frozen V17 event policy.

        The decision rules and their order intentionally mirror
        :meth:`_event_selection`. Only pairwise box geometry is evaluated in
        NumPy matrices rather than Python nested loops. Head/tail sampling is
        retained to make this a strict implementation baseline, even though
        the tail rule is not causal.
        """
        start = time.perf_counter()
        max_gap = int(config.get("max_gap", 1))
        initial_samples = int(config.get("initial_samples", 3))
        head_samples = max(int(config.get("head_samples", 0)), 0)
        tail_samples = max(int(config.get("tail_samples", 0)), 0)
        sparse_person_threshold = float(
            config.get("sparse_person_threshold", 10.0)
        )
        density_ema = float(config.get("density_ema", 0.90))
        min_iou = float(config.get("minimum_predicted_iou", 0.35))
        max_center = float(config.get("maximum_center_distance", 0.45))
        max_area_ratio = float(config.get("maximum_area_ratio", 1.5))
        overlap_iou = float(config.get("overlap_iou", 0.30))
        ambiguity_iou = float(config.get("ambiguity_iou", 0.10))
        ambiguity_center = float(
            config.get("ambiguity_center_distance", 0.80)
        )
        ambiguity_max_gap = int(config.get("ambiguity_max_gap", 1))
        ambiguity_min_score = float(
            config.get("ambiguity_min_score", 0.45)
        )
        ambiguity_margin = float(config.get("ambiguity_margin", 0.08))
        frame_numbers = sorted(frames)
        frame_offset = frame_numbers[0]
        selected: list[bool] = []
        reasons: list[str] = []
        last_by_source: dict[int, Detection] = {}
        observations_by_source: Counter = Counter()
        indices_by_source: dict[int, list[int]] = {}
        rolling_density: float | None = None
        pair_count = 0

        def predict(previous: Detection, frame: int) -> np.ndarray:
            if homographies is None:
                return previous.box
            source_index = previous.frame - frame_offset
            target_index = frame - frame_offset
            if not (
                0 <= source_index < len(homographies)
                and 0 <= target_index < len(homographies)
            ):
                return previous.box
            matrix = (
                homographies[target_index]
                @ np.linalg.inv(homographies[source_index])
            )
            return transform_box(previous.box, matrix)

        for frame in frame_numbers:
            detections = frames[frame]
            rolling_density = (
                float(len(detections))
                if rolling_density is None
                else density_ema * rolling_density
                + (1.0 - density_ema) * len(detections)
            )
            warmup_limit = (
                initial_samples
                if rolling_density <= sparse_person_threshold
                else 1
            )
            active_before = {
                identity: detection
                for identity, detection in last_by_source.items()
                if 0 < frame - detection.frame <= max(
                    max_gap, ambiguity_max_gap
                )
            }
            boxes = (
                np.stack([detection.box for detection in detections])
                if detections else np.empty((0, 4), np.float32)
            )
            if detections:
                current_iou = box_iou_matrix(boxes, boxes)
                np.fill_diagonal(current_iou, 0.0)
                overlap_flags = np.any(
                    current_iou >= overlap_iou, axis=1
                )
            else:
                overlap_flags = np.empty(0, dtype=bool)

            active_ids = list(active_before)
            active_detections = [
                active_before[identity] for identity in active_ids
            ]
            if active_detections and detections:
                predicted = np.stack(
                    [predict(item, frame) for item in active_detections]
                )
                iou_matrix = box_iou_matrix(predicted, boxes)
                predicted_centers = (
                    predicted[:, :2] + predicted[:, 2:]
                ) / 2
                detection_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                heights = np.maximum(
                    np.maximum(
                        (predicted[:, 3] - predicted[:, 1])[:, None],
                        (boxes[:, 3] - boxes[:, 1])[None, :],
                    ),
                    25.0,
                )
                center_matrix = np.linalg.norm(
                    predicted_centers[:, None]
                    - detection_centers[None, :],
                    axis=2,
                ) / heights
                predicted_areas = np.maximum(
                    predicted[:, 2] - predicted[:, 0], 0.0
                ) * np.maximum(predicted[:, 3] - predicted[:, 1], 0.0)
                detection_areas = np.maximum(
                    boxes[:, 2] - boxes[:, 0], 0.0
                ) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
                area_ratio_matrix = np.maximum(
                    predicted_areas[:, None], detection_areas[None, :]
                ) / np.maximum(
                    np.minimum(
                        predicted_areas[:, None], detection_areas[None, :]
                    ),
                    1.0,
                )
                score_matrix = (
                    0.65 * iou_matrix
                    + 0.35 * np.exp(-center_matrix)
                )
            else:
                shape = (len(active_ids), len(detections))
                iou_matrix = np.empty(shape, np.float32)
                center_matrix = np.empty(shape, np.float32)
                area_ratio_matrix = np.empty(shape, np.float32)
                score_matrix = np.empty(shape, np.float32)
            active_row = {
                identity: index for index, identity in enumerate(active_ids)
            }
            pair_count += len(active_ids) * len(detections)

            frame_choices = []
            for detection_index, detection in enumerate(detections):
                previous = last_by_source.get(detection.source_id)
                if previous is None:
                    reason = "new_identity"
                elif detection.frame - previous.frame > max_gap:
                    reason = "reappearance"
                elif observations_by_source[detection.source_id] < warmup_limit:
                    reason = "identity_warmup"
                elif bool(overlap_flags[detection_index]):
                    reason = "current_overlap"
                else:
                    own_prediction = predict(previous, frame)
                    own_box = own_prediction[None]
                    own_iou = float(
                        box_iou_matrix(own_box, detection.box[None])[0, 0]
                    )
                    own_center_point = (
                        own_prediction[:2] + own_prediction[2:]
                    ) / 2
                    detection_center = (
                        detection.box[:2] + detection.box[2:]
                    ) / 2
                    own_height = max(
                        own_prediction[3] - own_prediction[1],
                        detection.box[3] - detection.box[1],
                        25.0,
                    )
                    own_center = float(
                        np.linalg.norm(
                            own_center_point - detection_center
                        ) / own_height
                    )
                    own_area = float(np.prod(np.maximum(
                        own_prediction[2:] - own_prediction[:2], 0.0
                    )))
                    detection_area = float(np.prod(np.maximum(
                        detection.box[2:] - detection.box[:2], 0.0
                    )))
                    own_area_ratio = max(
                        own_area, detection_area
                    ) / max(min(own_area, detection_area), 1.0)
                    own_score = (
                        0.65 * own_iou
                        + 0.35 * float(np.exp(-own_center))
                    )
                    valid_alternatives = np.ones(
                        len(active_ids), dtype=bool
                    )
                    own_row = active_row.get(detection.source_id)
                    if own_row is not None:
                        valid_alternatives[own_row] = False
                    if len(active_ids):
                        valid_alternatives &= (
                            (iou_matrix[:, detection_index] >= ambiguity_iou)
                            | (
                                center_matrix[:, detection_index]
                                <= ambiguity_center
                            )
                        )
                        best_alternative = float(np.max(
                            score_matrix[valid_alternatives, detection_index],
                            initial=0.0,
                        ))
                    else:
                        best_alternative = 0.0
                    if (
                        best_alternative >= ambiguity_min_score
                        and best_alternative
                        >= own_score - ambiguity_margin
                    ):
                        reason = "ambiguous_candidates"
                    elif (
                        own_iou < min_iou
                        or own_center > max_center
                        or own_area_ratio > max_area_ratio
                    ):
                        reason = "geometry_change"
                    else:
                        reason = "reuse"
                frame_choices.append(reason)
            for detection, reason in zip(detections, frame_choices):
                observation_index = len(selected)
                selected.append(reason != "reuse")
                reasons.append(reason)
                last_by_source[detection.source_id] = detection
                observations_by_source[detection.source_id] += 1
                indices_by_source.setdefault(
                    detection.source_id, []
                ).append(observation_index)

        for indices in indices_by_source.values():
            for observation_index in indices[:head_samples]:
                if not selected[observation_index]:
                    selected[observation_index] = True
                    reasons[observation_index] = "identity_head"
            tail_indices = indices[-tail_samples:] if tail_samples else []
            for observation_index in tail_indices:
                if not selected[observation_index]:
                    selected[observation_index] = True
                    reasons[observation_index] = "identity_tail"
        return (
            np.asarray(selected, dtype=bool),
            reasons,
            {
                "reid_selector_seconds": time.perf_counter() - start,
                "reid_geometry_pairs": pair_count,
                "reid_vectorized_selector": True,
            },
        )

    @staticmethod
    def _budget_selection(
        frames: Dict[int, List[Detection]],
        config: dict,
        homographies: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, list[str], dict[str, int | float]]:
        """Causal, budget-aware selection of fresh appearance features.

        A cheap geometric pass first keeps at most ``top_k`` predecessor
        candidates for each observation. Temporal mask IoU is evaluated only
        inside that sparse candidate set. ReID is requested when the upstream
        identity is new, stale, geometrically unreliable, or not separated
        from the best alternative by a safe association margin.

        Unlike the legacy event policy, this method never marks trajectory
        tails after seeing the future. A periodic causal refresh maintains a
        recent appearance prototype for later tracklet linking.
        """
        frame_numbers = sorted(frames)
        if not frame_numbers:
            return np.empty(0, dtype=bool), [], CausalBudgetSelector(
                config, homographies
            ).summary()
        selector = CausalBudgetSelector(
            config, homographies, frame_offset=frame_numbers[0]
        )
        for frame in frame_numbers:
            selector.select_frame(frame, frames[frame])
        return (
            np.asarray(selector.selected, dtype=bool),
            selector.reasons,
            selector.summary(),
        )

        # The implementation below is intentionally unreachable during the
        # equivalence check and is removed once the frozen hashes pass.
        max_gap = max(int(config.get("max_gap", 1)), 1)
        initial_samples = max(int(config.get("initial_samples", 2)), 1)
        refresh_interval = max(int(config.get("refresh_interval", 18)), 1)
        top_k = max(int(config.get("top_k", 3)), 1)
        min_iou = float(config.get("minimum_predicted_iou", 0.25))
        max_center = float(config.get("maximum_center_distance", 0.60))
        max_area_ratio = float(config.get("maximum_area_ratio", 1.8))
        overlap_iou = float(config.get("overlap_iou", 0.30))
        ambiguity_min_score = float(
            config.get("ambiguity_min_score", 0.42)
        )
        ambiguity_margin = float(config.get("ambiguity_margin", 0.10))
        mask_weight = float(config.get("mask_weight", 0.15))
        mask_pre_margin = float(config.get("mask_pre_margin", 0.18))
        budget_ratio = min(
            max(float(config.get("budget_ratio", 1.0)), 0.0), 1.0
        )
        mandatory_reasons = set(
            config.get(
                "mandatory_reasons",
                ["new_identity"],
            )
        )
        frame_numbers = sorted(frames)
        frame_offset = frame_numbers[0]
        selected: list[bool] = []
        reasons: list[str] = []
        last_by_source: dict[int, Detection] = {}
        observations_by_source: Counter = Counter()
        last_fresh_observation: dict[int, int] = {}
        candidate_pairs = 0
        retained_pairs = 0
        mask_pairs = 0
        ambiguous_observations = 0
        budget_slots = 0
        budget_used = 0
        budget_suppressed = 0

        def predict(previous: Detection, frame: int) -> np.ndarray:
            if homographies is None:
                return previous.box
            source_index = previous.frame - frame_offset
            target_index = frame - frame_offset
            if not (
                0 <= source_index < len(homographies)
                and 0 <= target_index < len(homographies)
            ):
                return previous.box
            matrix = (
                homographies[target_index]
                @ np.linalg.inv(homographies[source_index])
            )
            return transform_box(previous.box, matrix)

        for frame in frame_numbers:
            detections = frames[frame]
            frame_choices: list[
                tuple[Detection, int, int, str, float, bool]
            ] = []
            active_before = {
                identity: detection
                for identity, detection in last_by_source.items()
                if 0 < frame - detection.frame <= max_gap
            }
            boxes = (
                np.stack([detection.box for detection in detections])
                if detections else np.empty((0, 4), np.float32)
            )
            if detections:
                overlap_matrix = box_iou_matrix(boxes, boxes)
                np.fill_diagonal(overlap_matrix, 0.0)
                overlap_flags = np.any(
                    overlap_matrix >= overlap_iou, axis=1
                )
            else:
                overlap_flags = np.empty(0, dtype=bool)
            active_ids = list(active_before)
            active_detections = [
                active_before[identity] for identity in active_ids
            ]
            if active_detections and detections:
                predicted = np.stack(
                    [predict(item, frame) for item in active_detections]
                )
                iou_matrix = box_iou_matrix(predicted, boxes)
                predicted_centers = (
                    predicted[:, :2] + predicted[:, 2:]
                ) / 2
                detection_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                heights = np.maximum(
                    np.maximum(
                        (predicted[:, 3] - predicted[:, 1])[:, None],
                        (boxes[:, 3] - boxes[:, 1])[None, :],
                    ),
                    25.0,
                )
                center_matrix = np.linalg.norm(
                    predicted_centers[:, None]
                    - detection_centers[None, :],
                    axis=2,
                ) / heights
                predicted_areas = np.maximum(
                    predicted[:, 2] - predicted[:, 0], 0.0
                ) * np.maximum(predicted[:, 3] - predicted[:, 1], 0.0)
                detection_areas = np.maximum(
                    boxes[:, 2] - boxes[:, 0], 0.0
                ) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
                area_ratio_matrix = np.maximum(
                    predicted_areas[:, None], detection_areas[None, :]
                ) / np.maximum(
                    np.minimum(
                        predicted_areas[:, None], detection_areas[None, :]
                    ),
                    1.0,
                )
                geometric_matrix = (
                    0.62 * iou_matrix
                    + 0.38 * np.exp(-center_matrix)
                )
            else:
                iou_matrix = center_matrix = area_ratio_matrix = (
                    geometric_matrix
                ) = np.empty((len(active_ids), len(detections)), np.float32)
            for detection_index, detection in enumerate(detections):
                source_id = detection.source_id
                previous = last_by_source.get(source_id)
                observation_number = observations_by_source[source_id]
                reason = "reuse"
                if previous is None:
                    reason = "new_identity"
                elif frame - previous.frame > max_gap:
                    reason = "reappearance"
                elif observation_number < initial_samples:
                    reason = "identity_warmup"
                elif overlap_flags[detection_index]:
                    # Crowded observations are retained because these are
                    # precisely the frames where an upstream short-term ID
                    # can switch between two people.
                    reason = "current_overlap"
                else:
                    candidate_pairs += len(active_ids)
                    order = np.argsort(
                        -geometric_matrix[:, detection_index], kind="stable"
                    )[:top_k]
                    sparse = [
                        (
                            float(geometric_matrix[index, detection_index]),
                            active_ids[index],
                            active_detections[index],
                            float(iou_matrix[index, detection_index]),
                            float(center_matrix[index, detection_index]),
                            float(area_ratio_matrix[index, detection_index]),
                        )
                        for index in order
                    ]
                    # Always retain the upstream predecessor, even if rapid
                    # motion pushed it outside the geometric Top-K set.
                    if source_id in active_before and not any(
                        item[1] == source_id for item in sparse
                    ):
                        own_index = active_ids.index(source_id)
                        own_item = (
                            float(geometric_matrix[own_index, detection_index]),
                            source_id,
                            active_detections[own_index],
                            float(iou_matrix[own_index, detection_index]),
                            float(center_matrix[own_index, detection_index]),
                            float(area_ratio_matrix[own_index, detection_index]),
                        )
                        if len(sparse) >= top_k:
                            sparse[-1] = own_item
                        else:
                            sparse.append(own_item)
                    retained_pairs += len(sparse)
                    geometric_own = next(
                        (item[0] for item in sparse if item[1] == source_id),
                        None,
                    )
                    geometric_alternative = max(
                        (item[0] for item in sparse if item[1] != source_id),
                        default=0.0,
                    )
                    inspect_masks = (
                        mask_weight > 0
                        and geometric_own is not None
                        and geometric_alternative
                        >= ambiguity_min_score - mask_pre_margin
                        and geometric_alternative
                        >= geometric_own - ambiguity_margin - mask_pre_margin
                    )
                    temporal_mask_ious = (
                        mask_utils.iou(
                            [item[2].rle for item in sparse],
                            [detection.rle],
                            [0],
                        )[:, 0]
                        if inspect_masks
                        else np.zeros(len(sparse), np.float32)
                    )
                    if inspect_masks:
                        mask_pairs += len(sparse)
                    scored = []
                    for (
                        geometric_score, identity, _candidate,
                        iou, center, area_ratio,
                    ), temporal_mask_iou in zip(
                        sparse, temporal_mask_ious
                    ):
                        score = geometric_score
                        if inspect_masks:
                            score = (
                                (1.0 - mask_weight) * geometric_score
                                + mask_weight * float(temporal_mask_iou)
                            )
                        scored.append(
                            (
                                score, identity, iou,
                                center, area_ratio,
                            )
                        )
                    own = next(
                        (item for item in scored if item[1] == source_id),
                        None,
                    )
                    alternatives = [
                        item for item in scored if item[1] != source_id
                    ]
                    best_alternative = max(
                        (item[0] for item in alternatives), default=0.0
                    )
                    if own is None:
                        reason = "missing_predecessor"
                    elif (
                        best_alternative >= ambiguity_min_score
                        and best_alternative >= own[0] - ambiguity_margin
                    ):
                        reason = "ambiguous_sparse_graph"
                        ambiguous_observations += 1
                    elif (
                        own[2] < min_iou
                        or own[3] > max_center
                        or own[4] > max_area_ratio
                    ):
                        reason = "geometry_change"
                    elif (
                        observation_number
                        - last_fresh_observation.get(source_id, -1)
                        >= refresh_interval
                    ):
                        reason = "periodic_refresh"
                priority = {
                    "ambiguous_sparse_graph": 1.00,
                    "current_overlap": 0.90,
                    "reappearance": 0.85,
                    "missing_predecessor": 0.85,
                    "geometry_change": 0.80,
                    "identity_warmup": 0.65,
                    "periodic_refresh": 0.50,
                }.get(reason, 0.0)
                frame_choices.append(
                    (
                        detection,
                        source_id,
                        observation_number,
                        reason,
                        priority,
                        reason in mandatory_reasons,
                    )
                )

            proposed = [
                index for index, choice in enumerate(frame_choices)
                if choice[3] != "reuse"
            ]
            mandatory = {
                index for index in proposed if frame_choices[index][5]
            }
            if budget_ratio >= 1.0:
                frame_limit = len(detections)
            else:
                frame_limit = int(math.ceil(budget_ratio * len(detections)))
            optional_limit = max(frame_limit - len(mandatory), 0)
            optional = sorted(
                (index for index in proposed if index not in mandatory),
                key=lambda index: (-frame_choices[index][4], index),
            )
            accepted = mandatory | set(optional[:optional_limit])
            budget_slots += frame_limit
            budget_used += len(accepted)
            budget_suppressed += len(proposed) - len(accepted)

            for index, (
                detection,
                source_id,
                observation_number,
                reason,
                _priority,
                _mandatory,
            ) in enumerate(frame_choices):
                is_fresh = index in accepted
                selected.append(is_fresh)
                reasons.append(reason if is_fresh or reason == "reuse" else "budget_reuse")
                if is_fresh:
                    last_fresh_observation[source_id] = observation_number
                last_by_source[source_id] = detection
                observations_by_source[source_id] += 1

        total = len(selected)
        return (
            np.asarray(selected, dtype=bool),
            reasons,
            {
                "reid_candidate_pairs": candidate_pairs,
                "reid_retained_pairs": retained_pairs,
                "reid_mask_pairs": mask_pairs,
                "reid_candidate_keep_ratio": (
                    retained_pairs / max(candidate_pairs, 1)
                ),
                "reid_ambiguous_observations": ambiguous_observations,
                "reid_budget_ratio": budget_ratio,
                "reid_budget_slots": budget_slots,
                "reid_budget_used": budget_used,
                "reid_budget_suppressed": budget_suppressed,
                "reid_budget_selected_ratio": (
                    sum(selected) / max(total, 1)
                ),
            },
        )

    @staticmethod
    def _event_selection(
        frames: Dict[int, List[Detection]],
        config: dict,
        homographies: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, list[str]]:
        """Select observations requiring a fresh appearance embedding.

        Decisions use source-track continuity, CMC-normalized geometry,
        current-frame overlap and ambiguity among recent source tracks.
        """
        max_gap = int(config.get("max_gap", 1))
        initial_samples = int(config.get("initial_samples", 3))
        head_samples = max(int(config.get("head_samples", 0)), 0)
        tail_samples = max(int(config.get("tail_samples", 0)), 0)
        sparse_person_threshold = float(
            config.get("sparse_person_threshold", 10.0)
        )
        density_ema = float(config.get("density_ema", 0.90))
        min_iou = float(config.get("minimum_predicted_iou", 0.35))
        max_center = float(config.get("maximum_center_distance", 0.45))
        max_area_ratio = float(config.get("maximum_area_ratio", 1.5))
        overlap_iou = float(config.get("overlap_iou", 0.30))
        ambiguity_iou = float(config.get("ambiguity_iou", 0.10))
        ambiguity_center = float(
            config.get("ambiguity_center_distance", 0.80)
        )
        ambiguity_max_gap = int(config.get("ambiguity_max_gap", 1))
        ambiguity_min_score = float(
            config.get("ambiguity_min_score", 0.45)
        )
        ambiguity_margin = float(config.get("ambiguity_margin", 0.08))
        frame_numbers = sorted(frames)
        frame_offset = frame_numbers[0]
        selected: list[bool] = []
        reasons: list[str] = []
        last_by_source: dict[int, Detection] = {}
        observations_by_source: Counter = Counter()
        indices_by_source: dict[int, list[int]] = {}
        rolling_density: float | None = None

        def predict(previous: Detection, frame: int) -> np.ndarray:
            if homographies is None:
                return previous.box
            source_index = previous.frame - frame_offset
            target_index = frame - frame_offset
            if not (
                0 <= source_index < len(homographies)
                and 0 <= target_index < len(homographies)
            ):
                return previous.box
            matrix = (
                homographies[target_index]
                @ np.linalg.inv(homographies[source_index])
            )
            return transform_box(previous.box, matrix)

        def geometry(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
            left_top = np.maximum(first[:2], second[:2])
            right_bottom = np.minimum(first[2:], second[2:])
            size = np.maximum(right_bottom - left_top, 0.0)
            intersection = float(size[0] * size[1])
            first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
            second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
            iou = intersection / max(
                first_area + second_area - intersection, 1e-6,
            )
            first_center = (first[:2] + first[2:]) / 2
            second_center = (second[:2] + second[2:]) / 2
            height = max(first[3] - first[1], second[3] - second[1], 25.0)
            center = float(np.linalg.norm(first_center - second_center) / height)
            area_ratio = max(first_area, second_area) / max(
                min(first_area, second_area), 1.0,
            )
            return iou, center, area_ratio

        for frame in frame_numbers:
            detections = frames[frame]
            rolling_density = (
                float(len(detections))
                if rolling_density is None
                else density_ema * rolling_density
                + (1.0 - density_ema) * len(detections)
            )
            warmup_limit = (
                initial_samples
                if rolling_density <= sparse_person_threshold
                else 1
            )
            active_before = {
                identity: detection
                for identity, detection in last_by_source.items()
                if 0 < frame - detection.frame <= max(
                    max_gap, ambiguity_max_gap,
                )
            }
            overlap_flags = [False] * len(detections)
            for first_index, first in enumerate(detections):
                for second_index in range(first_index + 1, len(detections)):
                    iou, _, _ = geometry(
                        first.box, detections[second_index].box,
                    )
                    if iou >= overlap_iou:
                        overlap_flags[first_index] = True
                        overlap_flags[second_index] = True

            frame_choices = []
            for index, detection in enumerate(detections):
                previous = last_by_source.get(detection.source_id)
                if previous is None:
                    reason = "new_identity"
                elif detection.frame - previous.frame > max_gap:
                    reason = "reappearance"
                elif observations_by_source[detection.source_id] < warmup_limit:
                    reason = "identity_warmup"
                elif overlap_flags[index]:
                    reason = "current_overlap"
                else:
                    own_prediction = predict(previous, frame)
                    own_iou, own_center, own_area_ratio = geometry(
                        own_prediction, detection.box,
                    )
                    own_score = (
                        0.65 * own_iou
                        + 0.35 * float(np.exp(-own_center))
                    )
                    best_alternative = 0.0
                    for identity, candidate in active_before.items():
                        if identity == detection.source_id:
                            continue
                        gap = frame - candidate.frame
                        if gap > ambiguity_max_gap:
                            continue
                        iou, center, _ = geometry(
                            predict(candidate, frame), detection.box,
                        )
                        if iou >= ambiguity_iou or center <= ambiguity_center:
                            alternative_score = (
                                0.65 * iou
                                + 0.35 * float(np.exp(-center))
                            )
                            best_alternative = max(
                                best_alternative, alternative_score,
                            )
                    if (
                        best_alternative >= ambiguity_min_score
                        and best_alternative
                        >= own_score - ambiguity_margin
                    ):
                        reason = "ambiguous_candidates"
                    else:
                        if (
                            own_iou < min_iou
                            or own_center > max_center
                            or own_area_ratio > max_area_ratio
                        ):
                            reason = "geometry_change"
                        else:
                            reason = "reuse"
                frame_choices.append(reason)
            for detection, reason in zip(detections, frame_choices):
                observation_index = len(selected)
                selected.append(reason != "reuse")
                reasons.append(reason)
                last_by_source[detection.source_id] = detection
                observations_by_source[detection.source_id] += 1
                indices_by_source.setdefault(detection.source_id, []).append(
                    observation_index
                )
        for indices in indices_by_source.values():
            for observation_index in indices[:head_samples]:
                if not selected[observation_index]:
                    selected[observation_index] = True
                    reasons[observation_index] = "identity_head"
            tail_indices = indices[-tail_samples:] if tail_samples else []
            for observation_index in tail_indices:
                if not selected[observation_index]:
                    selected[observation_index] = True
                    reasons[observation_index] = "identity_tail"
        return np.asarray(selected, dtype=bool), reasons

    def _crop(self, image: np.ndarray, det: Detection) -> np.ndarray:
        x1, y1, x2, y2 = det.box.round().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((32, 16, 3), np.uint8)
        elif self.mask_background:
            mask = (
                _decode_rle_crop(det, x1, y1, x2, y2)
                if self.crop_rle_decode
                else mask_utils.decode(det.rle)[y1:y2, x1:x2].astype(bool)
            )
            background = np.full_like(crop, 114)
            crop = np.where(mask[..., None], crop, background)
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def crop_from_mask(
        self, image: np.ndarray, det: Detection, mask: np.ndarray
    ) -> np.ndarray:
        """Create the exact ReID crop from an already materialised mask."""
        x1, y1, x2, y2 = det.box.round().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((32, 16, 3), np.uint8)
        elif self.mask_background:
            background = np.full_like(crop, 114)
            crop = np.where(mask[y1:y2, x1:x2, None], crop, background)
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    @torch.inference_mode()
    def encode_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.output_dim), np.float32)
        if self.yolo_reid is not None:
            if self.yolo_reid.is_pt:
                features = self.yolo_reid.model.predictor(crops)
                if len(features) != len(crops) and features[0].shape[0] == len(crops):
                    features = features[0]
                array = np.stack([
                    feature.detach().cpu().numpy() if torch.is_tensor(feature) else np.asarray(feature)
                    for feature in features
                ]).astype(np.float32)
                return torch.nn.functional.normalize(
                    torch.from_numpy(array).flatten(1), dim=1
                ).numpy()
            # AppearanceEncoder crops are RGB, while Ultralytics' ONNX helper
            # accepts BGR crops and converts them to RGB internally.  Convert
            # back here so the official preprocessing remains exact while the
            # existing mask-background crop is preserved.
            bgr_crops = [np.ascontiguousarray(crop[..., ::-1]) for crop in crops]
            batch = self.yolo_reid._crops_to_tensor(bgr_crops)
            backend = self.yolo_reid.model
            fixed_batch = self.yolo_reid.batch_size
            if fixed_batch is None or len(batch) == fixed_batch:
                features = backend(batch)
            else:
                chunks = []
                for start in range(0, len(batch), fixed_batch):
                    chunk = batch[start:start + fixed_batch]
                    valid = len(chunk)
                    if valid < fixed_batch:
                        chunk = torch.cat([
                            chunk,
                            chunk[-1:].expand(
                                fixed_batch - valid, *chunk.shape[1:]
                            ),
                        ], dim=0)
                    chunks.append(backend(chunk)[:valid])
                features = torch.cat(chunks, dim=0)
            if isinstance(features, (tuple, list)):
                features = features[0]
            return torch.nn.functional.normalize(
                features.float().flatten(1), dim=1
            ).cpu().numpy()
        tensor = (
            torch.from_numpy(np.stack(crops))
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        tensor = (tensor - self.mean) / self.std
        features = self.model(tensor.to(self.device)).flatten(1)
        return torch.nn.functional.normalize(
            features, dim=1
        ).cpu().numpy()

    def _flush(self, crops: list, owners: list, output: np.ndarray) -> None:
        if not crops:
            return
        output[np.asarray(owners)] = self.encode_crops(crops)
        crops.clear()
        owners.clear()


class StreamingAppearanceCollector:
    """Fuse causal ReID crop extraction into the detector's frame pass."""

    def __init__(
        self,
        encoder: AppearanceEncoder,
        event_config: dict,
        cache: Path | None,
        frame_offset: int = 0,
        max_buffer_frames: int = 0,
    ) -> None:
        self.encoder = encoder
        self.selector = CausalBudgetSelector(
            event_config, homographies=None, frame_offset=frame_offset
        )
        self.cache = cache
        self.detections: list[Detection] = []
        self.crops: list[np.ndarray] = []
        self.owners: list[int] = []
        self.fresh_features: dict[int, np.ndarray] = {}
        self.selector_seconds = 0.0
        self.crop_seconds = 0.0
        self.model_seconds = 0.0
        self.model_batches = 0
        self.output: np.ndarray | None = None
        self.max_buffer_frames = max(int(max_buffer_frames), 0)
        self.buffered_frames = 0
        self.max_buffered_crops = 0

    def _flush(self) -> None:
        if not self.crops:
            return
        batch_count = min(len(self.crops), self.encoder.batch_size)
        batch_crops = self.crops[:batch_count]
        batch_owners = self.owners[:batch_count]
        start = time.perf_counter()
        features = self.encoder.encode_crops(batch_crops)
        self.model_seconds += time.perf_counter() - start
        self.model_batches += 1
        for owner, feature in zip(batch_owners, features):
            self.fresh_features[owner] = feature
        del self.crops[:batch_count]
        del self.owners[:batch_count]

    def add_frame(
        self,
        frame: int,
        image: np.ndarray,
        detections: list[Detection],
        masks: list[np.ndarray],
    ) -> None:
        if len(detections) != len(masks):
            raise ValueError("Streaming ReID detections and masks differ in length")
        selector_start = time.perf_counter()
        selected, _reasons = self.selector.select_frame(frame, detections)
        self.selector_seconds += time.perf_counter() - selector_start
        offset = len(self.detections)
        self.detections.extend(detections)
        for local_index, (detection, mask, fresh) in enumerate(
            zip(detections, masks, selected)
        ):
            if not fresh:
                continue
            crop_start = time.perf_counter()
            self.crops.append(
                self.encoder.crop_from_mask(image, detection, mask)
            )
            self.crop_seconds += time.perf_counter() - crop_start
            self.owners.append(offset + local_index)
        if self.crops:
            self.buffered_frames += 1
            self.max_buffered_crops = max(
                self.max_buffered_crops, len(self.crops)
            )
        while (
            self.max_buffer_frames > 0
            and len(self.crops) >= self.encoder.batch_size
        ):
            self._flush()
        if (
            self.crops
            and self.max_buffer_frames > 0
            and self.buffered_frames >= self.max_buffer_frames
        ):
            while self.crops:
                self._flush()
        if not self.crops:
            self.buffered_frames = 0

    def finalize(self) -> dict[str, int | float | bool | dict]:
        while self.crops:
            self._flush()
        assignment_start = time.perf_counter()
        output = np.empty(
            (len(self.detections), self.encoder.output_dim), np.float32
        )
        last_feature: dict[int, np.ndarray] = {}
        for index, detection in enumerate(self.detections):
            if self.selector.selected[index]:
                feature = self.fresh_features[index]
                last_feature[detection.source_id] = feature
            else:
                feature = last_feature.get(detection.source_id)
                if feature is None:
                    raise RuntimeError(
                        "Streaming ReID skipped an identity before its first encoding: "
                        f"{detection.source_id}"
                    )
            output[index] = feature
        self.output = output
        assignment_seconds = time.perf_counter() - assignment_start
        cache_start = time.perf_counter()
        if self.cache is not None:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache,
                features=output,
                encoded_instances=np.asarray(
                    [sum(self.selector.selected)], np.int32
                ),
                selected=np.asarray(self.selector.selected, dtype=np.uint8),
                reasons=np.asarray(self.selector.reasons, dtype="<U32"),
                normalized=np.asarray([1], dtype=np.uint8),
            )
        cache_write_seconds = time.perf_counter() - cache_start
        encoded = int(sum(self.selector.selected))
        stats: dict[str, int | float | bool | dict] = {
            "stream_reid": True,
            "stream_reid_total_instances": len(self.detections),
            "stream_reid_encoded_instances": encoded,
            "stream_reid_encoded_ratio": encoded / max(len(self.detections), 1),
            "stream_reid_selector_seconds": self.selector_seconds,
            "stream_reid_crop_seconds": self.crop_seconds,
            "stream_reid_model_seconds": self.model_seconds,
            "stream_reid_model_batches": self.model_batches,
            "stream_reid_max_buffer_frames": self.max_buffer_frames,
            "stream_reid_max_buffered_crops": self.max_buffered_crops,
            "stream_reid_assignment_seconds": assignment_seconds,
            "stream_reid_cache_write_seconds": cache_write_seconds,
            "stream_reid_selection_reasons": dict(Counter(self.selector.reasons)),
        }
        stats.update(self.selector.summary())
        return stats

    def attach_sequence(
        self,
        frames: Dict[int, List[Detection]],
        policy: str = "budget_aware",
    ) -> dict[str, int | float | bool | dict]:
        """Attach normalized streaming features without a disk round trip."""
        if self.output is None:
            raise RuntimeError("Streaming ReID must be finalized before attachment")
        ordered = [detection for frame in sorted(frames) for detection in frames[frame]]
        if len(ordered) != len(self.output):
            raise ValueError(
                f"Streaming ReID has {len(self.output)} features for "
                f"{len(ordered)} parsed detections"
            )
        start = time.perf_counter()
        for detection, feature, fresh, reason in zip(
            ordered,
            self.output,
            self.selector.selected,
            self.selector.reasons,
        ):
            detection.feature = feature
            detection.feature_is_fresh = bool(fresh)
            detection.feature_reason = str(reason)
        attach_seconds = time.perf_counter() - start
        encoded = int(sum(self.selector.selected))
        result: dict[str, int | float | bool | dict] = {
            "reid_policy": policy,
            "reid_total_instances": len(ordered),
            "reid_encoded_instances": encoded,
            "reid_skipped_instances": len(ordered) - encoded,
            "reid_encoded_ratio": encoded / max(len(ordered), 1),
            "reid_seconds": 0.0,
            "reid_assignment_seconds": attach_seconds,
            "reid_selection_reasons": dict(Counter(self.selector.reasons)),
            "reid_memory_handoff": True,
            "reid_cache_hit": False,
        }
        result.update(self.selector.summary())
        return result


class AsyncCudaStreamingAppearanceCollector(StreamingAppearanceCollector):
    """Run ordered per-frame ReID batches on a dedicated CUDA stream."""

    supports_async_cpu_postprocess = True

    def __init__(
        self,
        encoder: AppearanceEncoder,
        event_config: dict,
        cache: Path | None,
        frame_offset: int = 0,
    ) -> None:
        if encoder.device.type != "cuda":
            raise ValueError("async CUDA ReID requires a CUDA AppearanceEncoder")
        super().__init__(
            encoder,
            event_config,
            cache,
            frame_offset=frame_offset,
            max_buffer_frames=0,
        )
        self.cuda_stream = torch.cuda.Stream(device=encoder.device)
        self.cuda_stream.wait_stream(torch.cuda.current_stream(encoder.device))
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reid-cuda-stream"
        )
        self.futures: list[Future] = []
        self.gpu_intervals: list[tuple[float, float]] = []
        self.max_pending_batches = 0

    def _encode_async(
        self, owners: list[int], crops: list[np.ndarray]
    ) -> tuple[list[int], np.ndarray, float, float]:
        start = time.perf_counter()
        with torch.cuda.device(self.encoder.device):
            with torch.cuda.stream(self.cuda_stream):
                features = self.encoder.encode_crops(crops)
        end = time.perf_counter()
        return owners, features, start, end

    def add_frame(
        self,
        frame: int,
        image: np.ndarray,
        detections: list[Detection],
        masks: list[np.ndarray],
    ) -> None:
        if len(detections) != len(masks):
            raise ValueError("Streaming ReID detections and masks differ in length")
        selector_start = time.perf_counter()
        selected, _reasons = self.selector.select_frame(frame, detections)
        self.selector_seconds += time.perf_counter() - selector_start
        offset = len(self.detections)
        self.detections.extend(detections)
        owners: list[int] = []
        crops: list[np.ndarray] = []
        for local_index, (detection, mask, fresh) in enumerate(
            zip(detections, masks, selected)
        ):
            if not fresh:
                continue
            crop_start = time.perf_counter()
            crops.append(self.encoder.crop_from_mask(image, detection, mask))
            self.crop_seconds += time.perf_counter() - crop_start
            owners.append(offset + local_index)
        if crops:
            self.max_buffered_crops = max(self.max_buffered_crops, len(crops))
            self.futures.append(self.executor.submit(self._encode_async, owners, crops))
            self.max_pending_batches = max(
                self.max_pending_batches,
                sum(not future.done() for future in self.futures),
            )

    def finalize(self) -> dict[str, int | float | bool | dict]:
        wait_start = time.perf_counter()
        for future in self.futures:
            owners, features, start, end = future.result()
            self.gpu_intervals.append((start, end))
            self.model_seconds += end - start
            self.model_batches += 1
            for owner, feature in zip(owners, features):
                self.fresh_features[owner] = feature
        wait_seconds = time.perf_counter() - wait_start
        self.executor.shutdown(wait=True)
        stats = super().finalize()
        stats.update(
            {
                "async_cuda_reid": True,
                "async_cuda_reid_wait_seconds": wait_seconds,
                "async_cuda_reid_max_pending_batches": self.max_pending_batches,
            }
        )
        return stats
