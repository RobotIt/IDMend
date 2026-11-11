from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from .geometry import box_iou_matrix, transform_box
from .mots_io import Detection


@dataclass
class _ProxyTrack:
    proxy_id: int
    observations: List[Detection] = field(default_factory=list)
    last_fresh: int = -1

    @property
    def last(self) -> Detection:
        return self.observations[-1]


def select_graph_budget(
    frames: Dict[int, List[Detection]],
    config: dict,
    homographies: list[np.ndarray] | None,
) -> tuple[np.ndarray, list[str], dict[str, int | float]]:
    """Select ReID observations using a source-ID-independent causal graph.

    The proxy tracker intentionally reads no upstream identity. It first
    forms a cheap geometry graph, evaluates temporal masks only for ambiguous
    Top-K edges, and maintains its own short-term identities. The resulting
    ``reid_key`` is used only for appearance reuse; final tracking remains a
    separate evaluation stage.
    """
    start = time.perf_counter()
    max_gap = max(int(config.get("max_gap", 2)), 1)
    top_k = max(int(config.get("top_k", 3)), 1)
    initial_samples = max(int(config.get("initial_samples", 3)), 1)
    refresh_interval = max(int(config.get("refresh_interval", 8)), 1)
    overlap_iou = float(config.get("overlap_iou", 0.30))
    match_min_score = float(config.get("match_min_score", 0.38))
    ambiguity_min_score = float(config.get("ambiguity_min_score", 0.40))
    ambiguity_margin = float(config.get("ambiguity_margin", 0.10))
    maximum_center = float(config.get("maximum_center_distance", 0.85))
    maximum_area_ratio = float(config.get("maximum_area_ratio", 2.0))
    mask_weight = float(config.get("mask_weight", 0.15))
    mask_pre_margin = float(config.get("mask_pre_margin", 0.18))
    replace_source_id = bool(config.get("replace_source_id", True))
    budget_ratio = min(max(float(config.get("budget_ratio", 1.0)), 0.0), 1.0)
    frame_numbers = sorted(frames)
    frame_offset = frame_numbers[0]
    tracks: dict[int, _ProxyTrack] = {}
    next_proxy_id = 1
    selected: list[bool] = []
    reasons: list[str] = []
    candidate_pairs = retained_pairs = mask_pairs = 0
    ambiguous_components = 0
    budget_slots = budget_used = budget_suppressed = 0

    def prediction(track: _ProxyTrack, frame: int) -> np.ndarray:
        last = track.last
        predicted = last.box.copy()
        if homographies is not None:
            source_index = last.frame - frame_offset
            target_index = frame - frame_offset
            if (
                0 <= source_index < len(homographies)
                and 0 <= target_index < len(homographies)
            ):
                predicted = transform_box(
                    predicted,
                    homographies[target_index]
                    @ np.linalg.inv(homographies[source_index]),
                )
        if len(track.observations) >= 2:
            previous = track.observations[-2]
            gap = max(last.frame - previous.frame, 1)
            last_center = (last.box[:2] + last.box[2:]) / 2
            previous_center = (previous.box[:2] + previous.box[2:]) / 2
            velocity = (last_center - previous_center) / gap
            dt = frame - last.frame
            predicted[:2] += velocity * dt
            predicted[2:] += velocity * dt
        return predicted

    for frame in frame_numbers:
        detections = frames[frame]
        boxes = (
            np.stack([detection.box for detection in detections])
            if detections else np.empty((0, 4), np.float32)
        )
        active = [
            track for track in tracks.values()
            if 0 < frame - track.last.frame <= max_gap
        ]
        if detections:
            overlap_matrix = box_iou_matrix(boxes, boxes)
            np.fill_diagonal(overlap_matrix, 0.0)
            overlap_flags = np.any(overlap_matrix >= overlap_iou, axis=1)
        else:
            overlap_flags = np.empty(0, dtype=bool)

        if active and detections:
            predicted = np.stack([prediction(track, frame) for track in active])
            iou = box_iou_matrix(predicted, boxes)
            pc = (predicted[:, :2] + predicted[:, 2:]) / 2
            dc = (boxes[:, :2] + boxes[:, 2:]) / 2
            height = np.maximum(
                np.maximum(
                    (predicted[:, 3] - predicted[:, 1])[:, None],
                    (boxes[:, 3] - boxes[:, 1])[None, :],
                ),
                25.0,
            )
            center = np.linalg.norm(pc[:, None] - dc[None], axis=2) / height
            pa = np.maximum(predicted[:, 2] - predicted[:, 0], 0) * np.maximum(
                predicted[:, 3] - predicted[:, 1], 0
            )
            da = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
                boxes[:, 3] - boxes[:, 1], 0
            )
            area_ratio = np.maximum(pa[:, None], da[None]) / np.maximum(
                np.minimum(pa[:, None], da[None]), 1.0
            )
            score = 0.62 * iou + 0.38 * np.exp(-center)
            valid = (center <= maximum_center) & (
                area_ratio <= maximum_area_ratio
            )
            candidate_pairs += int(score.size)
            sparse_valid = np.zeros_like(valid)
            for row in range(len(active)):
                order = np.argsort(-score[row], kind="stable")[:top_k]
                sparse_valid[row, order] = valid[row, order]
            retained_pairs += int(sparse_valid.sum())

            ambiguous_detections = np.zeros(len(detections), dtype=bool)
            ambiguous_tracks = np.zeros(len(active), dtype=bool)
            for column in range(len(detections)):
                rows = np.flatnonzero(sparse_valid[:, column])
                if len(rows) < 2:
                    continue
                ordered = rows[np.argsort(-score[rows, column], kind="stable")]
                first, second = ordered[:2]
                if (
                    score[first, column] >= ambiguity_min_score
                    and score[first, column] - score[second, column]
                    <= ambiguity_margin + mask_pre_margin
                ):
                    inspect = ordered[:top_k]
                    temporal_masks = mask_utils.iou(
                        [active[row].last.rle for row in inspect],
                        [detections[column].rle],
                        [0],
                    )[:, 0]
                    mask_pairs += len(inspect)
                    refined = (
                        (1.0 - mask_weight) * score[inspect, column]
                        + mask_weight * temporal_masks
                    )
                    refined_order = np.argsort(-refined, kind="stable")
                    if (
                        len(refined_order) >= 2
                        and refined[refined_order[0]]
                        - refined[refined_order[1]] <= ambiguity_margin
                    ):
                        ambiguous_detections[column] = True
                        ambiguous_tracks[inspect[refined_order[:2]]] = True
                        ambiguous_components += 1
            for row in range(len(active)):
                columns = np.flatnonzero(sparse_valid[row])
                if len(columns) < 2:
                    continue
                ordered = columns[
                    np.argsort(-score[row, columns], kind="stable")
                ]
                if (
                    score[row, ordered[0]] >= ambiguity_min_score
                    and score[row, ordered[0]] - score[row, ordered[1]]
                    <= ambiguity_margin
                ):
                    ambiguous_tracks[row] = True
                    ambiguous_detections[ordered[:2]] = True
                    ambiguous_components += 1

            cost = 1.0 - score
            cost[~sparse_valid] = 1e5
            rows, columns = linear_sum_assignment(cost)
            matches = [
                (int(row), int(column))
                for row, column in zip(rows, columns)
                if sparse_valid[row, column]
                and score[row, column] >= match_min_score
            ]
        else:
            iou = center = area_ratio = np.empty(
                (len(active), len(detections)), np.float32
            )
            ambiguous_detections = np.zeros(len(detections), dtype=bool)
            ambiguous_tracks = np.zeros(len(active), dtype=bool)
            matches = []

        matched_detections = set()
        frame_decisions: dict[
            int, tuple[bool, str, int, _ProxyTrack, int, float, bool]
        ] = {}
        for row, column in matches:
            track = active[row]
            detection = detections[column]
            observation_number = len(track.observations)
            if observation_number < initial_samples:
                reason = "proxy_warmup"
            elif overlap_flags[column]:
                reason = "current_overlap"
            elif ambiguous_detections[column] or ambiguous_tracks[row]:
                reason = "ambiguous_graph"
            elif (
                iou[row, column] < 0.20
                or center[row, column] > maximum_center * 0.75
                or area_ratio[row, column] > maximum_area_ratio * 0.80
            ):
                reason = "geometry_change"
            elif observation_number - track.last_fresh >= refresh_interval:
                reason = "periodic_refresh"
            else:
                reason = "reuse"
            fresh = reason != "reuse"
            priority = {
                "ambiguous_graph": 1.00,
                "current_overlap": 0.90,
                "geometry_change": 0.80,
                "proxy_warmup": 0.65,
                "periodic_refresh": 0.50,
                "reuse": 0.0,
            }[reason]
            detection.reid_key = track.proxy_id
            if replace_source_id:
                detection.source_id = track.proxy_id
            track.observations.append(detection)
            frame_decisions[column] = (
                fresh, reason, track.proxy_id, track,
                observation_number, priority, False,
            )
            matched_detections.add(column)

        for column, detection in enumerate(detections):
            if column in matched_detections:
                continue
            track = _ProxyTrack(next_proxy_id)
            next_proxy_id += 1
            detection.reid_key = track.proxy_id
            if replace_source_id:
                detection.source_id = track.proxy_id
            track.observations.append(detection)
            track.last_fresh = 0
            tracks[track.proxy_id] = track
            frame_decisions[column] = (
                True, "new_proxy", track.proxy_id, track,
                0, float("inf"), True,
            )

        frame_limit = (
            len(detections)
            if budget_ratio >= 1.0
            else int(math.ceil(budget_ratio * len(detections)))
        )
        mandatory_columns = [
            column for column, item in frame_decisions.items()
            if item[6]
        ]
        optional_columns = [
            column for column, item in frame_decisions.items()
            if item[0] and not item[6]
        ]
        optional_columns.sort(
            key=lambda column: (
                -frame_decisions[column][5], column
            )
        )
        optional_budget = max(frame_limit - len(mandatory_columns), 0)
        allowed_optional = set(optional_columns[:optional_budget])
        budget_slots += frame_limit
        budget_used += len(mandatory_columns) + len(allowed_optional)
        budget_suppressed += max(
            len(optional_columns) - len(allowed_optional), 0
        )

        for column in range(len(detections)):
            (
                fresh, reason, _proxy_id, track,
                observation_number, _priority, mandatory,
            ) = frame_decisions[column]
            if fresh and not mandatory and column not in allowed_optional:
                fresh = False
                reason = "budget_reuse"
            if fresh:
                track.last_fresh = observation_number
            selected.append(fresh)
            reasons.append(reason)

    total = len(selected)
    return (
        np.asarray(selected, dtype=bool),
        reasons,
        {
            "reid_selector_seconds": time.perf_counter() - start,
            "reid_candidate_pairs": candidate_pairs,
            "reid_retained_pairs": retained_pairs,
            "reid_mask_pairs": mask_pairs,
            "reid_candidate_keep_ratio": retained_pairs
            / max(candidate_pairs, 1),
            "reid_ambiguous_components": ambiguous_components,
            "reid_proxy_tracks": len(tracks),
            "reid_source_id_independent": True,
            "reid_source_id_replaced": replace_source_id,
            "reid_graph_selected_ratio": sum(selected) / max(total, 1),
            "reid_budget_ratio": budget_ratio,
            "reid_budget_slots": budget_slots,
            "reid_budget_used": budget_used,
            "reid_budget_suppressed": budget_suppressed,
        },
    )
