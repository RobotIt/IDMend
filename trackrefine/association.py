from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import time

from .geometry import box_iou_matrix, transform_box
from .mots_io import Detection, mask_iou
from .link_model import LinkMLP
from .link_rejector import LinkRejector
from .local_ranker import EnsembleLocalRanker, LocalRanker
from .part_reid import tracklet_part_features
from .association_score import PartAwareAssociationScore


@dataclass
class Track:
    track_id: int
    observations: List[Detection] = field(default_factory=list)
    feature: np.ndarray | None = None
    gallery: List[np.ndarray] = field(default_factory=list)
    gallery_scores: List[float] = field(default_factory=list)
    # V45 keeps the existing bounded gallery, but replaces its single
    # fresh-mean readout with a fast short-term representation and a slowly
    # updated, quality-gated stable representation.  The latter is never
    # updated from a crowded or internally inconsistent observation.
    short_feature: np.ndarray | None = None
    stable_feature: np.ndarray | None = None
    stable_feature_writes: int = 0
    stable_feature_rejections: int = 0
    # V47 keeps several reliable and mutually diverse views as a sidecar.
    # The baseline fresh-mean feature above remains the only representation
    # used by ordinary association and by the pretrained LinkMLP.
    prototype_bank: List[np.ndarray] = field(default_factory=list)
    prototype_bank_scores: List[float] = field(default_factory=list)
    prototype_bank_writes: int = 0
    prototype_bank_replacements: int = 0
    prototype_bank_rejections: int = 0
    selective_update_skips: int = 0

    def update(
        self,
        detection: Detection,
        ema: float,
        feature_config: dict | None = None,
    ) -> None:
        detection.assigned_id = self.track_id
        self.observations.append(detection)
        value = detection.feature
        if value is not None:
            feature_config = feature_config or {}
            fresh_only = bool(feature_config.get("fresh_only", False))
            if fresh_only and not detection.feature_is_fresh:
                return
            # V48-1 (CATrack SU adaptation): when a feature was requested
            # specifically for an overlap/ambiguity event, keep the detection
            # in its source tracklet but do not let that potentially mixed crop
            # overwrite the existing ordinary V42-B1 appearance gallery.  A
            # new track still receives its first feature so later stages never
            # encounter an uninitialized representation.
            selective_cfg = feature_config.get("selective_update", {})
            if (
                bool(selective_cfg.get("enabled", False))
                and self.gallery
                and detection.feature_reason
                in set(
                    selective_cfg.get(
                        "reject_reasons",
                        ["current_overlap", "ambiguous_sparse_graph"],
                    )
                )
            ):
                self.selective_update_skips += 1
                return
            self.gallery.append(value.copy())
            self.gallery_scores.append(float(detection.score))
            gallery_size = int(feature_config.get("gallery_size", 30))
            if len(self.gallery) > gallery_size:
                self.gallery.pop(0)
                self.gallery_scores.pop(0)
            prototype_mode = feature_config.get("prototype", "ema")
            if prototype_mode == "last_fresh":
                self.feature = value.copy()
            elif prototype_mode == "dual_timescale_quality":
                short_ema = float(feature_config.get("short_ema", ema))
                self.short_feature = (
                    value.copy()
                    if self.short_feature is None
                    else short_ema * self.short_feature
                    + (1.0 - short_ema) * value
                )
                self.short_feature /= (
                    np.linalg.norm(self.short_feature) + 1e-12
                )

                # Long-term memory accepts only observations that are both
                # visually compatible with its existing representation and
                # structurally reliable.  In particular, fresh ReID crops
                # requested for current overlap are useful for disambiguation
                # but are unsafe to write back into a persistent identity.
                stable_min_length = int(
                    feature_config.get("stable_min_track_length", 3)
                )
                stable_min_score = float(
                    feature_config.get("stable_min_score", 0.60)
                )
                stable_min_similarity = float(
                    feature_config.get("stable_min_similarity", 0.70)
                )
                reject_reasons = set(
                    feature_config.get(
                        "stable_reject_reasons",
                        ["current_overlap", "ambiguous_sparse_graph"],
                    )
                )
                eligible = (
                    len(self.observations) >= stable_min_length
                    and float(detection.score) >= stable_min_score
                    and detection.feature_reason not in reject_reasons
                )
                if eligible and self.stable_feature is not None:
                    eligible = bool(
                        float(self.stable_feature @ value)
                        >= stable_min_similarity
                    )
                if eligible:
                    stable_ema = float(
                        feature_config.get("stable_ema", 0.95)
                    )
                    self.stable_feature = (
                        value.copy()
                        if self.stable_feature is None
                        else stable_ema * self.stable_feature
                        + (1.0 - stable_ema) * value
                    )
                    self.stable_feature /= (
                        np.linalg.norm(self.stable_feature) + 1e-12
                    )
                    self.stable_feature_writes += 1
                else:
                    self.stable_feature_rejections += 1

                stable_blend = float(
                    feature_config.get("stable_blend", 0.50)
                )
                if self.stable_feature is None:
                    self.feature = self.short_feature.copy()
                else:
                    self.feature = (
                        (1.0 - stable_blend) * self.short_feature
                        + stable_blend * self.stable_feature
                    )
            elif prototype_mode in (
                "fresh_mean",
                "stable_read_gated",
                "quality_diverse_bank",
            ):
                sample_count = int(
                    feature_config.get("prototype_samples", gallery_size)
                )
                samples = np.stack(self.gallery[-sample_count:])
                weights = np.asarray(
                    self.gallery_scores[-sample_count:], np.float32
                )
                weights = np.maximum(weights, 0.05)
                prototype = np.average(samples, axis=0, weights=weights)
                prototype /= np.linalg.norm(prototype) + 1e-12
                minimum_similarity = float(
                    feature_config.get("prototype_outlier_similarity", 0.35)
                )
                similarity = samples @ prototype
                keep = similarity >= minimum_similarity
                if int(keep.sum()) >= 2:
                    kept_weights = weights[keep]
                    prototype = np.average(
                        samples[keep], axis=0, weights=kept_weights
                    )
                self.feature = prototype
                # V46 deliberately leaves the ordinary fresh-mean feature
                # untouched.  It writes a separately quality-gated stable
                # prototype which is read only by the pre-registered global
                # link rejection gate below; it is never blended into normal
                # adjacent/lost association or the LinkMLP input.
                if prototype_mode == "stable_read_gated":
                    stable_min_length = int(
                        feature_config.get("stable_min_track_length", 3)
                    )
                    stable_min_score = float(
                        feature_config.get("stable_min_score", 0.60)
                    )
                    stable_min_similarity = float(
                        feature_config.get("stable_min_similarity", 0.70)
                    )
                    reject_reasons = set(
                        feature_config.get(
                            "stable_reject_reasons",
                            ["current_overlap", "ambiguous_sparse_graph"],
                        )
                    )
                    eligible = (
                        len(self.observations) >= stable_min_length
                        and float(detection.score) >= stable_min_score
                        and detection.feature_reason not in reject_reasons
                    )
                    if eligible and self.stable_feature is not None:
                        eligible = bool(
                            float(self.stable_feature @ value)
                            >= stable_min_similarity
                        )
                    if eligible:
                        stable_ema = float(feature_config.get("stable_ema", 0.95))
                        self.stable_feature = (
                            value.copy()
                            if self.stable_feature is None
                            else stable_ema * self.stable_feature
                            + (1.0 - stable_ema) * value
                        )
                        self.stable_feature /= (
                            np.linalg.norm(self.stable_feature) + 1e-12
                        )
                        self.stable_feature_writes += 1
                    else:
                        self.stable_feature_rejections += 1
                elif prototype_mode == "quality_diverse_bank":
                    # Write only confirmed, clear and motion-consistent views.
                    minimum_length = int(
                        feature_config.get("bank_min_track_length", 3)
                    )
                    minimum_score = float(
                        feature_config.get("bank_min_score", 0.60)
                    )
                    reject_reasons = set(
                        feature_config.get(
                            "bank_reject_reasons",
                            ["current_overlap", "ambiguous_sparse_graph"],
                        )
                    )
                    motion_residual = 0.0
                    if len(self.observations) >= 3:
                        previous = self.observations[-2]
                        before = self.observations[-3]
                        previous_center = (
                            previous.box[:2] + previous.box[2:]
                        ) / 2
                        before_center = (before.box[:2] + before.box[2:]) / 2
                        current_center = (
                            detection.box[:2] + detection.box[2:]
                        ) / 2
                        previous_gap = max(previous.frame - before.frame, 1)
                        current_gap = max(detection.frame - previous.frame, 1)
                        velocity = (
                            previous_center - before_center
                        ) / previous_gap
                        predicted_center = previous_center + velocity * current_gap
                        height = max(
                            detection.box[3] - detection.box[1],
                            previous.box[3] - previous.box[1],
                            25,
                        )
                        motion_residual = float(
                            np.linalg.norm(current_center - predicted_center)
                            / height
                        )
                    eligible = (
                        len(self.observations) >= minimum_length
                        and float(detection.score) >= minimum_score
                        and detection.feature_reason not in reject_reasons
                        and motion_residual
                        <= float(feature_config.get("bank_max_motion_residual", 1.25))
                    )
                    minimum_consistency = float(
                        feature_config.get("bank_min_consistency", 0.60)
                    )
                    redundancy_similarity = float(
                        feature_config.get("bank_redundancy_similarity", 0.90)
                    )
                    if eligible and self.prototype_bank:
                        bank = np.stack(self.prototype_bank)
                        similarities = bank @ value
                        maximum_similarity = float(np.max(similarities))
                        eligible = (
                            maximum_similarity >= minimum_consistency
                            and maximum_similarity < redundancy_similarity
                        )
                    if not eligible:
                        self.prototype_bank_rejections += 1
                    else:
                        capacity = int(feature_config.get("bank_capacity", 5))
                        candidate_score = float(detection.score)
                        if len(self.prototype_bank) < capacity:
                            self.prototype_bank.append(value.copy())
                            self.prototype_bank_scores.append(candidate_score)
                            self.prototype_bank_writes += 1
                        else:
                            # Keep the most diverse reliable subset: find the
                            # most redundant pair among old prototypes plus the
                            # candidate, then discard its lower-quality member.
                            vectors = self.prototype_bank + [value.copy()]
                            scores = self.prototype_bank_scores + [candidate_score]
                            matrix = np.stack(vectors) @ np.stack(vectors).T
                            np.fill_diagonal(matrix, -np.inf)
                            left, right = np.unravel_index(
                                int(np.argmax(matrix)), matrix.shape
                            )
                            remove = left if scores[left] <= scores[right] else right
                            if remove == len(vectors) - 1:
                                self.prototype_bank_rejections += 1
                            else:
                                vectors.pop(remove)
                                scores.pop(remove)
                                self.prototype_bank = vectors
                                self.prototype_bank_scores = scores
                                self.prototype_bank_writes += 1
                                self.prototype_bank_replacements += 1
            elif prototype_mode == "adaptive_ema":
                # V48-2 (CATrack AEMA adaptation): a confident observation
                # may update the primary representation faster, while a low
                # confidence observation mostly preserves its history. The
                # gallery remains present for auditability but is not read by
                # this deliberately independent prototype replacement.
                lower_score = float(feature_config.get("aema_min_score", 0.35))
                clipped_score = float(np.clip(detection.score, lower_score, 1.0))
                confidence = (clipped_score - lower_score) / max(
                    1.0 - lower_score, 1e-6
                )
                minimum_ema = float(feature_config.get("aema_min_ema", 0.70))
                maximum_ema = float(feature_config.get("aema_max_ema", 0.95))
                adaptive_ema = maximum_ema - confidence * (
                    maximum_ema - minimum_ema
                )
                self.feature = (
                    value.copy()
                    if self.feature is None
                    else adaptive_ema * self.feature
                    + (1.0 - adaptive_ema) * value
                )
            else:
                self.feature = (
                    value.copy()
                    if self.feature is None
                    else ema * self.feature + (1 - ema) * value
                )
            self.feature /= np.linalg.norm(self.feature) + 1e-12

    @property
    def last(self) -> Detection:
        return self.observations[-1]


class GlobalTracker:
    def __init__(self, homographies: List[np.ndarray], config: dict):
        self.h = homographies
        self.cfg = config
        self.feature_cfg = config.get("feature_management", {})
        self.split_cfg = config.get("ambiguity_split", {})
        self.frame_detections: Dict[int, List[Detection]] = {}
        self.tracks: Dict[int, Track] = {}
        # KITTI-MOTS encodes pedestrians as 2000 + instance_id, with
        # instance_id 0 reserved. Start at the first valid pedestrian ID.
        self.next_id = 2001
        self.frame_offset = 0
        self.stats = {
            "association_mode": "uninitialized",
            "ambiguity_splits": 0,
            "global_link_candidates": 0,
            "global_links": 0,
            "tracklet_build_seconds": 0.0,
            "global_link_seconds": 0.0,
            "link_rejector_enabled": bool(
                config.get("link_rejector", {}).get("enabled", False)
            ),
            "link_rejector_reads": 0,
            "link_rejector_kept": 0,
            "link_rejector_rejected": 0,
            "global_pair_capacity": 0,
            "global_temporal_pairs": 0,
            "global_motion_pairs": 0,
            "global_scored_pairs": 0,
            # The inactive-memory route is deliberately separate from the
            # existing short-gap global linker.  These counters make it clear
            # whether a run actually used the new route.
            "inactive_memory_temporal_pairs": 0,
            "inactive_memory_candidates": 0,
            "inactive_memory_links": 0,
            "quality_memory_enabled": bool(
                self.feature_cfg.get("prototype")
                in ("dual_timescale_quality", "stable_read_gated")
            ),
            "quality_memory_stable_writes": 0,
            "quality_memory_stable_rejections": 0,
            "quality_memory_tracks_with_stable_feature": 0,
            "quality_memory_global_reads": 0,
            "quality_memory_global_passes": 0,
            "quality_memory_global_rejections": 0,
            "quality_memory_global_unavailable": 0,
            "diverse_memory_enabled": bool(
                self.feature_cfg.get("prototype") == "quality_diverse_bank"
            ),
            "diverse_memory_writes": 0,
            "diverse_memory_replacements": 0,
            "diverse_memory_rejections": 0,
            "diverse_memory_tracks": 0,
            "diverse_memory_prototypes": 0,
            "diverse_memory_global_reads": 0,
            "diverse_memory_global_boosts": 0,
            "diverse_memory_new_candidates": 0,
            "selective_update_enabled": bool(
                self.feature_cfg.get("selective_update", {}).get("enabled", False)
            ),
            "selective_update_skips": 0,
            "state_template_enabled": bool(
                self.feature_cfg.get("state_aware_template", {}).get("enabled", False)
            ),
            "state_template_global_reads": 0,
            "state_template_global_boosts": 0,
            "state_template_new_candidates": 0,
            "state_template_unavailable": 0,
            # V54: a geometry-only object-permanence cue.  It does not write
            # an appearance memory or alter a normal accepted link.  These
            # counters show whether the cue reached an ambiguous LinkMLP
            # candidate and whether it actually created a new candidate.
            "observability_fusion_enabled": bool(
                config.get("observability_fusion", {}).get("enabled", False)
            ),
            "observability_fusion_reads": 0,
            "observability_fusion_survivors": 0,
            "observability_fusion_boosts": 0,
            "observability_fusion_new_candidates": 0,
            # V57: resolve only local, competing global-link candidates as
            # independent maximum-weight bipartite subproblems.  The module
            # never changes candidate generation or individual edge scores.
            "confusion_decomposition_enabled": bool(
                config.get("confusion_decomposition", {}).get("enabled", False)
            ),
            "confusion_decomposition_components": 0,
            "confusion_decomposition_ambiguous_components": 0,
            "confusion_decomposition_candidate_edges": 0,
            "confusion_decomposition_changed_links": 0,
            "local_ranker_enabled": bool(
                config.get("local_ranker", {}).get("enabled", False)
            ),
            "local_ranker_ambiguous_edges": 0,
            "dual_head_link_enabled": False,
            "dual_head_ranked_edges": 0,
        }

    def run(self, frames: Dict[int, List[Detection]]) -> List[Track]:
        self.stats["association_mode"] = "online_global_tracker"
        self.frame_detections = frames
        frame_numbers = sorted(frames)
        self.frame_offset = frame_numbers[0]
        for frame in frame_numbers:
            detections = frames[frame]
            active = [
                track for track in self.tracks.values()
                if 0 < frame - track.last.frame <= self.cfg["max_age"]
            ]
            adjacent = [track for track in active if frame - track.last.frame == 1]
            lost = [track for track in active if frame - track.last.frame > 1]
            high_threshold = float(self.cfg.get("high_confidence", .35))
            new_threshold = float(
                self.cfg.get("new_track_confidence", high_threshold)
            )
            unmatched_high = {
                i for i, detection in enumerate(detections)
                if detection.score >= high_threshold
            }
            unmatched_low = set(range(len(detections))) - unmatched_high
            used_tracks = set()
            for pool, stage in ((adjacent, "adjacent"), (lost, "lost")):
                pool = [track for track in pool if track.track_id not in used_tracks]
                candidates = sorted(unmatched_high)
                matches = self._match(pool, [detections[i] for i in candidates], frame, stage)
                for track_index, local_det_index in matches:
                    track, det_index = pool[track_index], candidates[local_det_index]
                    track.update(
                        detections[det_index],
                        self.cfg["ema"],
                        self.feature_cfg,
                    )
                    used_tracks.add(track.track_id)
                    unmatched_high.discard(det_index)

            # Low-confidence detections may recover an adjacent trajectory,
            # but never create a new identity on their own.
            remaining_adjacent = [
                track for track in adjacent if track.track_id not in used_tracks
            ]
            low_candidates = sorted(unmatched_low)
            matches = self._match(
                remaining_adjacent,
                [detections[i] for i in low_candidates],
                frame,
                "low",
            )
            for track_index, local_det_index in matches:
                track = remaining_adjacent[track_index]
                det_index = low_candidates[local_det_index]
                track.update(
                    detections[det_index],
                    self.cfg["ema"],
                    self.feature_cfg,
                )
                used_tracks.add(track.track_id)
                unmatched_low.discard(det_index)

            for index in sorted(unmatched_high):
                if detections[index].score < new_threshold:
                    continue
                track = Track(self.next_id)
                self.next_id += 1
                track.update(
                    detections[index],
                    self.cfg["ema"],
                    self.feature_cfg,
                )
                self.tracks[track.track_id] = track
        tracks = list(self.tracks.values())
        self._record_quality_memory_stats(tracks)
        self._record_diverse_memory_stats(tracks)
        self._record_selective_update_stats(tracks)
        self._global_link(tracks)
        return tracks

    def refine_tracklets(self, frames: Dict[int, List[Detection]]) -> List[Track]:
        """Refine upstream short-term tracklets without changing their frames.

        This mode is intended for cached YOLO-MOTS results: the upstream IDs
        provide reliable adjacent-frame associations, while the independent
        global graph links fragments across occlusion. No source ID is copied
        blindly to the final namespace.
        """
        tracks = self.prepare_refine_tracklets(frames)
        return self.link_prepared_tracklets(tracks)

    def prepare_refine_tracklets(self, frames: Dict[int, List[Detection]]) -> List[Track]:
        """Build and namespace upstream fragments before global linking."""
        self.stats["association_mode"] = "upstream_tracklets_global_link"
        self.frame_detections = frames
        frame_numbers = sorted(frames)
        self.frame_offset = frame_numbers[0]
        build_start = time.perf_counter()
        tracks = self.build_tracklets(frames)
        # Upstream IDs are made sequence-local and collision-safe.
        for index, track in enumerate(
            sorted(tracks, key=lambda item: item.observations[0].frame), start=2001
        ):
            track.track_id = index
            for observation in track.observations:
                observation.assigned_id = index
        self.stats["tracklet_build_seconds"] = (
            time.perf_counter() - build_start
        )
        return tracks

    def link_prepared_tracklets(self, tracks: List[Track]) -> List[Track]:
        """Apply frozen global candidate gate, final score and solver."""
        self._record_quality_memory_stats(tracks)
        self._record_diverse_memory_stats(tracks)
        self._record_selective_update_stats(tracks)
        if bool(self.cfg.get("global_link_enabled", True)):
            self._global_link(tracks)
        else:
            self.stats["global_link_disabled"] = True
        return tracks

    def part_feature_observations(self, tracks: List[Track]) -> list[Detection]:
        """Boundary observations used by ambiguous admitted LinkMLP edges.

        V72-B calls this after short-track construction and before final
        scoring.  It mirrors the frozen V70 temporal, motion and gate tests;
        it does not score or alter any edge.
        """
        model_path = self.cfg.get("link_model")
        if not model_path or not Path(model_path).exists():
            return []
        linker = LinkMLP(Path(model_path))
        threshold = float(self.cfg.get("link_threshold", linker.threshold))
        maximum_gap = int(self.cfg["global_max_gap"])
        admitted = []
        for first in tracks:
            for second in tracks:
                gap = second.observations[0].frame - first.last.frame
                if gap <= 0 or gap > maximum_gap:
                    continue
                predicted = self._prediction(first, second.observations[0].frame)
                target = second.observations[0].box
                predicted_center = (predicted[:2] + predicted[2:]) / 2
                target_center = (target[:2] + target[2:]) / 2
                scale = max(predicted[3] - predicted[1], target[3] - target[1], 25)
                distance = float(np.linalg.norm(predicted_center - target_center) / scale)
                if (
                    bool(self.cfg.get("global_motion_gate_enabled", True))
                    and distance > 2.5 + .06 * gap
                ):
                    continue
                if linker.score(linker.features(first, second)) >= threshold:
                    admitted.append((first, second))
        source_degree, target_degree = {}, {}
        for first, second in admitted:
            source_degree[first.track_id] = source_degree.get(first.track_id, 0) + 1
            target_degree[second.track_id] = target_degree.get(second.track_id, 0) + 1
        requested = {}
        for first, second in admitted:
            if source_degree[first.track_id] <= 1 and target_degree[second.track_id] <= 1:
                continue
            for observation in first.observations[-min(8, len(first.observations)):]:
                requested[id(observation)] = observation
            for observation in second.observations[:min(8, len(second.observations))]:
                requested[id(observation)] = observation
        self.stats["part_feature_requested_observations"] = len(requested)
        self.stats["part_feature_ambiguous_candidate_edges"] = sum(
            source_degree[first.track_id] > 1 or target_degree[second.track_id] > 1
            for first, second in admitted
        )
        return list(requested.values())

    def build_tracklets(
        self, frames: Dict[int, List[Detection]]
    ) -> List[Track]:
        """Build freshness-aware upstream segments without global linking."""
        frame_numbers = sorted(frames)
        self.frame_offset = frame_numbers[0]
        grouped: List[Track] = []
        current_by_source: Dict[int, Track] = {}
        internal_id = 1
        for frame in frame_numbers:
            for detection in frames[frame]:
                track = current_by_source.get(detection.source_id)
                if track is not None and self._should_split(
                    track, detection
                ):
                    self.stats["ambiguity_splits"] += 1
                    track = None
                if track is None:
                    track = Track(internal_id)
                    internal_id += 1
                    grouped.append(track)
                    current_by_source[detection.source_id] = track
                track.update(
                    detection, self.cfg["ema"], self.feature_cfg
                )
        tracks = grouped
        return tracks

    def _should_split(self, track: Track, detection: Detection) -> bool:
        """Split an upstream ID only at a fresh, identity-ambiguous event."""
        if not bool(self.split_cfg.get("enabled", False)):
            return False
        if (
            not detection.feature_is_fresh
            or detection.feature is None
            or track.feature is None
        ):
            return False
        if len(track.observations) < int(
            self.split_cfg.get("minimum_track_length", 6)
        ):
            return False
        allowed_reasons = set(
            self.split_cfg.get(
                "reasons",
                [
                    "ambiguous_candidates",
                    "current_overlap",
                    "reappearance",
                ],
            )
        )
        if detection.feature_reason not in allowed_reasons:
            return False
        samples = (
            np.stack(track.gallery[-12:])
            if track.gallery
            else track.feature[None]
        )
        similarity = float(np.max(samples @ detection.feature))
        return similarity < float(
            self.split_cfg.get("similarity_threshold", 0.42)
        )

    def _record_quality_memory_stats(self, tracks: List[Track]) -> None:
        """Expose V45 memory writes so quality gating is auditable."""
        if not self.stats["quality_memory_enabled"]:
            return
        self.stats["quality_memory_stable_writes"] = int(
            sum(track.stable_feature_writes for track in tracks)
        )
        self.stats["quality_memory_stable_rejections"] = int(
            sum(track.stable_feature_rejections for track in tracks)
        )
        self.stats["quality_memory_tracks_with_stable_feature"] = int(
            sum(track.stable_feature is not None for track in tracks)
        )

    def _record_diverse_memory_stats(self, tracks: List[Track]) -> None:
        """Expose V47 multi-prototype writes and capacity use."""
        if not self.stats["diverse_memory_enabled"]:
            return
        self.stats["diverse_memory_writes"] = int(
            sum(track.prototype_bank_writes for track in tracks)
        )
        self.stats["diverse_memory_replacements"] = int(
            sum(track.prototype_bank_replacements for track in tracks)
        )
        self.stats["diverse_memory_rejections"] = int(
            sum(track.prototype_bank_rejections for track in tracks)
        )
        self.stats["diverse_memory_tracks"] = int(
            sum(bool(track.prototype_bank) for track in tracks)
        )
        self.stats["diverse_memory_prototypes"] = int(
            sum(len(track.prototype_bank) for track in tracks)
        )

    def _record_selective_update_stats(self, tracks: List[Track]) -> None:
        if not self.stats["selective_update_enabled"]:
            return
        self.stats["selective_update_skips"] = int(
            sum(track.selective_update_skips for track in tracks)
        )

    @staticmethod
    def _diverse_memory_similarity(first: Track, second: Track, top_k: int) -> float:
        """Symmetric top-k nearest-prototype similarity for V47."""
        first_bank = np.stack(first.prototype_bank)
        second_bank = np.stack(second.prototype_bank)
        cross = first_bank @ second_bank.T
        forward = np.max(cross, axis=1)
        backward = np.max(cross, axis=0)
        forward_k = min(max(top_k, 1), len(forward))
        backward_k = min(max(top_k, 1), len(backward))
        forward_score = float(np.mean(np.sort(forward)[-forward_k:]))
        backward_score = float(np.mean(np.sort(backward)[-backward_k:]))
        return 0.5 * (forward_score + backward_score)

    @staticmethod
    def _state_template(track: Track, from_end: bool, config: dict) -> np.ndarray | None:
        """Build a clear-view template from a reappearance boundary only."""
        boundary = (
            track.observations[-min(12, len(track.observations)):]
            if from_end
            else track.observations[:min(12, len(track.observations))]
        )
        rejected = set(
            config.get(
                "reject_reasons",
                ["current_overlap", "ambiguous_sparse_graph"],
            )
        )
        candidates = [
            item for item in boundary
            if item.feature_is_fresh
            and item.feature is not None
            and item.feature_reason not in rejected
            and float(item.score) >= float(config.get("min_score", 0.60))
        ]
        if not candidates:
            return None
        limit = int(config.get("template_top_k", 3))
        candidates.sort(key=lambda item: float(item.score), reverse=True)
        template = np.mean([item.feature for item in candidates[:limit]], axis=0)
        template /= np.linalg.norm(template) + 1e-12
        return template

    def _prediction(self, track: Track, frame: int) -> np.ndarray:
        last = track.last
        src, dst = last.frame - self.frame_offset, frame - self.frame_offset
        predicted = transform_box(last.box, self.h[dst] @ np.linalg.inv(self.h[src]))
        if len(track.observations) >= 2:
            previous = track.observations[-2]
            gap = max(1, last.frame - previous.frame)
            velocity = ((last.box[:2] + last.box[2:]) - (previous.box[:2] + previous.box[2:])) / (2 * gap)
            dt = frame - last.frame
            predicted[:2] += velocity * dt
            predicted[2:] += velocity * dt
        return predicted

    def _observability_persistence(
        self, track: Track, gap: int, config: dict
    ) -> float:
        """Estimate whether a disappeared track is likely still in view.

        This is deliberately a lightweight *prior*, not a new identity memory:
        it uses only the frozen detection stream around the disappearance.
        A reliable, interior trajectory with a nearby overlapping person box is
        more consistent with temporary occlusion than with an image-boundary
        exit.  The score decays with the unobserved duration.
        """
        last = track.last
        center = (last.box[:2] + last.box[2:]) / 2
        box_height = max(float(last.box[3] - last.box[1]), 25.0)
        edge_distance = min(
            float(center[0]), float(last.width - center[0]),
            float(center[1]), float(last.height - center[1]),
        )
        interior = float(np.clip(edge_distance / box_height, 0.0, 1.0))
        peers = [
            item for item in self.frame_detections.get(last.frame, [])
            if item is not last
        ]
        if peers:
            overlap = float(np.max(box_iou_matrix(
                last.box[None], np.stack([item.box for item in peers])
            )))
        else:
            overlap = 0.0
        overlap_floor = float(config.get("overlap_floor", 0.05))
        occlusion = float(np.clip(
            (overlap - overlap_floor) / max(1.0 - overlap_floor, 1e-6),
            0.0, 1.0,
        ))
        length = min(
            len(track.observations) / max(
                float(config.get("minimum_track_length", 4)), 1.0
            ),
            1.0,
        )
        score_floor = float(config.get("score_floor", 0.45))
        confidence = float(np.clip(
            (float(last.score) - score_floor) / max(1.0 - score_floor, 1e-6),
            0.0, 1.0,
        ))
        evidence = 0.35 * length + 0.30 * confidence + 0.25 * interior + 0.10 * occlusion
        decay = np.exp(-max(gap - 1, 0) / max(float(config.get("decay_frames", 24)), 1.0))
        return float(np.clip(evidence * decay, 0.0, 1.0))

    def _match(
        self, tracks: List[Track], detections: List[Detection], frame: int, stage: str
    ) -> List[Tuple[int, int]]:
        if not tracks or not detections:
            return []
        predicted = np.stack([self._prediction(track, frame) for track in tracks])
        boxes = np.stack([det.box for det in detections])
        iou = box_iou_matrix(predicted, boxes)
        track_features = np.stack([track.feature for track in tracks])
        det_features = np.stack([det.feature for det in detections])
        appearance = np.clip(track_features @ det_features.T, -1, 1)
        # An EMA can forget an older but clean view after a long occlusion.
        # Use the best recent clean observation as an additional cue.
        for row, track in enumerate(tracks):
            memory_vectors = [track.feature]
            if track.short_feature is not None:
                memory_vectors.append(track.short_feature)
            if (
                self.feature_cfg.get("prototype") == "dual_timescale_quality"
                and track.stable_feature is not None
            ):
                memory_vectors.append(track.stable_feature)
            if track.gallery:
                memory_vectors.extend(track.gallery[-12:])
            if memory_vectors:
                gallery = np.stack(memory_vectors)
                appearance[row] = np.maximum(
                    appearance[row],
                    np.max(gallery @ det_features.T, axis=0),
                )
        pc = (predicted[:, :2] + predicted[:, 2:]) / 2
        dc = (boxes[:, :2] + boxes[:, 2:]) / 2
        height = np.maximum((predicted[:, 3] - predicted[:, 1])[:, None], 25)
        center = np.linalg.norm(pc[:, None] - dc[None], axis=2) / height
        size_ratio = np.maximum(
            ((predicted[:, 2] - predicted[:, 0]) * (predicted[:, 3] - predicted[:, 1]))[:, None],
            ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[None],
        ) / np.maximum(
            np.minimum(
                ((predicted[:, 2] - predicted[:, 0]) * (predicted[:, 3] - predicted[:, 1]))[:, None],
                ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[None],
            ),
            1,
        )
        motion_score = np.exp(-center)
        size_score = np.exp(-np.abs(np.log(size_ratio)))
        if stage == "adjacent":
            mask = np.array(
                [[mask_iou(track.last, det) for det in detections] for track in tracks],
                np.float32,
            )
            affinity = .34 * iou + .18 * mask + .30 * appearance + .13 * motion_score + .05 * size_score
            valid = ((iou > .025) | (center < 1.35)) & (appearance >= self.cfg["appearance_gate"])
            max_cost = self.cfg["adjacent_max_cost"]
        elif stage == "lost":
            gaps = np.array([frame - track.last.frame for track in tracks])[:, None]
            affinity = .20 * iou + .52 * appearance + .23 * motion_score + .05 * size_score
            valid = (
                (appearance >= self.cfg["lost_appearance_gate"])
                & (center < 2.2 + .075 * gaps)
                & (size_ratio < 3.5)
            )
            max_cost = self.cfg["lost_max_cost"]
        else:
            mask = np.array(
                [[mask_iou(track.last, det) for det in detections] for track in tracks],
                np.float32,
            )
            affinity = (
                .42 * iou + .20 * mask + .18 * appearance
                + .15 * motion_score + .05 * size_score
            )
            valid = (
                ((iou > .02) | (center < 1.15))
                & (appearance >= self.cfg.get("low_appearance_gate", .45))
            )
            max_cost = self.cfg.get("low_max_cost", .80)
        cost = 1 - affinity
        cost[~valid] = 1e5
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]

    def _global_link(self, tracks: List[Track]) -> None:
        """Stitch tracklets, optionally extending only beyond the normal gap.

        The normal global linker is unchanged and handles gaps up to
        ``global_max_gap``.  When explicitly enabled, the inactive identity
        memory stores an ended tracklet's stable feature and last state, then
        considers it only for a later tracklet whose gap is *larger* than that
        normal window.  It therefore cannot duplicate a normal short-gap
        association.
        """
        link_start = time.perf_counter()
        self.stats["global_pair_capacity"] = len(tracks) * len(tracks)
        temporal_pairs = 0
        motion_pairs = 0
        scored_pairs = 0
        candidates = []
        memory_candidates = []
        memory_pairs: set[tuple[int, int]] = set()
        memory_cfg = self.cfg.get("inactive_memory", {})
        memory_enabled = bool(memory_cfg.get("enabled", False))
        observability_cfg = self.cfg.get("observability_fusion", {})
        observability_enabled = bool(observability_cfg.get("enabled", False))
        stable_read_enabled = (
            self.feature_cfg.get("prototype") == "stable_read_gated"
        )
        diverse_memory_enabled = (
            self.feature_cfg.get("prototype") == "quality_diverse_bank"
        )
        state_template_cfg = self.feature_cfg.get("state_aware_template", {})
        state_template_enabled = bool(state_template_cfg.get("enabled", False))
        stable_read_min_gap = int(
            self.feature_cfg.get("stable_read_min_gap", 15)
        )
        stable_read_min_similarity = float(
            self.feature_cfg.get("stable_read_min_similarity", 0.78)
        )
        stable_read_bypass_probability = float(
            self.feature_cfg.get("stable_read_bypass_probability", 0.94)
        )
        global_max_gap = int(self.cfg["global_max_gap"])
        memory_max_gap = int(memory_cfg.get("max_gap", global_max_gap))
        model_path = self.cfg.get("link_model")
        linker = getattr(self, "linker_override", None)
        if linker is None:
            linker = LinkMLP(Path(model_path)) if model_path and Path(model_path).exists() else None
        self.stats["dual_head_link_enabled"] = bool(
            linker is not None and linker.has_rank_head
        )
        part_link_cfg = self.cfg.get("part_link_ranker", {})
        part_link_enabled = bool(part_link_cfg.get("enabled", False))
        if part_link_enabled and (linker is None or not linker.has_part_rank_head):
            raise ValueError("part_link_ranker is enabled but the LinkMLP has no V71 part rank head")
        self.stats["part_link_ranker_enabled"] = part_link_enabled
        rejector_cfg = self.cfg.get("link_rejector", {})
        rejector_enabled = bool(rejector_cfg.get("enabled", False))
        rejector_path = rejector_cfg.get("model")
        rejector = (
            LinkRejector(Path(rejector_path))
            if rejector_enabled and rejector_path and Path(rejector_path).exists()
            else None
        )
        if rejector_enabled and rejector is None:
            raise ValueError("link_rejector is enabled but its model is unavailable")
        ranker_cfg = self.cfg.get("local_ranker", {})
        ranker_enabled = bool(ranker_cfg.get("enabled", False))
        ranker_path = ranker_cfg.get("model")
        ranker_paths = ranker_cfg.get("models", [])
        if ranker_enabled and ranker_paths:
            ranker = EnsembleLocalRanker([Path(path) for path in ranker_paths])
        else:
            ranker = (
                LocalRanker(Path(ranker_path))
                if ranker_enabled and ranker_path and Path(ranker_path).exists()
                else None
            )
        if ranker_enabled and ranker is None:
            raise ValueError("local_ranker is enabled but its model is unavailable")
        part_cfg = self.cfg.get("part_ranker", {})
        part_enabled = bool(part_cfg.get("enabled", False))
        part_path = part_cfg.get("model")
        part_ranker = LocalRanker(Path(part_path)) if part_enabled and part_path and Path(part_path).exists() else None
        if part_enabled and part_ranker is None:
            raise ValueError("part_ranker is enabled but its model is unavailable")
        score_cfg = self.cfg.get("association_score", {})
        score_enabled = bool(score_cfg.get("enabled", False))
        score_path = score_cfg.get("model")
        association_score = PartAwareAssociationScore(Path(score_path)) if score_enabled and score_path and Path(score_path).exists() else None
        if score_enabled and association_score is None:
            raise ValueError("association_score is enabled but its model is unavailable")
        self.stats["part_aware_association_score_enabled"] = score_enabled
        pair_details: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
        outgoing_scores: dict[int, list[tuple[int, float]]] = {}
        incoming_scores: dict[int, list[tuple[int, float]]] = {}
        for first in tracks:
            for second in tracks:
                gap = second.observations[0].frame - first.last.frame
                if gap <= 0 or gap > global_max_gap:
                    continue
                temporal_pairs += 1
                similarity = float(first.feature @ second.feature)
                predicted = self._prediction(first, second.observations[0].frame)
                target = second.observations[0].box
                pc, dc = (predicted[:2] + predicted[2:]) / 2, (target[:2] + target[2:]) / 2
                scale = max(predicted[3] - predicted[1], target[3] - target[1], 25)
                distance = float(np.linalg.norm(pc - dc) / scale)
                if distance > 2.5 + .06 * gap:
                    continue
                motion_pairs += 1
                if linker:
                    scored_pairs += 1
                    raw_features = linker.features(first, second)
                    probability = (
                        linker.score_pair(first, second, raw_features)
                        if hasattr(linker, "score_pair")
                        else linker.score(raw_features)
                    )
                    pair_details[(first.track_id, second.track_id)] = (
                        probability, raw_features,
                    )
                    outgoing_scores.setdefault(first.track_id, []).append(
                        (second.track_id, probability)
                    )
                    incoming_scores.setdefault(second.track_id, []).append(
                        (first.track_id, probability)
                    )
                    threshold = float(
                        self.cfg.get("link_threshold", linker.threshold)
                    )
                    baseline_probability = probability
                    # V54: retain the baseline decision everywhere except an
                    # explicitly pre-registered near-threshold gap.  The
                    # permanence prior can only promote a candidate; it never
                    # rejects a baseline link or changes ordinary association.
                    if (
                        observability_enabled
                        and int(observability_cfg.get("min_gap", 3)) <= gap
                        <= int(observability_cfg.get("max_gap", 30))
                        and probability >= float(
                            observability_cfg.get("probability_floor", 0.83)
                        )
                        and probability < threshold
                    ):
                        self.stats["observability_fusion_reads"] += 1
                        persistence = self._observability_persistence(
                            first, gap, observability_cfg
                        )
                        minimum_persistence = float(
                            observability_cfg.get("minimum_persistence", 0.70)
                        )
                        if persistence >= minimum_persistence:
                            self.stats["observability_fusion_survivors"] += 1
                            evidence = (
                                persistence - minimum_persistence
                            ) / max(1.0 - minimum_persistence, 1e-6)
                            bonus = float(
                                observability_cfg.get("max_boost", 0.06)
                            ) * evidence
                            probability = min(probability + bonus, 1.0)
                            if probability > baseline_probability:
                                self.stats["observability_fusion_boosts"] += 1
                    if baseline_probability < threshold <= probability:
                        self.stats["observability_fusion_new_candidates"] += 1
                    if (
                        state_template_enabled
                        and gap >= int(state_template_cfg.get("min_gap", 15))
                        and probability >= float(
                            state_template_cfg.get("probability_floor", 0.83)
                        )
                        and probability < float(
                            state_template_cfg.get("probability_ceiling", 0.94)
                        )
                    ):
                        first_template = self._state_template(
                            first, from_end=True, config=state_template_cfg
                        )
                        second_template = self._state_template(
                            second, from_end=False, config=state_template_cfg
                        )
                        if first_template is None or second_template is None:
                            self.stats["state_template_unavailable"] += 1
                        else:
                            self.stats["state_template_global_reads"] += 1
                            template_similarity = float(
                                first_template @ second_template
                            )
                            minimum_similarity = float(
                                state_template_cfg.get("min_similarity", 0.82)
                            )
                            if template_similarity >= minimum_similarity:
                                evidence = (
                                    template_similarity - minimum_similarity
                                ) / max(1.0 - minimum_similarity, 1e-6)
                                probability += float(
                                    state_template_cfg.get("max_boost", 0.06)
                                ) * evidence
                                probability = min(probability, 1.0)
                                self.stats["state_template_global_boosts"] += 1
                    if (
                        diverse_memory_enabled
                        and gap >= int(self.feature_cfg.get("bank_read_min_gap", 15))
                        and probability
                        >= float(self.feature_cfg.get("bank_read_probability_floor", 0.83))
                        and probability
                        < float(self.feature_cfg.get("bank_read_probability_ceiling", 0.94))
                        and first.prototype_bank
                        and second.prototype_bank
                    ):
                        self.stats["diverse_memory_global_reads"] += 1
                        memory_similarity = self._diverse_memory_similarity(
                            first,
                            second,
                            int(self.feature_cfg.get("bank_read_top_k", 2)),
                        )
                        minimum_memory_similarity = float(
                            self.feature_cfg.get("bank_read_min_similarity", 0.82)
                        )
                        if memory_similarity >= minimum_memory_similarity:
                            evidence = (
                                memory_similarity - minimum_memory_similarity
                            ) / max(1.0 - minimum_memory_similarity, 1e-6)
                            probability += float(
                                self.feature_cfg.get("bank_read_max_boost", 0.06)
                            ) * evidence
                            probability = min(probability, 1.0)
                            self.stats["diverse_memory_global_boosts"] += 1
                    if baseline_probability < threshold <= probability:
                        if diverse_memory_enabled:
                            self.stats["diverse_memory_new_candidates"] += 1
                        if state_template_enabled:
                            self.stats["state_template_new_candidates"] += 1
                    if probability >= threshold:
                        # V46 is a rejection-only precision gate.  It cannot
                        # create a new global link, alter the baseline MLP
                        # input, or affect short-gap/high-confidence links.
                        # When either stable prototype is unavailable, the
                        # original V42-B1 candidate is retained unchanged.
                        if (
                            stable_read_enabled
                            and gap >= stable_read_min_gap
                            and probability < stable_read_bypass_probability
                        ):
                            if (
                                first.stable_feature is None
                                or second.stable_feature is None
                            ):
                                self.stats["quality_memory_global_unavailable"] += 1
                            else:
                                self.stats["quality_memory_global_reads"] += 1
                                stable_similarity = float(
                                    first.stable_feature @ second.stable_feature
                                )
                                if stable_similarity < stable_read_min_similarity:
                                    self.stats["quality_memory_global_rejections"] += 1
                                    continue
                                self.stats["quality_memory_global_passes"] += 1
                        candidates.append(
                            (1 - probability, first, second, probability)
                        )
                elif similarity >= self.cfg["global_min_similarity"]:
                    scored_pairs += 1
                    score = .72 * similarity + .28 * np.exp(-distance)
                    cost = 1 - score
                    if cost <= self.cfg["global_max_cost"]:
                        candidates.append((cost, first, second, score))
        self.stats["global_link_candidates"] = len(candidates)

        # V56 is a strictly reject-only post-calibration.  It sees every
        # ordinary LinkMLP-scored pair to compute candidate competition, but it
        # evaluates only pairs that V42-B1 already accepted.  It can neither
        # create a candidate nor alter the base LinkMLP score.
        if rejector is not None and candidates:
            filtered = []
            for item in candidates:
                _cost, first, second, score = item
                detail = pair_details.get((first.track_id, second.track_id))
                if detail is None:
                    filtered.append(item)
                    continue
                base_score, raw_features = detail
                outgoing_alternatives = [
                    value for target_id, value in outgoing_scores[first.track_id]
                    if target_id != second.track_id
                ]
                incoming_alternatives = [
                    value for source_id, value in incoming_scores[second.track_id]
                    if source_id != first.track_id
                ]
                best_outgoing = max(outgoing_alternatives, default=0.0)
                best_incoming = max(incoming_alternatives, default=0.0)
                features = np.concatenate((
                    np.asarray([base_score], dtype=np.float32),
                    raw_features.astype(np.float32),
                    np.asarray([
                        base_score - best_outgoing,
                        base_score - best_incoming,
                    ], dtype=np.float32),
                ))
                self.stats["link_rejector_reads"] += 1
                keep, _probability = rejector.keep(features)
                if keep:
                    self.stats["link_rejector_kept"] += 1
                    filtered.append(item)
                else:
                    self.stats["link_rejector_rejected"] += 1
            candidates = filtered

        # V62: a rank head in the same LinkMLP shares the frozen gate-head
        # representation.  Gate scores and admitted candidate edges are fixed;
        # only conflicts are re-ordered by the optional rank head.
        if linker is not None and linker.has_rank_head and candidates:
            source_degree: dict[int, int] = {}
            target_degree: dict[int, int] = {}
            for _cost, first, second, _score in candidates:
                source_degree[first.track_id] = source_degree.get(first.track_id, 0) + 1
                target_degree[second.track_id] = target_degree.get(second.track_id, 0) + 1
            reranked = []
            for cost, first, second, score in candidates:
                if source_degree[first.track_id] <= 1 and target_degree[second.track_id] <= 1:
                    reranked.append((cost, first, second, score))
                    continue
                detail = pair_details.get((first.track_id, second.track_id))
                if detail is None:
                    reranked.append((cost, first, second, score))
                    continue
                _gate_score, raw_features = detail
                reranked.append((1.0 - linker.rank_score(raw_features), first, second, score))
                self.stats["dual_head_ranked_edges"] += 1
            candidates = reranked

        # V59: re-score only candidate edges in a one-to-many or many-to-one
        # competition. Candidate construction and the LinkMLP threshold remain
        # frozen; isolated links keep their exact V42-B1 cost.
        if ranker is not None and candidates:
            source_degree: dict[int, int] = {}
            target_degree: dict[int, int] = {}
            for _cost, first, second, _score in candidates:
                source_degree[first.track_id] = source_degree.get(first.track_id, 0) + 1
                target_degree[second.track_id] = target_degree.get(second.track_id, 0) + 1
            reranked = []
            for cost, first, second, score in candidates:
                if (
                    source_degree[first.track_id] <= 1
                    and target_degree[second.track_id] <= 1
                ):
                    reranked.append((cost, first, second, score))
                    continue
                detail = pair_details.get((first.track_id, second.track_id))
                if detail is None:
                    reranked.append((cost, first, second, score))
                    continue
                base_score, raw_features = detail
                ranker_columns = ranker_cfg.get("feature_columns")
                ranker_features = (
                    raw_features[np.asarray(ranker_columns, dtype=np.int64)]
                    if ranker_columns is not None else raw_features
                )
                local_features = np.concatenate((
                    np.asarray([base_score], np.float32), ranker_features.astype(np.float32)
                ))
                reranked.append((1.0 - ranker.score(local_features), first, second, score))
                self.stats["local_ranker_ambiguous_edges"] += 1
            candidates = reranked

        # V70 replaces V59 only for existing ambiguous edges.  The gate,
        # candidate graph and isolated-link costs remain frozen.
        if part_ranker is not None and candidates:
            source_degree, target_degree = {}, {}
            for _cost, first, second, _score in candidates:
                source_degree[first.track_id] = source_degree.get(first.track_id, 0) + 1
                target_degree[second.track_id] = target_degree.get(second.track_id, 0) + 1
            reranked = []
            for cost, first, second, score in candidates:
                if source_degree[first.track_id] <= 1 and target_degree[second.track_id] <= 1:
                    reranked.append((cost, first, second, score)); continue
                detail = pair_details.get((first.track_id, second.track_id))
                if detail is None:
                    reranked.append((cost, first, second, score)); continue
                base_score, raw_features = detail
                features = np.concatenate((np.asarray([base_score], np.float32), raw_features.astype(np.float32), tracklet_part_features(first, second)))
                reranked.append((1.0 - part_ranker.score(features), first, second, score))
                self.stats["part_ranker_ambiguous_edges"] = self.stats.get("part_ranker_ambiguous_edges", 0) + 1
            candidates = reranked

        # V72-A: this is the final matching cost for ambiguous candidate
        # edges, not a post-LinkMLP model.  The formula and edge scope are
        # intentionally bit-for-bit equivalent to V70 for controlled refactor.
        if association_score is not None and candidates:
            source_degree, target_degree = {}, {}
            for _cost, first, second, _score in candidates:
                source_degree[first.track_id] = source_degree.get(first.track_id, 0) + 1
                target_degree[second.track_id] = target_degree.get(second.track_id, 0) + 1
            rescored = []
            for cost, first, second, score in candidates:
                if source_degree[first.track_id] <= 1 and target_degree[second.track_id] <= 1:
                    rescored.append((cost, first, second, score)); continue
                detail = pair_details.get((first.track_id, second.track_id))
                if detail is None:
                    rescored.append((cost, first, second, score)); continue
                link_probability, link_values = detail
                final_score = association_score.score(link_probability, link_values, tracklet_part_features(first, second))
                rescored.append((1.0 - final_score, first, second, score))
                self.stats["part_aware_association_edges"] = self.stats.get("part_aware_association_edges", 0) + 1
            candidates = rescored

        # V71 shares frozen LinkMLP hidden features with a part-aware rank
        # head.  It remains conflict-only: the gate and isolated links stay
        # identical, and no jointly visible part falls back to the gate score.
        if part_link_enabled and candidates:
            source_degree, target_degree = {}, {}
            for _cost, first, second, _score in candidates:
                source_degree[first.track_id] = source_degree.get(first.track_id, 0) + 1
                target_degree[second.track_id] = target_degree.get(second.track_id, 0) + 1
            reranked = []
            for cost, first, second, score in candidates:
                if source_degree[first.track_id] <= 1 and target_degree[second.track_id] <= 1:
                    reranked.append((cost, first, second, score)); continue
                detail = pair_details.get((first.track_id, second.track_id))
                if detail is None:
                    reranked.append((cost, first, second, score)); continue
                _base_score, raw_features = detail
                parts = tracklet_part_features(first, second)
                reranked.append((1.0 - linker.part_rank_score(raw_features, parts), first, second, score))
                self.stats["part_link_ranker_ambiguous_edges"] = self.stats.get("part_link_ranker_ambiguous_edges", 0) + 1
            candidates = reranked

        # This is an inactive-track identity memory, not a second pass over
        # the normal global-link window.  The tracklet ending at ``first`` is
        # the memory entry; its last box/motion and stable appearance prototype
        # are checked before a later newly started tracklet can recover its ID.
        if memory_enabled and memory_max_gap > global_max_gap:
            minimum_length = int(memory_cfg.get("minimum_track_length", 3))
            min_similarity = float(memory_cfg.get("min_similarity", .84))
            max_distance = float(memory_cfg.get("max_center_distance", 3.5))
            max_size_ratio = float(memory_cfg.get("max_size_ratio", 2.5))
            max_cost = float(memory_cfg.get("max_cost", .24))
            for first in tracks:
                if len(first.observations) < minimum_length:
                    continue
                for second in tracks:
                    gap = second.observations[0].frame - first.last.frame
                    if gap <= global_max_gap or gap > memory_max_gap:
                        continue
                    if len(second.observations) < minimum_length:
                        continue
                    self.stats["inactive_memory_temporal_pairs"] += 1
                    similarity = float(first.feature @ second.feature)
                    if similarity < min_similarity:
                        continue
                    predicted = self._prediction(first, second.observations[0].frame)
                    target = second.observations[0].box
                    pc = (predicted[:2] + predicted[2:]) / 2
                    dc = (target[:2] + target[2:]) / 2
                    scale = max(predicted[3] - predicted[1], target[3] - target[1], 25)
                    distance = float(np.linalg.norm(pc - dc) / scale)
                    first_area = max(
                        (predicted[2] - predicted[0]) * (predicted[3] - predicted[1]), 1
                    )
                    second_area = max(
                        (target[2] - target[0]) * (target[3] - target[1]), 1
                    )
                    size_ratio = max(first_area, second_area) / min(first_area, second_area)
                    if distance > max_distance or size_ratio > max_size_ratio:
                        continue
                    score = (
                        .78 * similarity
                        + .17 * np.exp(-distance)
                        + .05 * np.exp(-abs(np.log(size_ratio)))
                    )
                    cost = 1 - score
                    if cost <= max_cost:
                        item = (cost, first, second, score)
                        memory_candidates.append(item)
                        memory_pairs.add((first.track_id, second.track_id))
        self.stats["inactive_memory_candidates"] = len(memory_candidates)
        candidates.extend(memory_candidates)
        if candidates and float(
            self.cfg.get("link_mutual_margin", 0.0)
        ) > 0:
            margin = float(self.cfg["link_mutual_margin"])
            high_confidence = float(
                self.cfg.get("link_high_confidence", 0.92)
            )
            outgoing: dict[int, list[float]] = {}
            incoming: dict[int, list[float]] = {}
            for _cost, first, second, score in candidates:
                outgoing.setdefault(first.track_id, []).append(score)
                incoming.setdefault(second.track_id, []).append(score)
            filtered = []
            for item in candidates:
                _cost, first, second, score = item
                alternatives_out = sorted(
                    (
                        value
                        for value in outgoing[first.track_id]
                        if value != score
                    ),
                    reverse=True,
                )
                alternatives_in = sorted(
                    (
                        value
                        for value in incoming[second.track_id]
                        if value != score
                    ),
                    reverse=True,
                )
                outgoing_margin = score - (
                    alternatives_out[0] if alternatives_out else 0.0
                )
                incoming_margin = score - (
                    alternatives_in[0] if alternatives_in else 0.0
                )
                if (
                    score >= high_confidence
                    or (
                        outgoing_margin >= margin
                        and incoming_margin >= margin
                    )
                ):
                    filtered.append(item)
            candidates = filtered

        # V57: A greedy descending-score pass is sufficient for isolated
        # candidates, but can be suboptimal when several tracklet endings and
        # starts compete in the same local component.  Decompose only those
        # conflicts and solve each component with the existing Hungarian
        # solver.  Scores, gates, and candidates are intentionally untouched.
        confusion_cfg = self.cfg.get("confusion_decomposition", {})
        if bool(confusion_cfg.get("enabled", False)) and candidates:
            candidate_by_edge = {
                (first.track_id, second.track_id): item
                for item in candidates
                for _cost, first, second, _score in [item]
            }
            source_edges: dict[int, set[tuple[int, int]]] = {}
            target_edges: dict[int, set[tuple[int, int]]] = {}
            for edge in candidate_by_edge:
                source_edges.setdefault(edge[0], set()).add(edge)
                target_edges.setdefault(edge[1], set()).add(edge)

            remaining = set(candidate_by_edge)
            decomposed = []
            while remaining:
                seed = remaining.pop()
                component = {seed}
                pending = [seed]
                while pending:
                    source_id, target_id = pending.pop()
                    neighbours = (
                        source_edges.get(source_id, set())
                        | target_edges.get(target_id, set())
                    )
                    for edge in neighbours:
                        if edge not in component:
                            component.add(edge)
                            remaining.discard(edge)
                            pending.append(edge)

                self.stats["confusion_decomposition_components"] += 1
                sources = sorted({edge[0] for edge in component})
                targets = sorted({edge[1] for edge in component})
                ambiguous = len(component) > 1 and (
                    any(len(source_edges[source]) > 1 for source in sources)
                    or any(len(target_edges[target]) > 1 for target in targets)
                )
                if not ambiguous:
                    decomposed.extend(candidate_by_edge[edge] for edge in component)
                    continue

                self.stats["confusion_decomposition_ambiguous_components"] += 1
                self.stats["confusion_decomposition_candidate_edges"] += len(component)
                source_index = {value: index for index, value in enumerate(sources)}
                target_index = {value: index for index, value in enumerate(targets)}
                cost = np.full((len(sources), len(targets)), 1e5, np.float32)
                for source_id, target_id in component:
                    item = candidate_by_edge[(source_id, target_id)]
                    cost[source_index[source_id], target_index[target_id]] = item[0]
                rows, cols = linear_sum_assignment(cost)
                selected = {
                    (sources[row], targets[col])
                    for row, col in zip(rows, cols)
                    if cost[row, col] < 1e5
                }
                # Audit how often component optimisation differs from the
                # frozen greedy selection under exactly the same candidates.
                greedy_sources, greedy_targets, greedy = set(), set(), set()
                for item in sorted(
                    (candidate_by_edge[edge] for edge in component),
                    key=lambda value: value[0],
                ):
                    edge = (item[1].track_id, item[2].track_id)
                    if edge[0] not in greedy_sources and edge[1] not in greedy_targets:
                        greedy.add(edge)
                        greedy_sources.add(edge[0])
                        greedy_targets.add(edge[1])
                self.stats["confusion_decomposition_changed_links"] += len(
                    selected.symmetric_difference(greedy)
                )
                decomposed.extend(candidate_by_edge[edge] for edge in selected)
            candidates = decomposed
        used_end, used_start = set(), set()
        inactive_memory_links = 0
        parent = {track.track_id: track.track_id for track in tracks}
        for _cost, first, second, _score in sorted(
            candidates, key=lambda item: item[0]
        ):
            if first.track_id in used_end or second.track_id in used_start:
                continue
            used_end.add(first.track_id)
            used_start.add(second.track_id)
            parent[second.track_id] = first.track_id
            if (first.track_id, second.track_id) in memory_pairs:
                inactive_memory_links += 1
        self.stats["global_links"] = len(used_end)
        self.stats["inactive_memory_links"] = inactive_memory_links
        def root(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value
        for track in tracks:
            new_id = root(track.track_id)
            for observation in track.observations:
                observation.assigned_id = new_id
        self.stats["global_temporal_pairs"] = temporal_pairs
        self.stats["global_motion_pairs"] = motion_pairs
        self.stats["global_scored_pairs"] = scored_pairs
        self.stats["global_link_seconds"] = time.perf_counter() - link_start
