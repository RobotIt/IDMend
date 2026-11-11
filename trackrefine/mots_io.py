from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class Detection:
    frame: int
    cls: int
    height: int
    width: int
    counts: str
    box: np.ndarray
    area: float
    score: float = 1.0
    feature: np.ndarray | None = None
    feature_is_fresh: bool = False
    feature_reason: str = ""
    assigned_id: int = -1
    source_id: int = -1

    @property
    def rle(self) -> dict:
        return {
            "size": [self.height, self.width],
            "counts": self.counts.encode("utf-8"),
        }


def read_mots(path: Path, target_class: int = 2) -> Dict[int, List[Detection]]:
    """Read MOTS rows as anonymous instance detections.

    The input identity column is intentionally discarded. This permits using
    an exported YOLO segmentation cache without leaking another tracker's IDs.
    """
    frames: Dict[int, List[Detection]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ", 5)
            if len(parts) != 6:
                continue
            frame, source_id, cls, height, width = map(int, parts[:5])
            if cls != target_class:
                continue
            rle = {
                "size": [height, width],
                "counts": parts[5].encode("utf-8"),
            }
            box = mask_utils.toBbox(rle).astype(np.float32)
            box[2:] += box[:2]
            area = float(mask_utils.area(rle))
            frames.setdefault(frame, []).append(
                Detection(
                    frame, cls, height, width, parts[5], box, area,
                    source_id=source_id,
                )
            )
    score_path = path.with_suffix(".scores.npz")
    if score_path.exists():
        ordered = [det for frame in sorted(frames) for det in frames[frame]]
        scores = np.load(score_path)["scores"].astype(np.float32)
        if len(scores) != len(ordered):
            raise ValueError(
                f"{score_path}: {len(scores)} scores for {len(ordered)} detections"
            )
        for detection, score in zip(ordered, scores):
            detection.score = float(score)
    return frames


def write_mots(path: Path, detections: Iterable[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(detections, key=lambda item: (item.frame, item.assigned_id))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for det in rows:
            if det.assigned_id < 0:
                continue
            handle.write(
                f"{det.frame} {det.assigned_id} {det.cls} "
                f"{det.height} {det.width} {det.counts}\n"
            )


def mask_iou(first: Detection, second: Detection) -> float:
    return float(mask_utils.iou([first.rle], [second.rle], [0])[0, 0])
