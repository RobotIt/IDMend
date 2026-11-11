"""Final matching costs for candidate edges; distinct from candidate gating."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PartAwareAssociationScore:
    """V72 final score for ambiguous edges, numerically identical to V70."""

    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.mean = np.asarray(payload["mean"], np.float32)
        self.std = np.asarray(payload["std"], np.float32)
        self.weight = np.asarray(payload["weight"], np.float32)
        self.bias = float(payload["bias"])

    def score(self, link_probability: float, link_features: np.ndarray, part_features: np.ndarray) -> float:
        values = np.concatenate((
            np.asarray([link_probability], np.float32),
            np.asarray(link_features, np.float32),
            np.asarray(part_features, np.float32),
        ))
        value = float(((values - self.mean) / self.std) @ self.weight + self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))
