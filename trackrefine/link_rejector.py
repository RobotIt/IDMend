"""Optional reject-only calibration for already accepted global link pairs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinkRejector:
    """A compact, serialized logistic model that can only veto a base link."""

    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "trackrefine-link-rejector-v1":
            raise ValueError(f"Unsupported link rejector model: {path}")
        self.feature_names = tuple(payload["feature_names"])
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.std = np.maximum(
            np.asarray(payload["std"], dtype=np.float32), 1e-6,
        )
        self.weights = np.asarray(payload["weights"], dtype=np.float32)
        self.bias = float(payload["bias"])
        self.threshold = float(payload["threshold"])
        if not (
            len(self.feature_names)
            == len(self.mean)
            == len(self.std)
            == len(self.weights)
        ):
            raise ValueError(f"Inconsistent rejector dimensions: {path}")

    def probability(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(
                f"Expected {len(self.mean)} rejector features, got {values.shape}"
            )
        logit = float(((values - self.mean) / self.std) @ self.weights + self.bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def keep(self, features: np.ndarray) -> tuple[bool, float]:
        probability = self.probability(features)
        return probability >= self.threshold, probability
