from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LocalRanker:
    """Small frozen linear scorer for ranking existing LinkMLP candidates."""

    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.mean = np.asarray(payload["mean"], np.float32)
        self.std = np.asarray(payload["std"], np.float32)
        self.weight = np.asarray(payload["weight"], np.float32)
        self.bias = float(payload["bias"])

    def score(self, features: np.ndarray) -> float:
        value = float(((features - self.mean) / self.std) @ self.weight + self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class EnsembleLocalRanker:
    """Frozen score average of independently normalised local rankers."""

    def __init__(self, paths: list[Path]):
        if not paths:
            raise ValueError("EnsembleLocalRanker requires at least one model")
        self.members = [LocalRanker(path) for path in paths]

    def score(self, features: np.ndarray) -> float:
        return float(np.mean([member.score(features) for member in self.members]))
