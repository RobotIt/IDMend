"""Frozen YOLO segmentation and native BoT-SORT export.

MOTS20 deliberately uses the original full-frame ``model.track`` route.
KITTI deliberately predicts first and replays one BoT-SORT per category.
Those routes are not interchangeable when reproducing the existing results.
No historical experiment directory or dataset annotation is imported here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools import mask as mu


SCHEMA_VERSION = 1
DETECTOR_OPTIONS = {
    "confidence": 0.12,
    "iou": 0.65,
    "image_size": 1280,
    "retina_masks": True,
    "minimum_mask_area": 20,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _manifest(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _validate_paths(image_paths: list[Path]) -> list[Path]:
    if not image_paths:
        raise ValueError("The image sequence is empty")
    paths = [Path(path).resolve(strict=True) for path in image_paths]
    if any(not path.is_file() for path in paths):
        raise ValueError("Every image path must identify a regular file")
    try:
        frames = [int(path.stem) for path in paths]
    except ValueError as error:
        raise ValueError("Image filenames must have numeric frame stems") from error
    if frames != sorted(set(frames)):
        raise ValueError("Image paths must be ordered by unique frame numbers")
    return paths


def _request(
    paths: list[Path], detector_weights: Path, botsort_config: Path,
    dataset: str, device: str, version: str,
) -> dict:
    # Metadata fingerprints avoid reading the full video a second time. Model,
    # configuration, implementation and generated artifacts use content hashes.
    inventory = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        inventory.update(
            json.dumps(
                [str(path), stat.st_size, stat.st_mtime_ns],
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8") + b"\n"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "route": "full_baseline_model_track" if dataset == "mots20" else "predict_classwise_replay",
        "frames": len(paths),
        "first_frame": int(paths[0].stem),
        "last_frame": int(paths[-1].stem),
        "image_inventory_sha256": inventory.hexdigest(),
        "image_inventory_fingerprint": "absolute path, size and mtime_ns; not pixel-content hashes",
        "detector_sha256": _sha256(detector_weights),
        "botsort_config_sha256": _sha256(botsort_config),
        "frontend_sha256": _sha256(Path(__file__)),
        "ultralytics": version,
        "device": str(device),
        "detector": DETECTOR_OPTIONS,
        "native_reid": False,
        "frame_numbers": "unchanged numeric filename stems",
    }


def _matching_manifest(path: Path, request: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid cache manifest: {path}") from error
    if saved.get("request") != request:
        raise RuntimeError(
            f"The cached frontend protocol does not match this run: {path}. "
            "Use a new output directory or explicitly remove the old cache."
        )
    return saved


def _verify_raw(raw_cache: Path, saved: dict) -> None:
    if not raw_cache.is_file() or _sha256(raw_cache) != saved.get("raw_sha256"):
        raise RuntimeError(f"Raw detection cache is missing or changed: {raw_cache}")


def _verify_native(native_dir: Path, saved: dict) -> None:
    for filename, digest in saved.get("files", {}).items():
        path = native_dir / filename
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"Native tracking cache is missing or changed: {path}")


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return image


def _raw_rows(result, shape: tuple[int, int]) -> list[dict]:
    rows = []
    if result.boxes is None or result.masks is None:
        return rows
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    masks = result.masks.data.cpu().numpy()
    if not (len(boxes) == len(scores) == len(labels) == len(masks)):
        raise RuntimeError("YOLO box and mask counts differ")
    height, width = shape
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if mask.shape != shape:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        rle = mu.encode(np.asfortranarray((mask > 0.5).astype(np.uint8)))
        rows.append({
            "box": box.tolist(), "score": float(score), "coco": int(label),
            "size": [height, width], "counts": rle["counts"].decode("ascii"),
        })
    return rows


def _write_native(native_dir: Path, buffers: dict, classes: tuple[int, ...], sort: bool) -> dict:
    native_dir.mkdir(parents=True, exist_ok=True)
    files, statistics = {}, {}
    for cls in classes:
        rows = buffers.get(cls, [])
        if sort:
            rows = sorted(rows, key=lambda row: (row[0], row[1]))
        target = native_dir / f"class{cls}.txt"
        temporary = target.with_name(target.name + ".partial")
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            for row in rows:
                handle.write(" ".join(map(str, row[:6])) + "\n")
        temporary.replace(target)
        score_path = target.with_suffix(".scores.npz")
        temporary_score = score_path.with_name(score_path.name + ".partial")
        with temporary_score.open("wb") as handle:
            np.savez_compressed(handle, scores=np.asarray([row[6] for row in rows], np.float32))
        temporary_score.replace(score_path)
        files[target.name] = _sha256(target)
        files[score_path.name] = _sha256(score_path)
        statistics[str(cls)] = {
            "rows": len(rows), "tracks": len({row[1] for row in rows}),
            "native_file": str(target), "scores_file": str(score_path),
        }
    return {"files": files, "classes": statistics}


def _mots20(paths, native_dir, raw_cache, weights, tracker_yaml, device, YOLO) -> dict:
    model = YOLO(str(weights))
    buffers = defaultdict(list)
    next_id = 1_000_000
    temporary = raw_cache.with_name(raw_cache.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as raw:
        for index, path in enumerate(paths, 1):
            image = _read_image(path)
            height, width = image.shape[:2]
            frame = int(path.stem)
            # Keep source as a filename and batch=1, exactly as the frozen
            # keyframe_detector full_baseline route (not predict+replay).
            result = model.track(
                source=str(path), stream=False, classes=[0],
                conf=0.12, iou=0.65, imgsz=1280, batch=1,
                device=device, retina_masks=True, verbose=False,
                persist=True, tracker=str(tracker_yaml),
            )[0]
            raw.write(json.dumps(
                {"frame": frame, "detections": _raw_rows(result, (height, width))},
                separators=(",", ":"),
            ) + "\n")
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                identities = (
                    result.boxes.id.int().cpu().numpy()
                    if result.boxes.id is not None else None
                )
                instances = []
                for mask_index in np.argsort(scores)[::-1]:
                    mask = masks[mask_index]
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    binary = mask > 0.5
                    if int(binary.sum()) < 20:
                        continue
                    identity = int(identities[mask_index]) if identities is not None else next_id
                    next_id += 1
                    instances.append((binary, identity, float(scores[mask_index])))
                occupied = np.zeros((height, width), bool)
                # Python's stable sort preserves the original reverse-argsort
                # tie order, including the anonymous-ID fallback behavior.
                for binary, identity, score in sorted(instances, key=lambda item: item[2], reverse=True):
                    clipped = binary & ~occupied
                    if int(clipped.sum()) < 20:
                        continue
                    occupied |= clipped
                    counts = mu.encode(np.asfortranarray(clipped.astype(np.uint8)))["counts"].decode("ascii")
                    buffers[2].append((frame, identity, 2, height, width, counts, score))
            if index % 100 == 0 or index == len(paths):
                raw.flush()
                print(f"FRONTEND mots20 {index}/{len(paths)}", flush=True)
    temporary.replace(raw_cache)
    del model
    result = _write_native(native_dir, buffers, (2,), sort=False)
    result["raw_cache_semantics"] = "model.track returned observations before overlap cleanup; not all pre-tracker detections"
    return result


def _kitti_predict(paths, raw_cache, weights, device, YOLO) -> None:
    model = YOLO(str(weights))
    temporary = raw_cache.with_name(raw_cache.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, path in enumerate(paths, 1):
            image = _read_image(path)
            result = model.predict(
                image, conf=0.12, iou=0.65, imgsz=1280, classes=[0, 2],
                retina_masks=True, device=device, verbose=False,
            )[0]
            handle.write(json.dumps({
                "frame": int(path.stem),
                "detections": _raw_rows(result, image.shape[:2]),
            }, separators=(",", ":")) + "\n")
            if index % 50 == 0 or index == len(paths):
                handle.flush()
                print(f"FRONTEND kitti predict {index}/{len(paths)}", flush=True)
    temporary.replace(raw_cache)
    del model


def _kitti_replay(paths, native_dir, raw_cache, tracker_config) -> dict:
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace

    # The original experiment ran a distinct native tracker per semantic class.
    states = {
        cls: BOTSORT(args=IterableSimpleNamespace(**copy.deepcopy(tracker_config)))
        for cls in (1, 2)
    }
    buffers = defaultdict(list)
    with raw_cache.open(encoding="utf-8") as handle:
        for index, path in enumerate(paths, 1):
            line = handle.readline()
            if not line:
                raise RuntimeError("Raw cache ends before the image sequence")
            record = json.loads(line)
            frame = int(path.stem)
            if record["frame"] != frame:
                raise RuntimeError("Raw cache frame numbers differ from the image sequence")
            image = _read_image(path)
            height, width = image.shape[:2]
            for cls in (1, 2):
                rows = [row for row in record["detections"] if row["coco"] == (2 if cls == 1 else 0)]
                array = np.asarray(
                    [[*row["box"], row["score"], row["coco"]] for row in rows], np.float32,
                ).reshape(-1, 6)
                tracks = states[cls].update(Boxes(array, image.shape[:2]), image)
                entries = []
                for track in tracks:
                    detection_index, identity = int(track[-1]), int(track[4])
                    if not 0 <= detection_index < len(rows):
                        raise RuntimeError(f"BoT-SORT returned invalid mask index {detection_index}")
                    entries.append((identity, detection_index, rows[detection_index]))
                occupied = np.zeros((height, width), bool)
                for identity, _, row in sorted(entries, key=lambda item: item[2]["score"], reverse=True):
                    if row["size"] != [height, width]:
                        raise RuntimeError("Raw cache mask dimensions differ from the image")
                    rle = {"size": row["size"], "counts": row["counts"].encode("ascii")}
                    x, y, bw, bh = mu.toBbox(rle).astype(int)
                    patch = mu.decode(rle).astype(bool)[y:y + bh, x:x + bw]
                    region = occupied[y:y + bh, x:x + bw]
                    clipped = patch & ~region
                    if int(clipped.sum()) < 20:
                        continue
                    counts = row["counts"]
                    if np.any(patch & region):
                        mask = np.zeros((height, width), np.uint8)
                        mask[y:y + bh, x:x + bw] = clipped
                        counts = mu.encode(np.asfortranarray(mask))["counts"].decode("ascii")
                    region |= clipped
                    buffers[cls].append((frame, identity, cls, height, width, counts, row["score"]))
            if index % 100 == 0 or index == len(paths):
                print(f"FRONTEND kitti replay {index}/{len(paths)}", flush=True)
        if handle.readline():
            raise RuntimeError("Raw cache contains frames beyond the image sequence")
    result = _write_native(native_dir, buffers, (1, 2), sort=True)
    result["raw_cache_semantics"] = "all predict detections before native tracking; original KITTI background-LK input"
    return result


def export_sequence(
    image_paths: list[Path], native_dir: Path, raw_cache: Path,
    detector_weights: Path, botsort_config: Path, dataset: str,
    device: str = "cuda",
) -> dict:
    """Export an entire sequence, or reuse verified completed stage artifacts.

    Interrupted KITTI prediction restarts that stage; completed raw predictions
    can be reused while replaying tracking from frame one. MOTS tracking cannot
    safely resume mid-sequence without restoring tracker/GMC state, so an
    unfinished MOTS sequence restarts from its first frame.
    """
    if dataset not in {"mots20", "kitti"}:
        raise ValueError("dataset must be 'mots20' or 'kitti'")
    paths = _validate_paths(image_paths)
    weights = Path(detector_weights).resolve(strict=True)
    tracker_yaml = Path(botsort_config).resolve(strict=True)
    if not weights.is_file() or not tracker_yaml.is_file():
        raise ValueError("Detector weights and tracker YAML must be local files")
    tracker_config = yaml.safe_load(tracker_yaml.read_text(encoding="utf-8"))
    if tracker_config.get("tracker_type") != "botsort" or tracker_config.get("with_reid", False):
        raise ValueError("The frozen frontend requires BoT-SORT with internal ReID disabled")
    native_dir = Path(native_dir).resolve()
    raw_cache = Path(raw_cache).resolve()
    native_dir.mkdir(parents=True, exist_ok=True)
    raw_cache.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
    from ultralytics import YOLO, __version__ as ultra_version

    request = _request(paths, weights, tracker_yaml, dataset, device, ultra_version)
    native_manifest = native_dir / "frontend.manifest.json"
    raw_manifest = _manifest(raw_cache)
    saved_native = _matching_manifest(native_manifest, request)
    saved_raw = _matching_manifest(raw_manifest, request)
    if saved_native is not None:
        _verify_native(native_dir, saved_native)
        if saved_raw is None:
            raise RuntimeError(f"Native tracking cache has no raw detection manifest: {raw_cache}")
        _verify_raw(raw_cache, saved_raw)
        if saved_native["raw_sha256"] != saved_raw["raw_sha256"]:
            raise RuntimeError("Native and raw detection cache manifests disagree")
        return {**saved_native["report"], "cache_hit": True}

    started = time.perf_counter()
    raw_reused = False
    if dataset == "mots20":
        if saved_raw is not None:
            _verify_raw(raw_cache, saved_raw)
        exported = _mots20(paths, native_dir, raw_cache, weights, tracker_yaml, device, YOLO)
    else:
        if saved_raw is not None:
            _verify_raw(raw_cache, saved_raw)
            raw_reused = True
        else:
            _kitti_predict(paths, raw_cache, weights, device, YOLO)
            _dump(raw_manifest, {"request": request, "raw_sha256": _sha256(raw_cache)})
        exported = _kitti_replay(paths, native_dir, raw_cache, tracker_config)

    raw_digest = _sha256(raw_cache)
    _dump(raw_manifest, {"request": request, "raw_sha256": raw_digest})
    report = {
        "dataset": dataset, "frames": len(paths), "native_dir": str(native_dir),
        "raw_cache": str(raw_cache), "raw_sha256": raw_digest,
        "classes": exported["classes"], "raw_cache_semantics": exported["raw_cache_semantics"],
        "cache_hit": False, "raw_cache_reused": raw_reused,
        "seconds": time.perf_counter() - started,
        "protocol": request,
    }
    _dump(native_manifest, {
        "request": request, "raw_sha256": raw_digest,
        "files": exported["files"], "report": report,
    })
    return report
