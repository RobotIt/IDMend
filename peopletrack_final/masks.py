"""Standalone class merging, conservative mask refinement and MOTS validation.

Refinement runs after final identity association. Instance metadata is preserved;
the original foreground of every other instance has priority over new pixels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu


_METHODS = frozenset(('none', 'reference', 'guard1', 'guard2', 'close3'))


def _classes(dataset: str) -> tuple[int, ...]:
    if dataset == 'mots20':
        return (2,)
    if dataset == 'kitti':
        return (1, 2)
    raise ValueError(f'Unsupported dataset: {dataset}')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_classes(native_or_linked_folder: Path, target: Path, dataset: str) -> None:
    """Renumber class tracks and preserve the frozen pedestrian-first KITTI merge."""
    source, target = Path(native_or_linked_folder), Path(target)
    data = defaultdict(list)
    for cls in _classes(dataset):
        path = source / f'class{cls}.txt'
        rows = path.read_text(encoding='utf-8').splitlines()
        identities = sorted({int(row.split(' ', 2)[1]) for row in rows})
        if len(identities) >= 1000:
            raise ValueError(f'{path}: too many tracks for MOTS IDs')
        mapping = {identity: cls * 1000 + index + 1
                   for index, identity in enumerate(identities)}
        for line in rows:
            frame, identity, row_class, height, width, counts = line.split(' ', 5)
            if int(row_class) != cls:
                raise ValueError(f'{path}: class {row_class}, expected {cls}')
            data[int(frame)].append((cls, mapping[int(identity)], int(height), int(width), counts))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='ascii', newline='\n') as handle:
        for frame, rows in sorted(data.items()):
            if dataset == 'mots20':
                for cls, identity, height, width, counts in sorted(rows, key=lambda row: row[1]):
                    handle.write(f'{frame} {identity} {cls} {height} {width} {counts}\n')
                continue
            occupied = np.zeros((rows[0][2], rows[0][3]), bool)
            for cls, identity, height, width, counts in sorted(rows, key=lambda row: (-row[0], row[1])):
                rle = {'size': [height, width], 'counts': counts.encode('ascii')}
                x, y, box_width, box_height = mu.toBbox(rle).astype(int)
                patch = mu.decode(rle)[y:y+box_height, x:x+box_width].astype(bool)
                region = occupied[y:y+box_height, x:x+box_width]
                clipped = patch & ~region
                if clipped.sum() < 20:
                    continue
                if np.any(patch & region):
                    mask = np.zeros((height, width), np.uint8)
                    mask[y:y+box_height, x:x+box_width] = clipped
                    counts = mu.encode(np.asfortranarray(mask))['counts'].decode('ascii')
                region |= clipped
                handle.write(f'{frame} {identity} {cls} {height} {width} {counts}\n')


def refine_mask_boundaries(mask, box, image, max_size=256, iter_count=1):
    """Original ROI GrabCut reference, with its downsampling and one-iteration rule."""
    (H, W) = mask.shape[:2]
    (x1, y1, x2, y2) = map(int, box)
    pad = 20
    x1_p = max(0, x1 - pad)
    y1_p = max(0, y1 - pad)
    x2_p = min(W, x2 + pad)
    y2_p = min(H, y2 + pad)
    img_roi = image[y1_p:y2_p, x1_p:x2_p]
    mask_roi = mask[y1_p:y2_p, x1_p:x2_p]
    mask_roi = mask_roi.astype(np.uint8) * 255
    if img_roi.size == 0 or mask_roi.size == 0:
        return mask
    (roi_h, roi_w) = img_roi.shape[:2]
    scale = 1.0
    if max(roi_h, roi_w) > max_size:
        scale = max_size / float(max(roi_h, roi_w))
        new_size = (int(roi_w * scale), int(roi_h * scale))
        img_proc = cv2.resize(img_roi, new_size, interpolation=cv2.INTER_AREA)
        mask_proc = cv2.resize(mask_roi, new_size, interpolation=cv2.INTER_NEAREST)
    else:
        img_proc = img_roi
        mask_proc = mask_roi
    m_roi_u8 = (mask_proc > 0).astype(np.uint8)
    k_size = max(3, int(min(img_proc.shape[:2]) * 0.05))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    sure_fg = cv2.erode(m_roi_u8, kernel, iterations=1)
    sure_bg = 1 - cv2.dilate(m_roi_u8, kernel, iterations=1)
    gc_mask = np.ones_like(m_roi_u8, dtype=np.uint8) * cv2.GC_PR_FGD
    gc_mask[sure_bg == 1] = cv2.GC_BGD
    gc_mask[sure_fg == 1] = cv2.GC_FGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_proc, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return mask
    refined_small = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
    if scale != 1.0:
        refined_bin = cv2.resize(refined_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        refined_bin = refined_small
    refined_mask = mask.copy()
    if refined_bin.sum() > 0:
        refined_mask[y1_p:y2_p, x1_p:x2_p] = refined_bin > 0
    return refined_mask


def _init_worker() -> None:
    cv2.setNumThreads(1)


def _process_frame(task):
    path, rows, methods = task
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    occupied = np.zeros(image.shape[:2], bool)
    originals = []
    for row in rows:
        rle = {'size': [int(row[3]), int(row[4])], 'counts': row[5].encode('ascii')}
        if tuple(rle['size']) != image.shape[:2]:
            raise ValueError(f'{path}: mask/image dimension mismatch')
        x, y, width, height = map(int, mu.toBbox(rle))
        if width <= 0 or height <= 0:
            raise ValueError(f'{path}: empty input mask {row[:2]}')
        a, b = max(0, y-20), max(0, x-20)
        c, d = min(image.shape[0], y+height+20), min(image.shape[1], x+width+20)
        decoded = mu.decode(rle).astype(bool)
        if (occupied & decoded).any():
            raise ValueError(f'{path}: input mask overlap {row[:2]}')
        occupied |= decoded
        originals.append((decoded[a:c, b:d].copy(), (a, b, c, d), (x-b, y-a, width, height)))
    output = []
    totals = {str(cls): {'rows': 0, 'changed': 0, 'pixels_added': 0, 'pixels_removed': 0}
              for cls in methods}
    claimed = np.zeros(image.shape[:2], bool)
    for row, (mask, (a, b, c, d), (x, y, width, height)) in zip(rows, originals):
        cls = int(row[2])
        method, area = methods[cls], int(mask.sum())
        candidate = mask.copy()
        reference = None
        if method in ('reference', 'guard1', 'guard2'):
            cv2.setRNGSeed((int(row[0])*100003+int(row[1])) % 2147483647)
            reference = refine_mask_boundaries(mask, (x, y, x+width, y+height), image[a:c, b:d])
        if method == 'reference':
            candidate = reference.copy()
        elif method in ('guard1', 'guard2') and area >= 100 and min(width, height) >= 8:
            radius = int(method[-1])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            ya, xb = max(0, y-3), max(0, x-3)
            yc, xd = min(mask.shape[0], y+height+3), min(mask.shape[1], x+width+3)
            crop = mask[ya:yc, xb:xd].astype(np.uint8)
            dil = cv2.dilate(crop, kernel).astype(bool)
            ero = cv2.erode(crop, kernel).astype(bool)
            patched = (reference[ya:yc, xb:xd] & dil) | ero
            ratio = float(patched.sum()) / area
            iou = np.count_nonzero(patched & crop.astype(bool)) / max(1, np.count_nonzero(patched | crop.astype(bool)))
            if .9 <= ratio <= 1.1 and iou >= .9:
                candidate[ya:yc, xb:xd] = patched
        elif method == 'close3':
            ya, xb = max(0, y-3), max(0, x-3)
            yc, xd = min(mask.shape[0], y+height+3), min(mask.shape[1], x+width+3)
            patch = cv2.morphologyEx(mask[ya:yc, xb:xd].astype(np.uint8), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            if patch.sum() <= 1.05*area:
                candidate[ya:yc, xb:xd] = patch.astype(bool)
        candidate &= ~(occupied[a:c, b:d] & ~mask)
        candidate &= ~claimed[a:c, b:d]
        if not candidate.any():
            candidate = mask.copy()
        if (candidate & claimed[a:c, b:d]).any():
            raise ValueError(f'{path}: refined mask overlap {row[:2]}')
        claimed[a:c, b:d] |= candidate
        changed = not np.array_equal(candidate, mask)
        stats = totals[str(cls)]
        stats['rows'] += 1
        stats['changed'] += int(changed)
        stats['pixels_added'] += int(np.count_nonzero(candidate & ~mask))
        stats['pixels_removed'] += int(np.count_nonzero(mask & ~candidate))
        if method == 'none' and changed:
            raise ValueError(f'{path}: disabled class refinement changed a mask')
        if changed:
            full = np.zeros(image.shape[:2], np.uint8)
            full[a:c, b:d] = candidate
            counts = mu.encode(np.asfortranarray(full))['counts'].decode('ascii')
        else:
            counts = row[5]
        output.append(' '.join([*row[:5], counts])+'\n')
    return output, totals


def refine_sequence(image_paths: list[Path], source: Path, target: Path,
                    methods: dict[int, str], workers: int = 4) -> dict:
    """Refine one merged sequence, keeping every frame/ID/class/size unchanged."""
    source, target = Path(source), Path(target)
    paths = sorted(map(Path, image_paths), key=lambda path: int(path.stem))
    image_map = {int(path.stem): path for path in paths}
    if not paths or len(image_map) != len(paths):
        raise ValueError('Image paths must be nonempty and have unique numeric stems')
    methods = {int(cls): method for cls, method in methods.items()}
    if not methods or any(cls not in (1, 2) for cls in methods):
        raise ValueError(f'Invalid class refinement methods: {methods}')
    if any(method not in _METHODS for method in methods.values()):
        raise ValueError(f'Unsupported refinement method: {methods}')
    frames = defaultdict(list)
    for line in source.read_text(encoding='ascii').splitlines():
        row = line.split(' ', 5)
        if len(row) != 6 or int(row[2]) not in methods:
            raise ValueError(f'{source}: missing or invalid class method: {line[:80]}')
        frame = int(row[0])
        if frame not in image_map:
            raise ValueError(f'{source}: frame {frame} has no image')
        frames[frame].append(row)
    tasks = [(str(path), sorted(frames[frame], key=lambda row: int(row[1])), methods)
             for frame, path in sorted(image_map.items()) if frame in frames]
    buffers = []
    totals = {str(cls): Counter() for cls in methods}
    start = time.time()
    if workers > 1 and tasks:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
        results = pool.map(_process_frame, tasks, chunksize=3)
    else:
        pool = None
        _init_worker()
        results = map(_process_frame, tasks)
    try:
        for index, (output, stats) in enumerate(results, 1):
            buffers.extend(output)
            for cls in methods:
                totals[str(cls)].update(stats[str(cls)])
            if index % 100 == 0 or index == len(tasks):
                print(f'REFINE {source.stem} {index}/{len(tasks)} {time.time()-start:.1f}s', flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    original = Counter(tuple(row[:5]) for rows in frames.values() for row in rows)
    actual = Counter(tuple(line.split(' ', 5)[:5]) for line in buffers)
    if original != actual:
        raise ValueError(f'{source}: refinement changed frame/ID/class/size metadata')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(''.join(buffers).encode('ascii'))
    return {'source': str(source), 'target': str(target), 'source_sha256': _sha(source),
            'sha256': _sha(target), 'image_frames': len(paths), 'frames_with_predictions': len(tasks),
            'rows': len(buffers), 'methods': {str(cls): method for cls, method in methods.items()},
            'stats': {cls: dict(stats) for cls, stats in totals.items()},
            'metadata_preserved': True, 'seconds': time.time()-start}


def validate_submission_file(path: Path, image_paths: list[Path], dataset: str) -> dict:
    """Validate canonical TXT, image dimensions, class IDs, RLE and pixel exclusion."""
    path = Path(path)
    image_map = {int(Path(image).stem): Path(image) for image in image_paths}
    if not image_map or len(image_map) != len(image_paths):
        raise ValueError('Image paths must be nonempty and have unique numeric stems')
    allowed = _classes(dataset)
    raw = path.read_bytes()
    if not raw or b'\r' in raw or raw.startswith(b'\xef\xbb\xbf') or not raw.endswith(b'\n'):
        raise ValueError(f'{path}: expected nonempty ASCII TXT with LF line endings')
    lines = raw.decode('ascii').splitlines()
    last_frame = None
    occupied = None
    ids, seen, counts = {}, set(), Counter()
    for line_number, line in enumerate(lines, 1):
        row = line.split(' ', 5)
        if len(row) != 6:
            raise ValueError(f'{path}:{line_number}: expected six fields')
        frame, identity, cls, height, width = map(int, row[:5])
        if frame not in image_map or cls not in allowed or identity <= 0 or min(height, width) <= 0:
            raise ValueError(f'{path}:{line_number}: invalid frame/ID/class/dimensions')
        if dataset == 'kitti' and (identity // 1000 != cls or not 0 < identity % 1000 < 1000):
            raise ValueError(f'{path}:{line_number}: ID does not encode its class')
        if ids.setdefault(identity, cls) != cls or (frame, identity) in seen:
            raise ValueError(f'{path}:{line_number}: duplicate or inconsistent identity')
        if frame != last_frame:
            if last_frame is not None and frame < last_frame:
                raise ValueError(f'{path}:{line_number}: unsorted frames')
            image = cv2.imread(str(image_map[frame]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_map[frame])
            shape = image.shape[:2]
            occupied = np.zeros(shape, bool)
            last_frame = frame
        if (height, width) != occupied.shape:
            raise ValueError(f'{path}:{line_number}: RLE/image dimension mismatch')
        rle = {'size': [height, width], 'counts': row[5].encode('ascii')}
        if not row[5] or float(mu.area(rle)) <= 0:
            raise ValueError(f'{path}:{line_number}: empty mask')
        mask = mu.decode(rle).astype(bool)
        if not mask.any() or np.any(occupied & mask):
            raise ValueError(f'{path}:{line_number}: empty or overlapping mask')
        occupied |= mask
        seen.add((frame, identity))
        counts[str(cls)] += 1
    return {'path': str(path), 'image_frames': len(image_map), 'frames_with_predictions': len({f for f, _ in seen}),
            'rows': len(lines), 'tracks': len(ids), 'rows_by_class': dict(counts),
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'overlap_pixels': 0, 'duplicate_ids': 0, 'LF': True}
