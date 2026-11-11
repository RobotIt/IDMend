"""Command-line inference, evaluation and submission packaging."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

MTRAIN = ('02', '05', '09', '11')
MTEST = ('01', '06', '07', '12')
KVALID = ('0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def discover_images(root, dataset, split, sequence):
    sequence_name = 'MOTS20-' + sequence if dataset == 'mots20' else sequence
    if dataset == 'mots20':
        part = 'test' if sequence in MTEST else 'train'
        candidates = [root / part / sequence_name / 'img1', root / sequence_name / 'img1',
                      root / part / 'sequences' / sequence_name / 'img1']
    else:
        part = 'testing' if split == 'test' else 'training'
        candidates = [root / part / 'image_02' / sequence, root / 'image_02' / sequence, root / sequence]
    for directory in candidates:
        if directory.is_dir():
            paths = sorted((p for p in directory.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                           key=lambda p: int(p.stem))
            if paths:
                numbers = [int(p.stem) for p in paths]
                first = 1 if dataset == 'mots20' else 0
                if numbers != list(range(first, first + len(paths))):
                    raise ValueError(f'Incomplete or incorrectly numbered image sequence: {directory}')
                return paths
    raise FileNotFoundError('No sequence images; checked: ' + ', '.join(str(p) for p in candidates))


def native_source(folder, dataset, sequence, cls):
    name = 'MOTS20-' + sequence if dataset == 'mots20' else sequence
    options = [folder / sequence / 'native' / f'class{cls}.txt', folder / sequence / f'class{cls}.txt',
               folder / sequence / 'detections' / f'{name}.txt',
               folder / name / 'detections' / f'{name}.txt', folder / f'{name}.txt']
    for path in options:
        if path.is_file():
            if not path.with_suffix('.scores.npz').is_file():
                raise FileNotFoundError(f'Native scores must accompany {path}')
            return path
    raise FileNotFoundError(f'Native class{cls} not found for {sequence} under {folder}')


def feature_source(folder, sequence, cls):
    if folder is None:
        return None
    options = [folder / sequence / f'class{cls}.npz', folder / sequence / f'features_class{cls}.npz',
               folder / f'class{cls}' / f'{sequence}.npz', folder / sequence / 'osnet.npz',
               folder / f'{sequence}.npz']
    for path in options:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Explicit feature cache is missing for {sequence}/class{cls}')


def normalise_sequence(dataset, value):
    return str(int(value.replace('MOTS20-', ''))).zfill(2 if dataset == 'mots20' else 4)


def package(output, dataset, sequences, partial=False, test_split=False):
    files = [(('MOTS20-' + seq if dataset == 'mots20' else seq) + '.txt') for seq in sequences]
    label = 'SMOKE_PARTIAL' if partial else ('Test_Paper_Final' if test_split else 'TrainEval_Paper_Final')
    target = output / (f'{dataset.upper()}_{label}_{len(files)}seq.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as archive:
        for name in files:
            source = output / 'submission' / name
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if archive.namelist() != files or archive.testzip() is not None:
            raise ValueError('ZIP integrity verification failed')
        for name in files:
            if archive.read(name) != (output / 'submission' / name).read_bytes():
                raise ValueError(f'ZIP byte mismatch: {name}')
        unpacked = sum(info.file_size for info in archive.infolist())
        if unpacked > 500 * 1024 ** 2:
            raise ValueError('Submission exceeds 500 MiB uncompressed')
    return {'path': str(target), 'sha256': sha(target), 'bytes': target.stat().st_size,
            'uncompressed_bytes': unpacked, 'files': files, 'CRC_verified': True,
            'partial_input': partial, 'full_test_submission': test_split and not partial and len(files) == (4 if dataset == 'mots20' else 29)}


def doctor(root):
    import cv2, numpy, scipy, torch, torchvision, yaml
    import ultralytics, torchreid, trackeval
    from . import backend, masks
    import trackrefine.association as association
    modules = {m.__name__: str(Path(m.__file__).resolve())
               for m in (ultralytics, torchreid, trackeval, backend, masks, association)}
    for name in ('ultralytics', 'torchreid', 'peopletrack_final.backend', 'peopletrack_final.masks',
                 'trackrefine.association', 'trackeval'):
        if not Path(modules[name]).is_relative_to(root):
            raise RuntimeError(f'Module escaped the final package: {name}: {modules[name]}')
    config = yaml.safe_load((root / 'configs/final.yaml').read_text(encoding='utf-8'))
    weights = [root / config['detector']['weights']]
    for dataset in config['datasets'].values():
        for entry in dataset['classes'].values():
            weights.extend(root / entry[k] for k in ('osnet', 'resnet', 'ranker'))
        weights.extend(root / p for p in dataset.get('heldout', {}).values())
    missing = [str(q) for q in weights if not q.is_file()]
    if missing:
        raise FileNotFoundError('Missing bundled weights: ' + ', '.join(missing))
    for path in set(weights):
        if path.suffix in ('.pt', '.pth'):
            with path.open('rb') as stream:
                if stream.read(42).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise RuntimeError(f'Checkpoint not downloaded: {path}. Run git lfs pull.')
    print(json.dumps({'python': sys.executable, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'opencv': cv2.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__,
        'ultralytics': ultralytics.__version__, 'torchreid': getattr(torchreid, '__version__', '0.2.5'),
        'local_modules': modules, 'bundled_weights_present': True,
        'paper_windows': {ds: {cl: v['window'] for cl, v in item['classes'].items()} for ds, item in config['datasets'].items()}}, indent=2))


def main(root):
    parser = argparse.ArgumentParser(description='Final YOLO26-seg + BoT-SORT + split/ResNetLink + mask refinement')
    parser.add_argument('--dataset', choices=('mots20', 'kitti'))
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    parser.add_argument('--data-root', type=Path, help='Dataset root (defaults to the path in the configuration)')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=root / 'configs/final.yaml')
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--window-ped', type=int)
    parser.add_argument('--window-car', type=int)
    parser.add_argument('--max-frames', type=int, help='Short execution check; partial output is NOT a website submission')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate train/validation predictions with bundled TrackEval')
    parser.add_argument('--gt-root', type=Path, help='Training annotation root; KITTI uses a folder of 0000.txt files')
    parser.add_argument('--native-root', type=Path, help='Import initial tracks instead of rerunning YOLO/BoT-SORT')
    parser.add_argument('--raw-root', type=Path, help='KITTI raw JSONL files; required with --native-root')
    parser.add_argument('--feature-root', type=Path, help='Explicit aligned float32 feature cache (reproduction only)')
    parser.add_argument('--check', action='store_true', help='Check imports and installed runtime, then exit')
    args = parser.parse_args()
    if args.check:
        doctor(root)
        return
    if args.dataset is None:
        parser.error('--dataset is required unless --check is used')
    if args.evaluate and args.split == 'test':
        parser.error('Official test GT is unavailable; --evaluate supports train/validation only')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if any(value is not None and value < 1 for value in (args.window_ped, args.window_car)):
        parser.error('Boundary windows must be positive')
    if args.native_root and args.dataset == 'kitti' and args.raw_root is None:
        parser.error('KITTI --native-root also requires --raw-root for background camera compensation')
    import cv2, numpy as np, torch, yaml
    from .backend import AssociationBackend
    from .camera import compute as compute_camera
    from .frontend import export_sequence
    from .masks import merge_classes, refine_sequence, validate_submission_file
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    dataset = args.dataset
    data_root = (args.data_root or root / config['data_roots'][dataset]).resolve()
    output = (args.output or root / 'results' / f'{dataset}_{args.split}').resolve()
    defaults = ((MTEST if args.split == 'test' else MTRAIN) if dataset == 'mots20'
                else tuple(f'{i:04d}' for i in range(29 if args.split == 'test' else 21)))
    if dataset == 'kitti' and args.split == 'validation':
        defaults = KVALID
    sequences = [normalise_sequence(dataset, item) for item in (args.sequences or defaults)]
    if len(sequences) != len(set(sequences)):
        parser.error('Duplicate sequence IDs')
    if dataset == 'mots20' and any(seq not in (MTEST if args.split == 'test' else MTRAIN) for seq in sequences):
        parser.error('MOTS20 sequence IDs do not match the selected split')
    inventory = {seq: discover_images(data_root, dataset, args.split, seq) for seq in sequences}
    if args.max_frames is not None:
        inventory = {seq: paths[:args.max_frames] for seq, paths in inventory.items()}
    classes = deepcopy(config['datasets'][dataset]['classes'])
    for key, settings in classes.items():
        for field in ('osnet', 'resnet', 'ranker'):
            settings[field] = str((root / settings[field]).resolve(strict=True))
        override = args.window_ped if int(key) == 2 else args.window_car
        if override is not None:
            settings['window'] = override
    heldout = config['datasets'][dataset].get('heldout', {})
    heldout = {seq: str((root / path).resolve(strict=True)) for seq, path in heldout.items()}
    detector = (root / config['detector']['weights']).resolve(strict=True)
    tracker_yaml = (root / config['detector']['tracker']).resolve(strict=True)
    image_inventory = {seq: hashlib.sha256(json.dumps(
        [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths], separators=(',', ':')).encode()).hexdigest()
        for seq, paths in inventory.items()}
    protocol = {'dataset': dataset, 'split': args.split, 'sequences': sequences,
        'data_root': str(data_root), 'max_frames': args.max_frames,
        'partial_input': args.max_frames is not None, 'mask_position': 'post_association',
        'images': {seq: len(paths) for seq, paths in inventory.items()}, 'image_inventory_sha256': image_inventory,
        'class_settings': classes, 'reid': config['reid'], 'tracking': config['tracking'],
        'native_root': str(args.native_root.resolve()) if args.native_root else None,
        'raw_root': str(args.raw_root.resolve()) if args.raw_root else None,
        'feature_root': str(args.feature_root.resolve()) if args.feature_root else None,
        'models_sha256': {path: sha(path) for path in sorted({str(detector), str(tracker_yaml),
            *heldout.values(), *(v[k] for v in classes.values() for k in ('osnet', 'resnet', 'ranker'))})},
        'implementation_sha256': {str(p.relative_to(root)): sha(p)
            for folder in ('peopletrack_final', 'trackrefine') for p in sorted((root / folder).glob('*.py'))},
        'official_test_score': None, 'GT_used': False, 'website_uploaded': False}
    mark = output / 'RUN_PROTOCOL.json'
    if mark.exists() and json.loads(mark.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('This output directory belongs to a different input/configuration. Choose another --output.')
    dump(mark, protocol)
    backends = {}
    reports = {}
    for seq, paths in inventory.items():
        print(f'SEQUENCE {dataset}/{seq}: {len(paths)} frames', flush=True)
        work = output / 'work' / seq
        completed = work / 'COMPLETE.json'
        name = ('MOTS20-' + seq if dataset == 'mots20' else seq) + '.txt'
        submission = output / 'submission' / name
        if completed.exists():
            prior = json.loads(completed.read_text(encoding='utf-8'))
            if submission.exists() and sha(submission) == prior['validation']['sha256']:
                reports[seq] = prior
                print('REUSE completed final result', flush=True)
                continue
            raise RuntimeError(f'Completed output was modified: {submission}')
        native_dir, raw = work / 'native', work / 'raw.jsonl'
        if args.native_root:
            native_dir.mkdir(parents=True, exist_ok=True)
            for cls in classes:
                source = native_source(args.native_root.resolve(), dataset, seq, int(cls))
                target = native_dir / f'class{cls}.txt'
                shutil.copyfile(source, target)
                shutil.copyfile(source.with_suffix('.scores.npz'), target.with_suffix('.scores.npz'))
            if args.raw_root:
                srcraw = args.raw_root / f'{seq}.jsonl'
                if not srcraw.is_file():
                    raise FileNotFoundError(srcraw)
                shutil.copyfile(srcraw, raw)
            front = {'imported_native': True, 'root': str(args.native_root)}
        else:
            front = export_sequence(paths, native_dir, raw, detector, tracker_yaml, dataset, args.device)
        camera = None
        if any(v['camera_mode'] == 'background_lk' for v in classes.values()):
            camera = compute_camera(paths, raw, work / 'camera.npz')
        linked = work / 'linked'
        class_reports = {}
        for key, settings in classes.items():
            deployed = deepcopy(settings)
            if dataset == 'mots20' and seq in heldout:
                deployed['resnet'] = heldout[seq]
            signature = json.dumps(deployed, sort_keys=True)
            if signature not in backends:
                backends[signature] = AssociationBackend(config['tracking'], config['reid'], deployed, args.device)
            class_reports[key] = backends[signature].process(
                native_dir / f'class{key}.txt', paths, linked / f'class{key}.txt',
                work / 'features' / f'class{key}.npz', camera,
                feature_source(args.feature_root, seq, int(key)))
        merged = work / 'merged.txt'
        if dataset == 'mots20':
            # Historical MOTS20 preserves linked IDs; guard1 RNG also uses ID.
            shutil.copyfile(linked / 'class2.txt', merged)
        else:
            merge_classes(linked, merged, dataset)
        ref = refine_sequence(paths, merged, submission,
            {int(k): v['mask_refinement'] for k, v in classes.items()}, args.workers)
        validation = validate_submission_file(submission, paths, dataset)
        record = {'sequence': seq, 'frontend': front, 'classes': class_reports,
            'refinement': ref, 'validation': validation,
            'native_imported': args.native_root is not None,
            'deployment': {k: {'window': v['window'], 'resnet': heldout.get(seq, v['resnet'])
                if dataset == 'mots20' else v['resnet']} for k, v in classes.items()}}
        dump(completed, record)
        reports[seq] = record
        print(f'COMPLETE {seq}: {validation["rows"]} rows, non-overlapping', flush=True)
    archive = package(output, dataset, sequences, partial=args.max_frames is not None, test_split=args.split == 'test')
    evaluation = None
    if args.evaluate:
        from .evaluation import evaluate_run
        evaluation = evaluate_run(root, output, args.gt_root, config=config)
    dump(output / 'RESULTS.json', {'protocol': protocol, 'sequences': reports, 'package': archive,
        'evaluation': evaluation,
        'note': 'Local training-set evaluation only when requested; official test scores require the benchmark server.'})
    print('SUBMISSION ' + str(archive['path']), flush=True)
