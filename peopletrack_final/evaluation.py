"""Local mask-based TrackEval evaluation; no hidden-test ground truth is read."""
from pathlib import Path
import csv
import json


def evaluate_run(root: Path, output: Path, gt_root: Path | None = None, *, config=None):
    import numpy as np
    import yaml
    import trackeval
    from .cli import dump, sha

    root, output = Path(root).resolve(), Path(output).resolve()
    if not Path(trackeval.__file__).resolve().is_relative_to(root / 'vendor'):
        raise RuntimeError('TrackEval must be loaded from this release')
    protocol = json.loads((output / 'RUN_PROTOCOL.json').read_text(encoding='utf-8'))
    dataset, split = protocol['dataset'], protocol['split']
    if split not in ('train', 'validation'):
        raise ValueError('Local evaluation is restricted to train/validation')
    if gt_root is None:
        if config is None:
            config = yaml.safe_load((root / 'configs/final.yaml').read_text(encoding='utf-8'))
        configured = config.get('ground_truth_roots', {}).get(dataset)
        gt_root = (root / configured if configured else
                   Path(protocol['data_root']) / ('train' if dataset == 'mots20' else 'instances_txt'))
    gt_root = Path(gt_root).resolve(strict=True)
    evaluation = output / 'evaluation'
    sequence_lengths, source_hashes = {}, {}
    shift = -1 if dataset == 'mots20' else 0

    def write_frame_subset(source, target, count):
        target.parent.mkdir(parents=True, exist_ok=True)
        first = 1 if dataset == 'mots20' else 0
        with source.open(encoding='utf-8-sig') as inp, target.open('w', encoding='ascii', newline='\n') as out:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                frame, rest = line.split(' ', 1)
                frame = int(frame)
                if first <= frame < first + count:
                    out.write(f'{frame + shift} {rest}\n')

    for seq in protocol['sequences']:
        name = 'MOTS20-' + seq if dataset == 'mots20' else seq
        count = int(protocol['images'][seq])
        sequence_lengths[name] = count
        source = (gt_root / name / 'gt/gt.txt' if dataset == 'mots20' else gt_root / (seq + '.txt'))
        prediction = output / 'submission' / (name + '.txt')
        if not source.is_file() or not prediction.is_file():
            raise FileNotFoundError(f'Missing annotation or prediction: {source}; {prediction}')
        source_hashes[str(source)] = sha(source)
        write_frame_subset(source, evaluation / 'gt' / name / 'gt.txt', count)
        write_frame_subset(prediction, evaluation / 'trackers/paper/data' / (name + '.txt'), count)

    data = trackeval.datasets.KittiMOTS({
        'GT_FOLDER': str(evaluation / 'gt'),
        'TRACKERS_FOLDER': str(evaluation / 'trackers'),
        'TRACKERS_TO_EVAL': ['paper'],
        'CLASSES_TO_EVAL': ['pedestrian'] if dataset == 'mots20' else ['car', 'pedestrian'],
        'SEQ_INFO': sequence_lengths, 'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt.txt',
        'PRINT_CONFIG': False,
    })
    evaluator = trackeval.Evaluator({
        'USE_PARALLEL': False, 'PRINT_CONFIG': False, 'PRINT_RESULTS': False,
        'TIME_PROGRESS': False, 'PLOT_CURVES': False, 'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
    })
    result, status = evaluator.evaluate([data], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    if status.get('KittiMOTS', {}).get('paper') != 'Success':
        raise RuntimeError(f'TrackEval did not succeed: {status}')

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    rows = []
    count_metrics = ('IDSW', 'CLR_TP', 'CLR_FP', 'CLR_FN', 'Frag')
    score_metrics = ('HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'LocA', 'IDF1', 'MOTA')
    for cls, metrics in result['KittiMOTS']['paper']['COMBINED_SEQ'].items():
        if cls not in ('car', 'pedestrian'):
            continue
        row = {'dataset': dataset, 'split': split, 'class': cls,
               'sequences': len(sequence_lengths), 'frames': sum(sequence_lengths.values())}
        for family in metrics.values():
            for key, value in family.items():
                if key in count_metrics + score_metrics:
                    row[key] = float(np.mean(value)) * (100 if key in score_metrics else 1)
        rows.append(row)
    dump(evaluation / 'FULL_RESULTS.json', plain(result))
    dump(evaluation / 'STATUS.json', status)
    dump(evaluation / 'METRICS.json', rows)
    with (evaluation / 'METRICS.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {'dataset': dataset, 'split': split, 'partial_input': protocol.get('partial_input', False),
              'official_test_score': False, 'GT_used_for_evaluation_only': True,
              'annotation_sha256': source_hashes, 'trackeval_file': str(Path(trackeval.__file__).resolve()),
              'metrics': rows,
              'scope': 'Historical development/training data; includes past model/parameter selection. Full KITTI train includes fitting/calibration sequences.'}
    dump(evaluation / 'EVALUATION_PROTOCOL.json', record)
    print('LOCAL METRICS ' + json.dumps(rows, ensure_ascii=False), flush=True)
    return record
