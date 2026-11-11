"""Execute fresh images through the complete pipeline and local evaluation."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mots20-root', type=Path)
    parser.add_argument('--mots20-gt-root', type=Path)
    parser.add_argument('--kitti-root', type=Path)
    parser.add_argument('--kitti-gt-root', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    check = (args.output or ROOT / 'checks/smoke_runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    check.mkdir(parents=True, exist_ok=True)
    runs = []
    for dataset, seq in [('mots20', '02'), ('kitti', '0014')]:
        out = check / dataset
        log = check / (dataset + '.log')
        command = [sys.executable, '-I', '-B', str(ROOT / 'run.py'), '--dataset', dataset,
                   '--split', 'train', '--sequences', seq, '--max-frames', str(args.frames),
                   '--device', args.device, '--workers', '1', '--evaluate', '--output', str(out)]
        for option, path in (('--data-root', getattr(args, dataset + '_root')),
                             ('--gt-root', getattr(args, dataset + '_gt_root'))):
            if path is not None:
                command.extend([option, str(path)])
        print('START ' + dataset + '/' + seq, flush=True)
        started = time.perf_counter()
        with log.open('w', encoding='utf-8') as stream:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if status.returncode:
            print(log.read_text(encoding='utf-8', errors='replace')[-9000:], flush=True)
            raise SystemExit(status.returncode)
        result = json.loads((out / 'RESULTS.json').read_text(encoding='utf-8'))
        assert result['protocol']['max_frames'] == args.frames
        assert all(v['window'] == 2 for v in result['protocol']['class_settings'].values())
        assert result['package']['CRC_verified'] and not result['package']['full_test_submission']
        assert all(not v['native_imported'] for v in result['sequences'].values())
        assert result['evaluation']['partial_input'] and not result['evaluation']['official_test_score']
        assert all(v['validation']['overlap_pixels'] == 0 for v in result['sequences'].values())
        row = {'dataset': dataset, 'sequence': seq, 'frames': args.frames,
               'seconds': time.perf_counter() - started, 'fresh_images': True,
               'native_cache_imported': False, 'feature_cache_imported': False,
               'isolated_import': True, 'package_crc': True,
               'local_metrics': result['evaluation']['metrics'], 'log': str(log)}
        runs.append(row)
        print('PASS ' + dataset + ' ' + json.dumps(row['local_metrics']), flush=True)
    (check / 'VERIFICATION.json').write_text(json.dumps({'status': 'passed', 'runs': runs,
        'scope': 'Short fresh-image runs, not full hidden-test reproduction or benchmark scores.'}, indent=2), encoding='utf-8')
    print('ALL FRESH-IMAGE CHECKS PASSED', flush=True)
