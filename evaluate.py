"""Evaluate a completed train/validation run without rerunning inference."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'vendor'), str(ROOT)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gt-root', type=Path)
    parser.add_argument('--config', type=Path, help='Use the ground-truth paths from this configuration')
    args = parser.parse_args()
    from peopletrack_final.evaluation import evaluate_run
    config = None
    if args.config:
        import yaml
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    evaluate_run(ROOT, args.run_dir, args.gt_root, config=config)
