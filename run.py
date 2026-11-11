"""One entry point for the final split, ResNetLink and mask pipeline."""
from __future__ import annotations
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))
RUNTIME = ROOT / '.runtime' / 'ultralytics'
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(RUNTIME))

if __name__ == '__main__':
    from peopletrack_final.cli import main
    main(ROOT)
