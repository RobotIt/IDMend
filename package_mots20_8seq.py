"""Combine completed full train/test predictions into the historical eight-file ZIP."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=Path, default=ROOT / 'results/mots20_train')
    parser.add_argument('--test-dir', type=Path, default=ROOT / 'results/mots20_test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/MOTS20_Paper_Final_8seq.zip')
    args = parser.parse_args()
    files = {}
    for part, directory, seqs in [('train', args.train_dir, ('02', '05', '09', '11')),
                                  ('test', args.test_dir, ('01', '06', '07', '12'))]:
        report = json.loads((directory / 'RESULTS.json').read_text(encoding='utf-8'))
        protocol = report['protocol']
        assert protocol['dataset'] == 'mots20' and protocol['split'] == part
        assert not protocol['partial_input'] and set(protocol['sequences']) == set(seqs)
        assert protocol['class_settings']['2']['window'] == 2
        for seq in seqs:
            name = 'MOTS20-' + seq + '.txt'
            path = directory / 'submission' / name
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == report['sequences'][seq]['validation']['sha256']
            assert report['sequences'][seq]['validation']['overlap_pixels'] == 0
            files[name] = raw
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, allowZip64=False) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None and z.namelist() == sorted(files)
        assert all(z.read(name) == raw for name, raw in files.items())
    print(str(args.output.resolve()))
