"""Independent verification of built v2 mix banks against their manifests
(auto-pipeline step-1 check; also usable standalone).

For every *.manifest.json in --mix-dir: re-hash the bank and index files
against the recorded sha256, re-read the bank and independently re-count
records and confident records (frozen 2-sigma rule), and compare against
the manifest's composition. Exit 0 only if every mix verifies.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill import MATCH_RECORD_V2, V2_MAGIC  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix-dir', default='expert_data/mixes')
    ap.add_argument('--expect', type=int, default=8,
                    help='number of manifests expected')
    args = ap.parse_args()

    manifests = sorted(glob.glob(os.path.join(args.mix_dir, '*.manifest.json')))
    ok = True
    if len(manifests) < args.expect:
        print(f"FAIL: {len(manifests)} manifests found, expected {args.expect}")
        ok = False
    for mpath in manifests:
        m = json.load(open(mpath))
        mix = m['mix']
        errs = []
        for key, path in (('bank_sha256', m['bank']), ('index_sha256', m['index'])):
            if not os.path.exists(path):
                errs.append(f'missing {path}')
            elif sha256(path) != m[key]:
                errs.append(f'{key} mismatch')
        if not errs:
            with open(m['bank'], 'rb') as fh:
                if fh.read(4) != V2_MAGIC:
                    errs.append('bad magic')
            n = (os.path.getsize(m['bank']) - 32) // MATCH_RECORD_V2.itemsize
            if n != m['records_written']:
                errs.append(f'record count {n} != manifest {m["records_written"]}')
            a = np.fromfile(m['bank'], dtype=MATCH_RECORD_V2, offset=32, count=n)
            play = (a['flags'] & 1) == 0
            gap = a['eq_best'].astype(np.float32) - a['eq_second']
            conf = int((play & (a['second_action'] != 0xFFFF)
                        & (gap > 2.0 * a['gap_se'])).sum())
            want_conf = sum(m['composition_confident'].values())
            if conf != want_conf:
                errs.append(f'confident recount {conf} != composition {want_conf}')
        status = 'OK' if not errs else 'FAIL: ' + '; '.join(errs)
        print(f"{mix}: {status}")
        ok = ok and not errs
    print('BANKS ' + ('VERIFIED' if ok else 'VERIFICATION FAILED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
