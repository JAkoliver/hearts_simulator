"""Phase 2 recorder validation (docs/phase2_visitcount_prereg.md
Stage A) + teacher-signal preview (Stage B quantities, informational
here; Stage B proper runs its registered bands on a dedicated probe).

Record: f32 obs[556] | u8 mask[52] | f32 pi[52] | i32 action |
i32 visits | u8 seat | u8 kind | u16 match | u8 deal | u8 pad |
i16 final[4] | f32 place[4]  (2522 bytes)

Usage: python validate_p2_records.py <file.hvt> [--iters N]
"""
import argparse
import sys

import numpy as np

DT = np.dtype([('obs', '<f4', 556), ('mask', 'u1', 52), ('pi', '<f4', 52),
               ('action', '<i4'), ('visits', '<i4'), ('seat', 'u1'),
               ('kind', 'u1'), ('match', '<u2'), ('deal', 'u1'),
               ('pad', 'u1'), ('final', '<i2', 4), ('place', '<f4', 4)])
assert DT.itemsize == 2522


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--iters', type=int, default=64)
    args = ap.parse_args()

    r = np.fromfile(args.path, dtype=DT)
    bad = []
    plays = r[r['kind'] == 0]
    trail = r[r['kind'] == 1]
    n = len(plays)
    if n == 0:
        sys.exit('no play records')

    idx = np.arange(n)
    in_mask = plays['mask'][idx, plays['action']] == 1
    if not in_mask.all():
        bad.append(f'{(~in_mask).sum()} actions outside mask')
    out_pi = (plays['pi'] * (plays['mask'] == 0)).sum(1)
    if (out_pi > 1e-4).any():
        bad.append(f'{(out_pi > 1e-4).sum()} records with pi mass off-mask')
    forced = plays['mask'].sum(1) == 1
    nf = plays[~forced]
    if len(nf):
        s = nf['pi'].sum(1)
        if (np.abs(s - 1.0) > 1e-3).any():
            bad.append(f'{(np.abs(s - 1) > 1e-3).sum()} pi not normalized')
        top = nf['pi'].max(1)
        agrees = nf['pi'][np.arange(len(nf)), nf['action']] >= top - 1e-6
        if not agrees.all():
            bad.append(f'{(~agrees).sum()} action != max-visit move')
        v = nf['visits']
        if ((v < args.iters) | (v > args.iters + 64)).any():
            bad.append(f'visits outside [{args.iters}, {args.iters}+64]: '
                       f'{((v < args.iters) | (v > args.iters + 64)).sum()}')
    if forced.any():
        fv = plays[forced]['visits']
        if (fv != 0).any():
            bad.append(f'{(fv != 0).sum()} forced records with visits != 0')
    for t in trail:
        if not (t['place'].sum() == 10.0 and t['action'] == -1):
            bad.append(f'bad trailer in match {t["match"]}')
            break

    # teacher-signal preview on non-forced plays
    if len(nf):
        top = nf['pi'].max(1)
        onehot = (top >= 0.90).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -np.where(nf['pi'] > 0, nf['pi'] * np.log(nf['pi']), 0).sum(1)
        print(f'plays {n} ({forced.sum()} forced) | trailers {len(trail)} | '
              f'top1 share median {np.median(top):.3f} | '
              f'>=0.90-one-hot {onehot * 100:.1f}% | '
              f'entropy median {np.median(ent):.3f}')
    if bad:
        print('SELF-CONSISTENCY FAIL:')
        for b in bad:
            print('  -', b)
        sys.exit(1)
    print('SELF-CONSISTENCY PASS')


if __name__ == '__main__':
    main()
