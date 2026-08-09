"""Round-2 Phase B2 pre-probe analysis: ORDERING ONLY (non-binding).

Reads equity_data/exploiter_r2/preprobe/<arm>_*.csv (defense-gate CSV
surface), computes per-arm moons conceded per match on the CRN-paired
16-seed set, and ranks candidate arms by paired delta vs the base arm.
Ordering picks the best <=2 drift-screen passers for full gating; no
claim is made or implied at n=16 (prereg: "non-binding, ordering only").
"""
import glob
import json
import os

import pandas as pd

OUT_DIR = 'equity_data/exploiter_r2/preprobe'


def per_match_moons(arm):
    frames = [pd.read_csv(p) for p in
              sorted(glob.glob(os.path.join(OUT_DIR, f'{arm}_*.csv')))]
    if not frames:
        raise SystemExit(f'no CSVs for arm {arm}')
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(['seed', 'match'])['moon_success'].sum()
    return g


def main():
    arms = sorted({os.path.basename(p).rsplit('_', 1)[0]
                   for p in glob.glob(os.path.join(OUT_DIR, '*.csv'))
                   if not p.endswith('.tricks.csv')})
    if 'base' not in arms:
        raise SystemExit('base arm missing')
    base = per_match_moons('base')
    print(f'base: {base.sum()} moons / {len(base)} matches '
          f'({base.mean():.3f}/match)')
    rows = []
    for arm in arms:
        if arm == 'base':
            continue
        cand = per_match_moons(arm)
        joined = pd.concat([base, cand], axis=1, keys=['b', 'c']).dropna()
        if len(joined) != len(base):
            print(f'WARNING: {arm} pairs {len(joined)}/{len(base)}')
        delta = (joined['c'] - joined['b'])
        rows.append({'arm': arm, 'moons_per_match': round(cand.mean(), 3),
                     'paired_delta': round(delta.mean(), 3),
                     'n_pairs': len(joined)})
        print(f'{arm}: {cand.sum()} moons ({cand.mean():.3f}/match)  '
              f'paired delta {delta.mean():+.3f}')
    rows.sort(key=lambda r: r['paired_delta'])
    print('\nORDERING (best first, non-binding):')
    for i, r in enumerate(rows, 1):
        print(f'  {i}. {r["arm"]}  delta {r["paired_delta"]:+.3f}')
    with open(os.path.join(OUT_DIR, 'ordering.json'), 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
