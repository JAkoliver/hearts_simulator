"""Exploiter-league round-1 defense-gate analysis (prereg + 2026-08-07
amendment): CRN-paired moons-conceded-per-match, candidate defenders vs
baseline defenders against the frozen SEL probe.

PRIMARY (the gate): paired delta of moons conceded per match, one-sided
alpha=0.05, fewer = pass. n = 64 paired seed-matches.
SECONDARY (reported, gates nothing): defender-side placement vs the
probe field, per arm.

Usage: python analyze_defense_gate.py [--tag r1t1]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import pandas as pd


def load_arm(out_dir, arm):
    frames = []
    for p in sorted(glob.glob(os.path.join(out_dir, f'{arm}_*.csv'))):
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise SystemExit(f'no CSVs for arm {arm} in {out_dir}')
    return pd.concat(frames, ignore_index=True)


def per_match(df):
    """(seed, match) -> dict(moons, shooter_seat placement of defenders)."""
    out = {}
    for (seed, match), g in df.groupby(['seed', 'match']):
        moons = int(g['moon_success'].sum())
        last = g.iloc[-1]
        totals = [last[f't{i}'] for i in range(4)]
        shooter = int(last['seat'])
        # defender placements: rank all four totals (low = better),
        # average the three defender seats' placements
        order = sorted(range(4), key=lambda s: totals[s])
        place = {s: i + 1 for i, s in enumerate(order)}
        dplace = [place[s] for s in range(4) if s != shooter]
        out[(seed, match)] = {'moons': moons,
                              'def_place_mean': sum(dplace) / 3.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r1t1')
    args = ap.parse_args()
    out_dir = f'equity_data/exploiter_r1/gate_{args.tag}'

    base = per_match(load_arm(out_dir, 'base'))
    cand = per_match(load_arm(out_dir, 'cand'))
    keys = sorted(set(base) & set(cand))
    if len(keys) != len(base) or len(keys) != len(cand):
        print(f'WARNING: unpaired matches dropped '
              f'(base {len(base)}, cand {len(cand)}, paired {len(keys)})')
    n = len(keys)
    deltas = [cand[k]['moons'] - base[k]['moons'] for k in keys]
    mean_b = sum(base[k]['moons'] for k in keys) / n
    mean_c = sum(cand[k]['moons'] for k in keys) / n
    md = sum(deltas) / n
    var = sum((d - md) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n)
    t = md / se if se > 0 else 0.0
    # one-sided p for improvement (delta < 0), normal approx (n=64)
    p = 0.5 * math.erfc(-t / math.sqrt(2)) if se > 0 else 1.0
    ci95 = (md - 1.6449 * se, md + 1.6449 * se)
    gate_pass = (md < 0) and (p < 0.05)

    print(f'n paired = {n}')
    print(f'moons conceded/match: baseline {mean_b:.3f} | candidate {mean_c:.3f}')
    print(f'paired delta {md:+.3f} (se {se:.3f}, one-sided 90% CI '
          f'[{ci95[0]:+.3f}, {ci95[1]:+.3f}]), t={t:.2f}, one-sided p={p:.4f}')
    dp_b = sum(base[k]['def_place_mean'] for k in keys) / n
    dp_c = sum(cand[k]['def_place_mean'] for k in keys) / n
    print(f'secondary - defender mean placement vs probe field: '
          f'baseline {dp_b:.3f} | candidate {dp_c:.3f} (reported, gates nothing)')
    print(f'DEFENSE GATE: {"PASS" if gate_pass else "HALT"} '
          f'(one-sided alpha=0.05, fewer = pass)')

    os.makedirs('equity_data/verdicts', exist_ok=True)
    json.dump({'tag': args.tag, 'n_paired': n,
               'base_moons_per_match': mean_b, 'cand_moons_per_match': mean_c,
               'paired_delta': md, 'se': se, 't': t, 'p_one_sided': p,
               'def_place_base': dp_b, 'def_place_cand': dp_c,
               'pass': bool(gate_pass)},
              open(f'equity_data/verdicts/exploiter_r1_defense_gate_{args.tag}.json',
                   'w'), indent=1)
    return 0 if gate_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
