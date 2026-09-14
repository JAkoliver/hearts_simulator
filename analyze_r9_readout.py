"""League r9 per-trial readout -> eligibility (docs/exploiter_league_r9_prereg.md §4/§5).

Reads, for trial <name> under equity_data/exploiter_r9/<name>/:
  vecprobe_shooter_sel_v2.json / vecprobe_shooter_sel_v1.json  (E1)
  strength.log  (neutral_raw_eval output; E2)
  transfer/<name>/ chunk CSVs paired against transfer/promoted_base/ (E3)
and prints the registered verdicts:
  E1: end vec-probe delta vs promoted UB95 < 0 on sel v2 AND point delta < 0 on sel v1
  E2: neutral-raw delta UB95 <= +0.05/deal
  E3: transfer n=128 paired delta vs promoted <= -0.15
  fail-fast (informs the driver): mid sel-v2 delta >= +0.10 with LB > 0
Writes equity_data/verdicts/r9_<name>_readout.json. Never promotes.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = 'equity_data/exploiter_r9'


def per_match_moons(df):
    out = {}
    for (seed, match), g in df.groupby(['seed', 'match']):
        last = g.iloc[-1]
        totals = np.array([last[f't{i}'] for i in range(4)], dtype=float)
        shooter = int(last['seat'])
        order = np.argsort(np.argsort(totals)) + 1
        defs = [s for s in range(4) if s != shooter]
        out[(int(seed), int(match))] = (int(g['moon_success'].sum()),
                                        float(np.mean(order[defs])))
    return out


def transfer(name):
    cand = pd.concat([pd.read_csv(p) for p in sorted([p for p in glob.glob(f'{ROOT}/transfer/{name}/s*m*.csv') if not p.endswith('.tricks.csv')])],
                     ignore_index=True)
    base = pd.concat([pd.read_csv(p) for p in sorted([p for p in glob.glob(f'{ROOT}/transfer/promoted_base/s*m*.csv') if not p.endswith('.tricks.csv')])],
                     ignore_index=True)
    c, b = per_match_moons(cand), per_match_moons(base)
    keys = sorted(set(c) & set(b))
    d = np.array([c[k][0] - b[k][0] for k in keys], dtype=float)
    n = len(d)
    mean = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n))
    p = float(stats.t.cdf(mean / se, n - 1)) if se > 0 else 0.0
    return {'n': n, 'delta': mean, 'se': se, 'p_one_sided': p,
            'cand_moons': float(np.mean([c[k][0] for k in keys])),
            'base_moons': float(np.mean([b[k][0] for k in keys])),
            'cand_def_place': float(np.mean([c[k][1] for k in keys])),
            'base_def_place': float(np.mean([b[k][1] for k in keys]))}


def strength(name):
    txt = open(f'{ROOT}/{name}/strength.log', encoding='utf-8', errors='replace').read()
    m = re.search(r'Neutral raw delta.*?:\s*([+-]?\d+\.\d+)\s*\(SE\s*(\d+\.\d+),\s*n=(\d+)\)', txt)
    if not m:
        return None
    mean, se, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
    return {'delta': mean, 'se': se, 'n': n, 'ub95': mean + 1.645 * se}


def vecprobe(name, clone):
    p = f'{ROOT}/{name}/vecprobe_{clone}.json'
    if not os.path.exists(p):
        return None
    r = json.load(open(p))['results']
    end = [v for k, v in r.items() if k.endswith('_end_ensemble.pth')][0]
    mid = [v for k, v in r.items() if k.endswith('_mid_ensemble.pth')][0]
    return {'end': end, 'mid': mid}


if __name__ == '__main__':
    name = sys.argv[1]
    v2, v1 = vecprobe(name, 'shooter_sel_v2'), vecprobe(name, 'shooter_sel_v1')
    st = strength(name)
    tr = transfer(name) if [p for p in glob.glob(f'{ROOT}/transfer/{name}/s*m*.csv') if not p.endswith('.tricks.csv')] else None
    out = {'trial': name, 'vecprobe_sel_v2': v2, 'vecprobe_sel_v1': v1,
           'strength': st, 'transfer': tr}
    print(f'== r9 readout {name}')
    ff = None
    if v2:
        m = v2['mid']
        ff = bool(m['delta'] >= 0.10 and (m['delta'] - 1.645 * m['se']) > 0)
        print(f"vec probe sel v2: mid {m['delta']:+.3f} (SE {m['se']:.3f})  "
              f"end {v2['end']['delta']:+.3f} (SE {v2['end']['se']:.3f}) "
              f"UB95 {v2['end']['delta'] + 1.645 * v2['end']['se']:+.3f}"
              + ('  FAIL-FAST' if ff else ''))
    if v1:
        print(f"vec probe sel v1 (held out): end {v1['end']['delta']:+.3f} (SE {v1['end']['se']:.3f})")
    e1 = bool(v2 and v1 and (v2['end']['delta'] + 1.645 * v2['end']['se']) < 0 and v1['end']['delta'] < 0)
    if st:
        print(f"strength vs promoted: {st['delta']:+.3f}/deal (SE {st['se']:.3f}) UB95 {st['ub95']:+.3f}")
    e2 = bool(st and st['ub95'] <= 0.05)
    if tr:
        print(f"transfer n={tr['n']}: {tr['delta']:+.3f} (SE {tr['se']:.3f}, p={tr['p_one_sided']:.3f}) "
              f"[{tr['cand_moons']:.3f} vs promoted {tr['base_moons']:.3f} moons/match]")
    e3 = bool(tr and tr['delta'] <= -0.15)
    out.update({'fail_fast': ff, 'E1': e1, 'E2': e2, 'E3': e3,
                'eligible': bool(e1 and e2 and e3)})
    print(f"E1 {e1}  E2 {e2}  E3 {e3}  -> {'ELIGIBLE' if out['eligible'] else 'not eligible'}")
    os.makedirs('equity_data/verdicts', exist_ok=True)
    json.dump(out, open(f'equity_data/verdicts/r9_{name}_readout.json', 'w'), indent=1, default=float)
