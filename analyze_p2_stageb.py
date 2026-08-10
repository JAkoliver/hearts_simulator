"""Phase 2 Stage B — teacher-signal probe analysis
(docs/phase2_visitcount_prereg.md, registered bands).

Per budget {200, 400, 800}, from the dual-probe outputs:
  - one-hot HALT (NON-AMENDABLE): >=80% of non-forced decisions with
    top-1 visit share >= 0.90  -> the r2 dead currency in disguise.
  - near-uniform HALT: median top-1 visit share < 0.35.
  - validity: Spearman(visit-share gap top1-top2, flat value gap
    best-second) >= 0.3 (amendable by ONE pre-unblinding re-band if
    shapes are healthy).
Verdict JSON: equity_data/verdicts/p2_stageb.json.
"""
import json
import sys

import numpy as np
from scipy import stats

from validate_p2_records import DT

BUDGETS = [200, 400, 800]
out = {'budgets': {}, 'halts': []}

for b in BUDGETS:
    try:
        rows = np.genfromtxt(f'expert_data/p2/stageb_{b}.csv', delimiter=',',
                             names=True)
    except OSError:
        out['halts'].append(f'budget {b}: csv missing')
        continue
    r = np.fromfile(f'expert_data/p2/stageb_{b}.hvt', dtype=DT)
    plays = r[(r['kind'] == 0) & (r['mask'].sum(1) > 1)]
    top = plays['pi'].max(1)
    onehot_frac = float((top >= 0.90).mean())
    med_top = float(np.median(top))

    vg = rows['pi1'] - rows['pi2']
    fg = rows['flat_best_v'] - rows['flat_second_v']
    ok = np.isfinite(vg) & np.isfinite(fg)
    rho, p = stats.spearmanr(vg[ok], fg[ok])
    entry = {'n_decisions': int(len(plays)), 'n_paired': int(ok.sum()),
             'median_top1_share': round(med_top, 3),
             'onehot_frac': round(onehot_frac, 3),
             'spearman_visitgap_valuegap': round(float(rho), 3),
             'spearman_p': float(p)}
    out['budgets'][str(b)] = entry
    if onehot_frac >= 0.80:
        out['halts'].append(f'budget {b}: ONE-HOT HALT (non-amendable) - '
                            f'{onehot_frac:.0%} of decisions >=0.90 share')
    if med_top < 0.35:
        out['halts'].append(f'budget {b}: near-uniform HALT - median top-1 '
                            f'share {med_top:.3f}')
    if rho < 0.3:
        out['halts'].append(f'budget {b}: Spearman {rho:.3f} < 0.3 '
                            f'(amendable band)')
    print(f'budget {b}: n={len(plays)} paired={ok.sum()} '
          f'top1-med {med_top:.3f} onehot {onehot_frac:.1%} '
          f'Spearman {rho:.3f} (p={p:.2e})')

out['pass'] = not out['halts']
json.dump(out, open('equity_data/verdicts/p2_stageb.json', 'w'), indent=1)
if out['halts']:
    print('STAGE B FLAGS:')
    for h in out['halts']:
        print(' -', h)
    sys.exit(1)
print('STAGE B PASS (all budgets)')
