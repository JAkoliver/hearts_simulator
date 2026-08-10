"""Phase 2 Stage C strength screen (prereg rider, signed 2026-08-10
pre-data): tree-200 teacher config vs deployed flat-64, CRN-paired
per-deal scores. BAND: one-sided 95% UB of (tree - flat) <= +1.0 pt.
Gross-failure screen only. Verdict: equity_data/verdicts/p2_strength.json
Exit 0 = PASS (generation may launch), 1 = HALT."""
import json
import sys

import numpy as np
from scipy import stats

tree, flat = [], []
for i in (0, 1):
    t = np.genfromtxt(f'expert_data/p2/strength_tree_s{i}.csv',
                      delimiter=',', names=True)['diff']
    f = np.genfromtxt(f'expert_data/p2/strength_flat_s{i}.csv',
                      delimiter=',', names=True)['diff']
    if len(t) != len(f):
        print(f'shard {i}: row mismatch {len(t)} vs {len(f)} - HALT')
        sys.exit(1)
    tree.append(t)
    flat.append(f)
d = np.concatenate(tree) - np.concatenate(flat)
mean = float(d.mean())
se = float(d.std(ddof=1) / np.sqrt(len(d)))
ub = mean + float(stats.t.ppf(0.95, len(d) - 1)) * se
ok = ub <= 1.0
out = {'n': int(len(d)), 'mean_tree_minus_flat': round(mean, 4),
       'se': round(se, 4), 'ub95': round(ub, 4), 'band': '+1.0',
       'pass': bool(ok)}
json.dump(out, open('equity_data/verdicts/p2_strength.json', 'w'), indent=1)
print(f'STRENGTH SCREEN n={len(d)}: tree-flat {mean:+.3f} (SE {se:.3f}) '
      f'UB95 {ub:+.3f} vs +1.0 -> {"PASS" if ok else "HALT"}')
sys.exit(0 if ok else 1)
