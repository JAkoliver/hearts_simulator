"""Side-probe guard run (docs/phase2_visitcount_prereg.md, registered):
the lambda=0.05 candidate landed in the 5-15% drift band, so the search
guard (n=4800, K=32, one-sided 95% UB vs +0.3) runs on it to capture
the guard-tolerance-vs-drift curve point rounds 1-3 never measured.
EXPLORATORY: not candidate-eligible; the number is the deliverable.

Usage: python run_probe_guard.py
"""
import json

from scipy import stats

from orchestrator import evaluate_candidate_search

CAND = 'cand_r3_probe005.pth'

_, mean, p, se = evaluate_candidate_search(CAND, deals=4800, k=32, alpha=0.05)
if mean is None:
    raise SystemExit('guard unavailable')
ub = mean + float(stats.t.ppf(0.95, 4800 - 1)) * se
out = {'candidate': CAND, 'context': 'lambda=0.05 side-probe, drift 7.77%',
       'delta': mean, 'se': se, 'ub95': ub, 'bar': 0.3,
       'within_guard': bool(ub <= 0.3)}
json.dump(out, open('equity_data/verdicts/r3_probe005_guard.json', 'w'),
          indent=1)
print(f'PROBE GUARD: delta {mean:+.4f} (SE {se:.4f}) UB95 {ub:+.4f} vs +0.3 '
      f'-> {"WITHIN" if out["within_guard"] else "OUTSIDE"} guard tolerance')
