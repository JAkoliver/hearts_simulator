"""Exploiter round-2 gates 2+3 (docs/exploiter_league_r2_prereg.md).

Runs on a candidate that already PASSED the round-2 defense gate:
  Gate 2 - MATCH NON-INFERIORITY, n RAISED to 6400 (prereg: round 1's
    ~zero point estimate failed on precision alone at 3200; SE ~0.0125
    at 6400 makes the +0.030 UB reachable for a truly-null candidate).
    Bound unchanged: one-sided 95% UB of placement delta <= +0.030.
  Gate 3 - SEARCH GUARD unchanged: n=4800, K=32, one-sided 95% UB of
    the searched-score delta <= +0.3 (candidate arm vs the FROZEN
    match-aware baseline trace; no hearts_model_final dependency -
    that file holds a rejected candidate).

Usage: python run_r2_gates.py <candidate.pth> <tag>
"""
import json
import sys

from scipy import stats

from orchestrator import (evaluate_candidate_match, evaluate_candidate_search)

BASELINE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'   # 8a89da90


def main():
    cand, tag = sys.argv[1], sys.argv[2]
    out = {'candidate': cand, 'baseline': BASELINE}

    print(f'=== GATE 2: match non-inferiority n=6400 ({cand}) ===', flush=True)
    _, m, se, p = evaluate_candidate_match(cand, BASELINE,
                                           matches=6400, workers=12)
    ub = m + 1.6449 * se
    g2 = ub <= 0.030
    out['match'] = {'n': 6400, 'dplace_mean': m, 'se': se, 'p_superior': p,
                    'ub95': ub, 'noninferior': bool(g2),
                    'superior': bool(p < 0.05 and m < 0)}
    print(f'GATE2 dplace {m:+.4f} (SE {se:.4f}) UB95 {ub:+.4f} vs +0.030 '
          f'-> {"PASS" if g2 else "FAIL"} '
          f'(superiority p={p:.4f}{", SUPERIOR" if out["match"]["superior"] else ""})',
          flush=True)

    print('=== GATE 3: search non-regression guard ===', flush=True)
    _, sg_mean, sg_p, sg_se = evaluate_candidate_search(cand, deals=4800,
                                                        k=32, alpha=0.05)
    if sg_mean is None:
        out['search'] = {'available': False}
        g3 = False
        print('GATE3 unavailable -> FAIL (never promote blind)', flush=True)
    else:
        ub3 = sg_mean + float(stats.t.ppf(0.95, 4800 - 1)) * sg_se
        g3 = ub3 <= 0.3
        out['search'] = {'delta': sg_mean, 'se': sg_se, 'ub95': ub3,
                         'pass': bool(g3)}
        print(f'GATE3 delta {sg_mean:+.4f} (SE {sg_se:.4f}) UB95 {ub3:+.4f} '
              f'vs +0.3 -> {"PASS" if g3 else "FAIL"}', flush=True)

    out['all_pass'] = bool(g2 and g3)
    json.dump(out, open(f'equity_data/verdicts/exploiter_r2_gates23_{tag}.json',
                        'w'), indent=1)
    print(f'GATES23 {"ALL PASS" if out["all_pass"] else "HALT"}', flush=True)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
