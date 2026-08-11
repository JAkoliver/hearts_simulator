"""Phase 2 Stage E — standard promotion battery for one candidate
(docs/phase2_visitcount_prereg.md): match gate n=3200 alpha=0.05
placement SUPERIORITY vs the md5-verified milestone + search guard
n=4800 K=32 one-sided 95% UB <= +0.3.

Usage: python run_p2_gates.py <candidate.pth> <tag>
"""
import json
import sys

from scipy import stats

from orchestrator import (evaluate_candidate_match, evaluate_candidate_search)

BASELINE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'   # 8a89da90


def main():
    cand, tag = sys.argv[1], sys.argv[2]
    out = {'candidate': cand, 'baseline': BASELINE}

    print(f'=== P2 GATE 1: match superiority ({cand}) ===', flush=True)
    # workers=6, not the orchestrator's 12: each CUDA worker takes
    # ~2.3GB of commit charge and 12 blew past the pagefile ceiling
    # (WinError 1455, 2026-08-11)
    _, m, se, p = evaluate_candidate_match(cand, BASELINE,
                                           matches=3200, workers=6)
    g1 = bool(p < 0.05 and m < 0)
    out['match'] = {'dplace_mean': m, 'se': se, 'p': p, 'pass': g1}
    print(f'GATE1 dplace {m:+.4f} (SE {se:.4f}) p={p:.4f} '
          f'-> {"PASS" if g1 else "FAIL"}', flush=True)

    print('=== P2 GATE 2: search non-regression guard ===', flush=True)
    _, sg_mean, sg_p, sg_se = evaluate_candidate_search(cand, deals=4800,
                                                        k=32, alpha=0.05)
    if sg_mean is None:
        out['search'] = {'available': False}
        g2 = False
        print('GATE2 unavailable -> FAIL (never promote blind)', flush=True)
    else:
        ub = sg_mean + float(stats.t.ppf(0.95, 4800 - 1)) * sg_se
        g2 = bool(ub <= 0.3)
        out['search'] = {'delta': sg_mean, 'se': sg_se, 'ub95': ub,
                         'pass': g2}
        print(f'GATE2 delta {sg_mean:+.4f} (SE {sg_se:.4f}) UB95 {ub:+.4f} '
              f'vs +0.3 -> {"PASS" if g2 else "FAIL"}', flush=True)

    out['all_pass'] = bool(g1 and g2)
    json.dump(out, open(f'equity_data/verdicts/p2_gates_{tag}.json', 'w'),
              indent=1)
    print(f'P2 GATES {tag}: {"ALL PASS" if out["all_pass"] else "HALT"}',
          flush=True)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
