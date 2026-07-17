"""Cross-hardware statistical equivalence gate (REQUIREMENTS R1).

Bitwise identity across GPUs/compilers is explicitly NOT the standard;
this is the standard. Criterion is fixed here, in committed code, before
any cloud run executes - same style as docs/sdpa_gate_preregistration.md:

  Inputs: two SearchEval CSVs over the SAME seeds - the cloud-built stack
  and the local stack, each vs the neutral v3-m7 anchor, K=32, pass-search.
  delta per deal = cloud_diff - local_diff (positive = cloud stack worse).

  PASS iff BOTH:
    1. mean(delta) < +0.20 pts/deal
    2. mean(delta) + 1.645 * SE < +0.30   (raw_guard_threshold margin)

  FAIL: cloud data must not feed a real iteration until the discrepancy is
  understood; any re-attempt is a fresh pre-registered run (new seeds).

Recommended n: >= 2000 paired deals (SE ~ 0.17 at the observed per-deal
std of ~7.6; run 2 of the SDPA gate is the precedent).

Usage: python cloud/xhw_gate.py cloud.csv local.csv
"""
import sys
import numpy as np

MARGIN = 0.30
POINT_GUARD = 0.20
Z_05 = 1.645


def main(cloud_csv, local_csv):
    c = np.genfromtxt(cloud_csv, delimiter=',', names=True)['diff']
    l = np.genfromtxt(local_csv, delimiter=',', names=True)['diff']
    if len(c) != len(l):
        raise SystemExit(f"unpaired inputs: {len(c)} vs {len(l)} deals")
    d = c - l
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    ub = mean + Z_05 * se
    c1, c2 = mean < POINT_GUARD, ub < MARGIN
    print(f"n={n}  mean {mean:+.4f}  SE {se:.4f}  one-sided 95% UB {ub:+.4f}")
    print(f"criterion 1 (mean < +{POINT_GUARD}): {'MET' if c1 else 'NOT MET'}")
    print(f"criterion 2 (UB < +{MARGIN}):   {'MET' if c2 else 'NOT MET'}")
    verdict = 'PASS' if (c1 and c2) else 'FAIL'
    print(f"VERDICT: {verdict}")
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
