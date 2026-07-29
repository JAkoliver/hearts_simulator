"""Pre-registered N=8000 validation analysis (match-aware search vs frozen
match-blind reference, anchor-field paired matches, CRN).

Primary (pre-registered, single analysis, alpha=0.05 one-sided):
  match-win McNemar on paired wins.
Also reported per pre-registration: realized paired placement SD (vs the
bridge run's 1.35) and realized discordance rate q.

LIMITATION (stated, not patched): the fleet binary logged only final match
outcomes, so the pre-registered S1/S2 stratification and the
flipped-decision dose-response secondaries are NOT computable from this
dataset. Match length (deals) is reported as an EXPLORATORY dose proxy
only - it is not the registered dose measure.
"""
import glob
import math
import sys

import numpy as np


def binom_sf_one_sided(b, n):
    """P(X >= b) for X ~ Binom(n, 0.5) - exact, one-sided, log-space."""
    if n == 0:
        return 1.0
    log_terms = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                 - n * math.log(2.0) for k in range(b, n + 1)]
    mx = max(log_terms)
    return math.exp(mx) * sum(math.exp(t - mx) for t in log_terms)


def main():
    files = sorted(glob.glob("equity_data/validation_v1/fleet_shard_*.csv"))
    rows = []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip()
            assert header.startswith("match,seat"), f"{f}: bad header"
            for line in fh:
                p = line.strip().split(",")
                if len(p) == 10:
                    rows.append([float(x) for x in p])
    n = len(rows)
    print(f"shards={len(files)} pairs={n}")
    if n == 0:
        sys.exit(1)

    a_win = np.array([r[5] for r in rows], dtype=int)
    b_win = np.array([r[9] for r in rows], dtype=int)
    a_place = np.array([r[4] for r in rows])
    b_place = np.array([r[8] for r in rows])
    a_deals = np.array([r[2] for r in rows], dtype=int)
    b_deals = np.array([r[6] for r in rows], dtype=int)

    # Primary: McNemar one-sided (H1: match-aware arm wins more)
    b_disc = int(np.sum((a_win == 1) & (b_win == 0)))  # A-only wins
    c_disc = int(np.sum((a_win == 0) & (b_win == 1)))  # B-only wins
    m = b_disc + c_disc
    p_one = binom_sf_one_sided(b_disc, m)
    print(f"\nPRIMARY McNemar (one-sided, alpha=0.05):")
    print(f"  wins A={a_win.sum()} ({100*a_win.mean():.2f}%)  "
          f"wins B={b_win.sum()} ({100*b_win.mean():.2f}%)")
    print(f"  discordant: A-only={b_disc}  B-only={c_disc}  q={m/n:.4f}")
    print(f"  p={p_one:.5f}  -> {'SIGNIFICANT' if p_one < 0.05 else 'NOT significant'}")

    # Win-rate diff with SE (paired)
    d = a_win - b_win
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"  win-rate diff (A-B): {d.mean()*100:+.2f} pts  SE {se*100:.2f} pts")

    # CRN reporting (pre-registered)
    pd = a_place - b_place
    print(f"\nCRN check: paired placement diff mean {pd.mean():+.4f}  "
          f"SD {pd.std(ddof=1):.3f} (bridge ref 1.35)  q={m/n:.4f}")

    # Placement distribution
    for arm, pl in (("A(match-aware)", a_place), ("B(ref)", b_place)):
        cnt = [int(np.sum(pl == k)) for k in (1, 2, 3, 4)]
        print(f"  places {arm}: P1={cnt[0]} P2={cnt[1]} P3={cnt[2]} P4={cnt[3]} "
              f"mean={pl.mean():.3f}")

    # EXPLORATORY ONLY: match-length dose proxy (not the registered measure)
    print("\nEXPLORATORY (not pre-registered dose measure): win diff by deals_a tercile")
    ter = np.quantile(a_deals, [1 / 3, 2 / 3])
    for lab, mask in (("short", a_deals <= ter[0]),
                      ("mid", (a_deals > ter[0]) & (a_deals <= ter[1])),
                      ("long", a_deals > ter[1])):
        dd = d[mask]
        if len(dd):
            print(f"  {lab:5s} n={len(dd):4d} diff {dd.mean()*100:+.2f} "
                  f"SE {dd.std(ddof=1)/math.sqrt(len(dd))*100:.2f}")

    # Per-shard consistency
    print("\nPer-shard win diff:")
    idx = 0
    for f in files:
        cnt = sum(1 for _ in open(f)) - 1
        dd = d[idx:idx + cnt]
        print(f"  {f.split('_')[-1]}: n={cnt} diff {dd.mean()*100:+.2f}")
        idx += cnt


if __name__ == "__main__":
    main()
