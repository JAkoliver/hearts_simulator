"""Paired search-vs-search comparison of two SEARCH TRACES on neutral tables.

Both sides play 1 search seat vs 3 neutral v3-m7 raw anchors on identical
deals (gate-style sharded pairing); reports the per-deal delta A - B
(negative = A better). Built 2026-07-23 to measure the current deployed
player against the 2026-07-14 calibration opponent (v4-m10).

Usage:
  python eval_search_pair.py --a hearts_ai_search.pt \
      --b hearts_ai_search_v4m10.pt [--deals 1200] [--k 64] [--shards 4]
"""
import argparse
import time

import numpy as np
import scipy.stats as stats

from orchestrator import (_GATE_SHARD_STRIDE, NEUTRAL_OPPONENT,
                          _search_finish, _search_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='search trace A (e.g. current)')
    ap.add_argument('--b', required=True, help='search trace B (e.g. v4-m10)')
    ap.add_argument('--deals', type=int, default=1200)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--shards', type=int, default=4)
    args = ap.parse_args()

    seed = int(time.time())
    per = args.deals // args.shards
    extra = args.deals % args.shards
    print(f"Search pair: {args.a} vs {args.b}, {args.deals} paired deals, "
          f"K={args.k}, {args.shards} shard pairs, neutral anchors, seed {seed}")

    procs = []
    for i in range(args.shards):
        n = per + (1 if i < extra else 0)
        if n == 0:
            continue
        s = seed + i * _GATE_SHARD_STRIDE
        a_csv, b_csv = f'se_pair_a{i}.csv', f'se_pair_b{i}.csv'
        procs.append((i, n,
                      _search_start(args.a, NEUTRAL_OPPONENT, n, args.k, s, a_csv), a_csv,
                      _search_start(args.b, NEUTRAL_OPPONENT, n, args.k, s, b_csv), b_csv))

    a_parts, b_parts = [], []
    for i, n, pa, a_csv, pb, b_csv in procs:
        a = _search_finish(pa, a_csv)
        b = _search_finish(pb, b_csv)
        if len(np.atleast_1d(a)) != n or len(np.atleast_1d(b)) != n:
            raise RuntimeError(f"shard {i}: row count mismatch")
        a_parts.append(np.atleast_1d(a))
        b_parts.append(np.atleast_1d(b))

    delta = np.concatenate(a_parts) - np.concatenate(b_parts)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    t_two, p_two = stats.ttest_1samp(delta, 0.0)
    print(f"PAIR RESULT (negative = {args.a} better): {mean:+.3f} "
          f"(SE {se:.3f}, n={len(delta)})")
    print(f"T-Statistic: {t_two:.3f}, two-sided P-Value: {p_two:.5f}")


if __name__ == '__main__':
    main()
