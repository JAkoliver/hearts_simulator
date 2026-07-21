"""Neutral-opponent RAW evaluation (diagnostic A, 2026-07-21).

The orchestrator's raw guard seats the candidate among the very baseline
it trained against, so its differential conflates real strength with
opponent exploitation. This evaluator seats BOTH nets, one at a time, at
the same seat of the same deal against three neutral v3-m7 anchor seats
(the search gate's opponent convention) and tests the paired per-deal
differential:

    diff = score(candidate @ seat, 3x v3 table)
         - score(baseline  @ seat, 3x v3 table)     (negative = candidate better)

Usage:
    python neutral_raw_eval.py --cand <ckpt> --base <ckpt>
                               [--deals 2500] [--workers 12] [--seed auto]

The v3-m7 anchor is a jit trace with a 238-dim observation - a prefix of
the current 550-dim layout (same convention SearchPlayer.hpp uses when it
probes {550, 238, 181}).
"""
import argparse
import os
import time

import numpy as np
import scipy.stats as stats
import torch

import hearts_env
from hearts_net import net_from_checkpoint
from orchestrator import play_round

NEUTRAL_TRACE = os.path.join('legacy_v3_pass238', 'hearts_ai_grandmaster_v3_milestone7.pt')
_SEED_STRIDE = 1_000_000  # matches the gate-shard convention


class _LegacySeat(torch.nn.Module):
    """Adapts the 238-dim v3 trace to the current 550-dim observation."""

    def __init__(self, traced):
        super().__init__()
        self.traced = traced

    def forward(self, observation, legal_actions_mask):
        return self.traced(observation[:, :238], legal_actions_mask)


def _chunk(job):
    cand_path, base_path, seed, deal_offset, n_deals = job
    torch.set_num_threads(1)

    cand = net_from_checkpoint(cand_path)
    cand.eval()
    base = net_from_checkpoint(base_path)
    base.eval()
    neutral = _LegacySeat(torch.jit.load(NEUTRAL_TRACE))
    neutral.eval()

    # Two envs on the same seed stay deal-synchronized: the engine RNG is
    # consumed only by reset()'s shuffle (same property _eval_chunk relies on).
    env_a = hearts_env.HeartsEnv(seed=seed)
    env_b = hearts_env.HeartsEnv(seed=seed)

    diffs, cand_scores, base_scores = [], [], []
    for i in range(n_deals):
        seat = (deal_offset + i) % 4

        seats_a = [neutral] * 4
        seats_a[seat] = cand
        a_scores = play_round(env_a, seats_a)

        seats_b = [neutral] * 4
        seats_b[seat] = base
        b_scores = play_round(env_b, seats_b)

        diffs.append(a_scores[seat] - b_scores[seat])
        cand_scores.append(a_scores[seat])
        base_scores.append(b_scores[seat])
    return diffs, cand_scores, base_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--base', default='hearts_model_final.pth')
    ap.add_argument('--deals', type=int, default=2500)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    print(f"Neutral raw eval: {args.cand} vs {args.base} @ seat, "
          f"3x v3-m7 anchors, {args.deals} paired deals, seed {seed}")

    per = args.deals // args.workers
    extra = args.deals % args.workers
    jobs, offset = [], 0
    for w in range(args.workers):
        n = per + (1 if w < extra else 0)
        if n == 0:
            continue
        jobs.append((args.cand, args.base, seed + w * _SEED_STRIDE, offset, n))
        offset += n

    import multiprocessing
    t0 = time.time()
    with multiprocessing.Pool(len(jobs)) as pool:
        results = pool.map(_chunk, jobs)

    diffs = np.array([d for r in results for d in r[0]], dtype=np.float64)
    cand_s = np.array([s for r in results for s in r[1]], dtype=np.float64)
    base_s = np.array([s for r in results for s in r[2]], dtype=np.float64)

    mean = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    t_stat, p = stats.ttest_1samp(diffs, 0.0, alternative='less')
    print(f"Candidate mean score vs neutral table: {cand_s.mean():.3f}")
    print(f"Baseline  mean score vs neutral table: {base_s.mean():.3f}")
    print(f"NEUTRAL RAW DELTA (negative = candidate better): {mean:+.3f} "
          f"(SE {se:.3f}, n={len(diffs)})")
    print(f"T-Statistic: {t_stat:.3f}, P-Value: {p:.5f}   [{time.time() - t0:.0f}s]")


if __name__ == '__main__':
    main()
