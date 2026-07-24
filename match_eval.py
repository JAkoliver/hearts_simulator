"""Paired match-to-100 gate (docs/ROADMAP.md phase 1).

Candidate and baseline each play the SAME seat of the SAME deal-sequence
seed against three neutral v3-m7 anchor seats, one full match per side per
seed. Primary paired statistics on the test seat: match win, placement,
final score. Telemetry rider (experiment_rules.md #5, informs-never-gates):
moons for/against, deals per match, phase-relevant outcome counts.

Usage:
  python match_eval.py --cand <ckpt> [--base hearts_model_final.pth]
                       [--matches 800] [--workers 12] [--seed auto]
"""
import argparse
import os
import time

import numpy as np
import scipy.stats as stats
import torch

from hearts_match_env import MatchEnv
from hearts_net import net_from_checkpoint
from orchestrator import NEUTRAL_OPPONENT, _LegacySeat

_SEED_STRIDE = 1_000_000


def _act(net, obs, mask_list):
    mask = torch.zeros((1, 52), dtype=torch.bool)
    for a in mask_list:
        if a != -1:
            mask[0, a] = True
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(obs).unsqueeze(0), mask)
    return int(torch.argmax(logits, dim=1).item())


def _play_match(menv, seat_nets):
    """Play one match to 100; returns (placements, final_scores, telemetry)."""
    menv.reset_match()
    moons = np.zeros(4, dtype=np.int64)
    while True:
        cp = menv.get_current_player()
        action = _act(seat_nets[cp], menv.observe(), menv.get_legal_actions())
        deal_done, match_done, round_scores = menv.step(action)
        if deal_done:
            srt = np.sort(round_scores)
            if srt[0] == 0 and np.all(srt[1:] == 26):
                moons[int(np.argmin(round_scores))] += 1
        if match_done:
            return menv.placements(), menv.match_scores.copy(), \
                {'deals': menv.deals_played, 'moons': moons}


def _chunk(job):
    cand_path, base_path, seed, offset, n_matches = job
    torch.set_num_threads(1)
    cand = net_from_checkpoint(cand_path)
    cand.eval()
    base = net_from_checkpoint(base_path)
    base.eval()
    anchor = _LegacySeat(torch.jit.load(NEUTRAL_OPPONENT))
    anchor.eval()

    rows = []
    for m in range(n_matches):
        idx = offset + m
        seat = idx % 4
        match_seed = seed + idx * 1000
        out = {}
        for label, net in (('a', cand), ('b', base)):
            seats = [anchor] * 4
            seats[seat] = net
            menv = MatchEnv(seed=match_seed)
            placements, finals, tele = _play_match(menv, seats)
            out[label] = (placements[seat], finals[seat], tele['deals'],
                          tele['moons'][seat],
                          int(tele['moons'].sum() - tele['moons'][seat]))
        rows.append((out['a'], out['b']))
    return rows


def run_gate(cand, base, matches=800, workers=12, seed=None):
    """Run the paired match gate; prints the report and returns a stats dict."""
    seed = seed if seed is not None else int(time.time())
    print(f"Match gate: {cand} vs {base} @ shared seat, 3x v3-m7 "
          f"anchors, {matches} paired matches to 100, seed {seed}")

    per = matches // workers
    extra = matches % workers
    jobs, offset = [], 0
    for w in range(workers):
        n = per + (1 if w < extra else 0)
        if n == 0:
            continue
        jobs.append((cand, base, seed + w * _SEED_STRIDE, offset, n))
        offset += n

    import multiprocessing
    t0 = time.time()
    with multiprocessing.Pool(len(jobs)) as pool:
        results = pool.map(_chunk, jobs)
    rows = [r for chunk in results for r in chunk]

    a = np.array([[r[0][0], r[0][1], r[0][2], r[0][3], r[0][4]] for r in rows])
    b = np.array([[r[1][0], r[1][1], r[1][2], r[1][3], r[1][4]] for r in rows])
    n = len(rows)

    win_a = (a[:, 0] == 1.0)
    win_b = (b[:, 0] == 1.0)
    dplace = a[:, 0] - b[:, 0]
    dscore = a[:, 1] - b[:, 1]

    t_pl, p_pl = stats.ttest_1samp(dplace, 0.0, alternative='less')
    t_sc, p_sc = stats.ttest_1samp(dscore, 0.0, alternative='less')
    # Paired win comparison: discordant matches only (McNemar-style binomial)
    a_only = int(np.sum(win_a & ~win_b))
    b_only = int(np.sum(~win_a & win_b))
    p_win = stats.binomtest(a_only, a_only + b_only, 0.5,
                            alternative='greater').pvalue \
        if (a_only + b_only) > 0 else 1.0

    print(f"\nMATCH RESULT ({n} paired matches; candidate=A, baseline=B):")
    print(f"  win rate:   A {win_a.mean() * 100:.1f}%  B {win_b.mean() * 100:.1f}%  "
          f"(discordant {a_only}:{b_only}, one-sided p={p_win:.5f})")
    print(f"  placement:  A {a[:, 0].mean():.3f}  B {b[:, 0].mean():.3f}  "
          f"paired delta {dplace.mean():+.3f} (SE {dplace.std(ddof=1) / np.sqrt(n):.3f}, "
          f"p={p_pl:.5f})")
    print(f"  final score: A {a[:, 1].mean():.2f}  B {b[:, 1].mean():.2f}  "
          f"paired delta {dscore.mean():+.2f} (SE {dscore.std(ddof=1) / np.sqrt(n):.2f}, "
          f"p={p_sc:.5f})")
    print(f"\nTELEMETRY (informs, never gates):")
    print(f"  deals/match: A {a[:, 2].mean():.2f}  B {b[:, 2].mean():.2f}")
    print(f"  moons shot (test seat): A {int(a[:, 3].sum())}  B {int(b[:, 3].sum())}")
    print(f"  moons conceded at table: A {int(a[:, 4].sum())}  B {int(b[:, 4].sum())}")
    print(f"  [{time.time() - t0:.0f}s]")

    return {
        'n': n, 'win_a': float(win_a.mean()), 'win_b': float(win_b.mean()),
        'discordant': (a_only, b_only), 'p_win': float(p_win),
        'dplace_mean': float(dplace.mean()),
        'dplace_se': float(dplace.std(ddof=1) / np.sqrt(n)),
        'p_place': float(p_pl),
        'dscore_mean': float(dscore.mean()),
        'dscore_se': float(dscore.std(ddof=1) / np.sqrt(n)),
        'p_score': float(p_sc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--base', default='hearts_model_final.pth')
    ap.add_argument('--matches', type=int, default=800)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    run_gate(args.cand, args.base, args.matches, args.workers, args.seed)


if __name__ == '__main__':
    main()
