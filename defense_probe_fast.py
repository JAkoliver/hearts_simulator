"""Exploiter league round 4 - FAST DEFENSE PROBE (docs/exploiter_league_r4_prereg.md §3.2).

Three DEFENDER seats = the net under test (raw v5 policy, argmax - the
same behaviour the gates measure), one ATTACKER seat = the certified
SEL shooter clone (shooter_sel_v1.pth, argmax), attacker seat rotating
by match index, matches to 100 on MatchEnv. CRN-paired: for every match
seed the BASELINE defenders (8a89da90) play the same seed, and each
candidate's paired delta is (cand moons conceded - base moons conceded)
per match. Within one worker chunk the base arm is played ONCE per seed
and shared by every candidate in the call, so validating 7 nets costs
8 matches per seed, not 14.

This is a PROXY for the registered SEL search-shooter defense gate
(clone attacker, millisecond speed) - it selects and halts; it never
promotes. It must pass the §3.3 validation on archived candidates
before it may be used for either.

Primary per-match statistic: moons conceded to the attacker.
Secondaries (reported, gate nothing): defender-seat mean placement,
attacker final score, deals per match, defenders' accidental moons.
NOT implemented here (lives in the search gate's tricks CSV): points
paid while a threat is alive.

Seeds: --seed base (prereg 740,000,000) + worker * 1,000,000 + idx * 1000
(match_eval's convention). Audit against used blocks before launch.

Usage:
  python defense_probe_fast.py --nets a.pth b.pth ... [--base <champion>]
        --matches 1000 --workers 12 --seed 740000000 --out probe.csv
Rows: net,idx,attacker_seat,seed,cand_conceded,base_conceded,
      cand_def_place,base_def_place,cand_att_score,base_att_score,
      cand_deals,base_deals,cand_def_moons,base_def_moons
"""
import argparse
import os
import time

import numpy as np
import scipy.stats as stats
import torch

import headroom
from hearts_match_env import MatchEnv
from hearts_net import HeartsNetV5, net_from_checkpoint
from match_eval import _play_match

CHAMPION = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
ATTACKER = 'shooter_sel_v1.pth'
_SEED_STRIDE = 1_000_000


def load_attacker(path=None):
    path = path or ATTACKER
    ck = torch.load(path, weights_only=True, map_location='cpu')
    net = HeartsNetV5(obs_dim=556, d_model=ck['d_model'],
                      num_layers=ck['num_layers'],
                      num_heads=ck.get('num_heads', 6))
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net


def _one(menv_seed, defenders, attacker, att_seat):
    seats = [defenders] * 4
    seats[att_seat] = attacker
    menv = MatchEnv(seed=menv_seed)
    placements, finals, tele = _play_match(menv, seats)
    moons = tele['moons']
    conceded = int(moons[att_seat])
    def_idx = [s for s in range(4) if s != att_seat]
    return {
        'conceded': conceded,
        'def_place': float(np.mean([placements[s] for s in def_idx])),
        'att_score': float(finals[att_seat]),
        'deals': int(tele['deals']),
        'def_moons': int(sum(moons[s] for s in def_idx)),
    }


def _chunk(job):
    net_paths, base_path, seed, offset, n_matches, attacker_path = job
    torch.set_num_threads(1)
    attacker = load_attacker(attacker_path)
    base = net_from_checkpoint(base_path); base.eval()
    cands = []
    for p in net_paths:
        n = net_from_checkpoint(p); n.eval(); cands.append(n)
    rows = []
    for m in range(n_matches):
        idx = offset + m
        att_seat = idx % 4
        match_seed = seed + idx * 1000
        b = _one(match_seed, base, attacker, att_seat)
        for p, c in zip(net_paths, cands):
            a = _one(match_seed, c, attacker, att_seat)
            rows.append((os.path.basename(p), idx, att_seat, match_seed, a, b))
    return rows


def run(net_paths, base_path, matches, workers, seed, out, attacker=None):
    attacker = attacker or ATTACKER
    workers = headroom.scaled_workers(workers)
    print(f"FAST DEFENSE PROBE: {len(net_paths)} nets vs base {base_path} | "
          f"attacker {attacker} | {matches} CRN-paired matches | seed {seed} | "
          f"workers {workers}", flush=True)
    per, extra = divmod(matches, workers)
    jobs, offset = [], 0
    for w in range(workers):
        n = per + (1 if w < extra else 0)
        if n == 0:
            continue
        jobs.append((net_paths, base_path, seed + w * _SEED_STRIDE, offset, n, attacker))
        offset += n
    import multiprocessing
    t0 = time.time()
    with multiprocessing.Pool(len(jobs),
                              initializer=headroom.apply_process_priority) as pool:
        results = pool.map(_chunk, jobs)
    rows = [r for ch in results for r in ch]
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        if new:
            f.write("net,idx,attacker_seat,seed,cand_conceded,base_conceded,"
                    "cand_def_place,base_def_place,cand_att_score,base_att_score,"
                    "cand_deals,base_deals,cand_def_moons,base_def_moons\n")
        for tag, idx, s, ms, a, b in rows:
            f.write(f"{tag},{idx},{s},{ms},{a['conceded']},{b['conceded']},"
                    f"{a['def_place']:.4f},{b['def_place']:.4f},"
                    f"{a['att_score']:.1f},{b['att_score']:.1f},"
                    f"{a['deals']},{b['deals']},{a['def_moons']},{b['def_moons']}\n")
    print(f"rows -> {out}  [{time.time() - t0:.0f}s]", flush=True)
    return summarize(rows, base_path)


def summarize(rows, base_path):
    by = {}
    for tag, idx, s, ms, a, b in rows:
        by.setdefault(tag, []).append((a, b))
    out = {}
    print(f"\nPAIRED DELTA moons conceded/match (negative = candidate defends better), base={base_path}")
    for tag, pairs in by.items():
        d = np.array([a['conceded'] - b['conceded'] for a, b in pairs], float)
        n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
        if se > 0:
            t, p = stats.ttest_1samp(d, 0.0, alternative='less')
        else:
            t, p = 0.0, 1.0
        h = stats.t.ppf(0.975, n - 1) * se if se > 0 else 0.0
        cb = np.mean([b['conceded'] for a, b in pairs]); ca = np.mean([a['conceded'] for a, b in pairs])
        dp = np.mean([a['def_place'] - b['def_place'] for a, b in pairs])
        out[tag] = {'n': n, 'cand_moons_per_match': ca, 'base_moons_per_match': cb,
                    'delta': mean, 'se': se, 'p_one_sided': float(p),
                    'ci95': [mean - h, mean + h], 'ddef_place': dp}
        print(f"  {tag:32s} n={n:5d}  cand {ca:.3f}  base {cb:.3f}  "
              f"delta {mean:+.4f} (SE {se:.4f}) CI [{mean-h:+.3f},{mean+h:+.3f}]  "
              f"p={p:.4f}  ddef_place {dp:+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', required=True)
    ap.add_argument('--base', default=CHAMPION)
    ap.add_argument('--matches', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=740_000_000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--attacker', default=None, help='clone attacker checkpoint (default shooter_sel_v1.pth)')
    a = ap.parse_args()
    res = run(a.nets, a.base, a.matches, a.workers, a.seed, a.out, a.attacker)
    if a.json:
        import json
        json.dump({'base': a.base, 'attacker': a.attacker or ATTACKER, 'matches': a.matches,
                   'seed': a.seed, 'results': res}, open(a.json, 'w'),
                  indent=1, default=float)


if __name__ == '__main__':
    main()
