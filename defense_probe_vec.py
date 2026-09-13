"""League r9 - VECTORIZED fast defense probe (docs/exploiter_league_r9_prereg.md §3.3).

The same measurement as defense_probe_fast.py (3 DEFENDER seats = the net
under test at argmax, 1 ATTACKER seat = a clone at argmax, attacker seat
rotating by match index, matches to 100, CRN-paired against a BASE net on
identical env seeds, primary = moons conceded per match) played on
MatchVecEnv with GPU batching instead of one MatchEnv per CPU worker.
Estimation instrument only (selects, informs, halts; never promotes).

Seeds: HeartsVecEnv(n, seed0) seeds env i with seed0 + i; match m of the
run lives in env m (one match per env, n envs = n matches), so the seed of
match m is seed0 + m. Registered block seed0 = 760,000,000 (fresh; the
sequential probe's 740M block and the guard's 745-756M blocks are
disjoint). Base and candidate arms play the SAME env seeds (CRN).

Moon accounting = the established one (match_eval._play_match): a deal
whose round scores sort to [0, 26, 26, 26] is a moon by the 0 seat.

Usage:
  python defense_probe_vec.py --nets a.pth b.pth --base <promoted.pth>
      --attacker shooter_sel_v2.pth --matches 1000 --seed 760000000
      --out probe.csv --json probe.json
Rows: net,idx,attacker_seat,seed,cand_conceded,base_conceded,
      cand_def_place,base_def_place,cand_att_score,base_att_score,
      cand_deals,base_deals,cand_def_moons,base_def_moons
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.stats as stats
import torch

import headroom
from hearts_match_env import MatchVecEnv
from hearts_net import net_from_checkpoint
from defense_probe_fast import load_attacker, summarize


def _argmax_batch(net, obs_np, mask_np, device):
    obs = torch.from_numpy(obs_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    with torch.no_grad():
        logits, _ = net(obs, mask)          # fp32, no autocast: the served numerics
    return logits.argmax(1).cpu().numpy()


def play_arm(defender, attacker, n, seed0, device, tag):
    """One arm: n envs = n matches; env m's attacker seat = m % 4. Returns a
    list of per-match dicts (same keys as defense_probe_fast._one)."""
    vec = MatchVecEnv(n, seed0)
    att_seat = np.arange(n) % 4
    alive = np.ones(n, dtype=bool)
    moons = np.zeros((n, 4), dtype=np.int64)
    deals = np.zeros(n, dtype=np.int64)
    finals = np.zeros((n, 4), dtype=np.float64)
    places = np.zeros((n, 4), dtype=np.float64)
    def_882 = getattr(defender, 'obs_dim', 0) == 882
    t0 = time.time()
    steps = 0
    while alive.any():
        headroom.pace()
        g_all = np.flatnonzero(alive)
        cp = vec.current_players()[g_all]
        is_att = (cp == att_seat[g_all])
        actions = np.zeros(len(g_all), dtype=np.int64)
        mask = vec.legal_mask_batch(g_all)
        if (~is_att).any():
            gd = g_all[~is_att]
            obs = vec.observe_v2_batch(gd) if def_882 else vec.observe_batch(gd)
            actions[~is_att] = _argmax_batch(defender, obs, mask[~is_att], device)
        if is_att.any():
            ga = g_all[is_att]
            obs = vec.observe_batch(ga)          # clones are 556 nets
            actions[is_att] = _argmax_batch(attacker, obs, mask[is_att], device)
        # match totals BEFORE the step (step_batch zeroes them at match end)
        pre_scores = vec.match_scores[g_all].copy()
        deal_dones, match_dones, placements, rs = vec.step_batch(g_all, actions)
        for j in np.flatnonzero(deal_dones):
            e = int(g_all[j])
            r = np.asarray(rs[j], dtype=np.float64)
            srt = np.sort(r)
            if srt[0] == 0 and np.all(srt[1:] == 26):
                moons[e, int(np.argmin(r))] += 1
            deals[e] += 1
            if match_dones[j]:
                finals[e] = pre_scores[j] + r
                places[e] = np.asarray(placements[j], dtype=np.float64)
                alive[e] = False
        steps += 1
        if steps % 500 == 0:
            print(f"  [{tag}] step {steps}: {int((~alive).sum())}/{n} matches done "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out = []
    for m in range(n):
        a = int(att_seat[m])
        d = [s for s in range(4) if s != a]
        out.append({'conceded': int(moons[m, a]),
                    'def_place': float(np.mean(places[m, d])),
                    'att_score': float(finals[m, a]),
                    'deals': int(deals[m]),
                    'def_moons': int(moons[m, d].sum())})
    print(f"  [{tag}] {n} matches in {time.time() - t0:.0f}s", flush=True)
    return out


def run(net_paths, base_path, matches, seed0, out, attacker_path, json_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attacker = load_attacker(attacker_path).to(device).eval()
    print(f"VEC DEFENSE PROBE: {len(net_paths)} nets vs base {base_path} | "
          f"attacker {attacker_path} | {matches} CRN-paired matches | seed0 {seed0} | "
          f"{device}", flush=True)
    base = net_from_checkpoint(base_path).to(device).eval()
    b_rows = play_arm(base, attacker, matches, seed0, device, 'base')
    rows = []
    for p in net_paths:
        cand = net_from_checkpoint(p).to(device).eval()
        c_rows = play_arm(cand, attacker, matches, seed0, device, os.path.basename(p))
        for m in range(matches):
            rows.append((os.path.basename(p), m, m % 4, seed0 + m, c_rows[m], b_rows[m]))
        del cand
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='') as f:
        f.write("net,idx,attacker_seat,seed,cand_conceded,base_conceded,"
                "cand_def_place,base_def_place,cand_att_score,base_att_score,"
                "cand_deals,base_deals,cand_def_moons,base_def_moons\n")
        for tag, idx, s, ms, a, b in rows:
            f.write(f"{tag},{idx},{s},{ms},{a['conceded']},{b['conceded']},"
                    f"{a['def_place']:.4f},{b['def_place']:.4f},"
                    f"{a['att_score']:.1f},{b['att_score']:.1f},"
                    f"{a['deals']},{b['deals']},{a['def_moons']},{b['def_moons']}\n")
    print(f"rows -> {out}", flush=True)
    res = summarize(rows, base_path)
    if json_path:
        with open(json_path, 'w') as f:
            json.dump({'base': base_path, 'attacker': attacker_path,
                       'matches': matches, 'seed0': seed0, 'harness': 'vec',
                       'results': res}, f, indent=1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', required=True)
    ap.add_argument('--base', default='hybrid_champ_arma_moonhead_0p1.pth')
    ap.add_argument('--attacker', default='shooter_sel_v1.pth')
    ap.add_argument('--matches', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=760_000_000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    headroom.apply_process_priority()
    run(args.nets, args.base, args.matches, args.seed, args.out, args.attacker,
        args.json)
