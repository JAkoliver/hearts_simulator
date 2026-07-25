"""Spine gate: decision-level action-flip rate + SNR from --probe-log data.

For every logged decision (per-action x per-determinization completed-
rollout deal scores + match context), scores each action two ways:
  deal-points: mean relative reward (avg - own), what search does today
  equity:      mean objective value of equity(totals + deal_scores),
               terminal states analytic, else the selected equity model
and reports, per DECISION-LEVEL score-state band (tension / runaway /
early):
  - flip rate: how often the equity argmax differs from the deal-point
    argmax (objectives: win = P(place 1), place = -E[place])
  - SNR: median |top-two equity gap| / SE(K-mean), with the same ratio
    for deal-point scoring as the known-good reference.

Pre-registered spine gate (docs/match_aware_search_design.md): tension
flip rate >= 5% AND tension SNR >= 1.0 (win objective) to proceed to the
C++ integration; HALT otherwise. Emits equity_data/verdicts/flip_snr.json.

Usage: python flip_snr_analysis.py --log probe_decisions_v1.csv \
           [--model equity_v1.pth] [--gate]
"""
import argparse

import numpy as np
import pandas as pd
import torch

from hearts_match_env import placements_of
from train_equity import EquityNet, emit_verdict, file_sha


def equity_values(model, totals_after, deals_after, seat):
    """P(place 1..4) for `seat` given post-deal totals; analytic at terminal."""
    n = len(totals_after)
    out = np.empty((n, 4), dtype=np.float32)
    terminal = totals_after.max(axis=1) >= 100.0
    for i in np.flatnonzero(terminal):
        pl = placements_of(totals_after[i])
        v = np.zeros(4, dtype=np.float32)
        lo = int(np.floor(pl[seat])) - 1
        if pl[seat] == np.floor(pl[seat]):
            v[lo] = 1.0
        else:
            v[lo] = 0.5
            v[lo + 1] = 0.5
        out[i] = v
    live = ~terminal
    if live.any():
        t = totals_after[live]
        rot = np.stack([t[:, (seat + k) % 4] for k in range(4)], axis=1)
        onehot = np.zeros((live.sum(), 4), dtype=np.float32)
        onehot[:, deals_after % 4] = 1.0
        X = np.concatenate([rot / 100.0,
                            (deals_after / 20.0) * np.ones((live.sum(), 1)),
                            ((100.0 - t.max(axis=1)) / 100.0)[:, None],
                            onehot], axis=1).astype(np.float32)
        with torch.no_grad():
            out[live] = torch.softmax(model(torch.from_numpy(X)), 1).numpy()
    return out


def band_of(totals):
    t = np.sort(totals)[::-1]
    if t[0] >= 85 and (t[0] - t[1]) <= 10:
        return 'tension'
    if t[0] >= 85:
        return 'runaway'
    return 'early'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--model', default='equity_v1.pth')
    ap.add_argument('--gate', action='store_true',
                    help='emit the spine-gate verdict (halt-default)')
    args = ap.parse_args()

    ck = torch.load(args.model, weights_only=True)
    model = EquityNet(ck['in_dim'])
    model.load_state_dict(ck['state_dict'])
    model.eval()

    df = pd.read_csv(args.log)
    stats = {b: {'n': 0, 'flip_win': 0, 'flip_place': 0,
                 'snr_eq': [], 'snr_dp': []}
             for b in ('tension', 'runaway', 'early')}

    for (dec, seat), g in df.groupby(['decision', 'seat']):
        seat = int(seat)
        totals = g.iloc[0][['t0', 't1', 't2', 't3']].to_numpy(dtype=np.float64)
        deals = int(g.iloc[0]['deals_played'])
        band = band_of(totals)
        scores = g[['s0', 's1', 's2', 's3']].to_numpy(dtype=np.float64)
        actions = g['action_card'].to_numpy()

        eq = equity_values(model, totals[None, :] + scores, deals + 1, seat)
        win_v = eq[:, 0]
        place_v = -(eq * np.array([1, 2, 3, 4])).sum(1)
        dp_v = scores.mean(axis=1) - scores[:, seat]  # relative reward

        per_action = {}
        for a in np.unique(actions):
            sel = actions == a
            k = int(sel.sum())
            per_action[a] = {
                'win': (win_v[sel].mean(), win_v[sel].std(ddof=1) / np.sqrt(k) if k > 1 else np.inf),
                'place': (place_v[sel].mean(), 0.0),
                'dp': (dp_v[sel].mean(), dp_v[sel].std(ddof=1) / np.sqrt(k) if k > 1 else np.inf),
            }
        if len(per_action) < 2:
            continue
        best = {obj: max(per_action, key=lambda a: per_action[a][obj][0])
                for obj in ('win', 'place', 'dp')}
        s = stats[band]
        s['n'] += 1
        s['flip_win'] += int(best['win'] != best['dp'])
        s['flip_place'] += int(best['place'] != best['dp'])
        for obj, key in (('win', 'snr_eq'), ('dp', 'snr_dp')):
            vals = sorted((per_action[a][obj][0] for a in per_action), reverse=True)
            gap = vals[0] - vals[1]
            se = per_action[best[obj]][obj][1]
            if np.isfinite(se) and se > 0:
                s[key].append(gap / se)

    metrics = {}
    print(f"{'band':<9}{'n':>7}{'flip(win)':>11}{'flip(place)':>13}"
          f"{'SNR eq':>9}{'SNR dp':>9}")
    for b, s in stats.items():
        if s['n'] == 0:
            continue
        fw = s['flip_win'] / s['n']
        fp = s['flip_place'] / s['n']
        snr_eq = float(np.median(s['snr_eq'])) if s['snr_eq'] else float('nan')
        snr_dp = float(np.median(s['snr_dp'])) if s['snr_dp'] else float('nan')
        print(f"{b:<9}{s['n']:>7}{fw:>10.1%}{fp:>12.1%}{snr_eq:>9.2f}{snr_dp:>9.2f}")
        metrics[b] = {'n': s['n'], 'flip_win': fw, 'flip_place': fp,
                      'snr_equity': snr_eq, 'snr_dealpoints': snr_dp}

    if args.gate:
        t = metrics.get('tension', {})
        ok = (t.get('n', 0) >= 300
              and t.get('flip_win', 0) >= 0.05
              and t.get('snr_equity', 0) >= 1.0)
        emit_verdict('flip_snr', metrics,
                     {'tension_flip_win_min': 0.05, 'tension_snr_min': 1.0,
                      'tension_n_min': 300}, ok,
                     data_sha=file_sha(args.log)[:16])
        if not ok:
            print("HALT: spine gate failed - do not proceed to C++ integration")
            raise SystemExit(1)
        print("SPINE GATE PASS - C++ integration may proceed")


if __name__ == '__main__':
    main()
