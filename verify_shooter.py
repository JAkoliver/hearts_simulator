"""Phase B quality bar (docs/exploiter_league_prereg.md, halt-default):
seat a DISTILLED shooter clone against 3x the frozen baseline and measure
its moon rate. The clone only has to carry the THREAT, not match the
search-shooter's strength: the registered bar is

    clone moon rate >= 50% of the search-shooter's Phase A rate
        agg: 0.5148/deal -> bar 0.2574
        sel: 0.3666/deal -> bar 0.1833

Phase A measured those with the frozen probe (shooter_v1, K=64 flat), so
this verification uses the same defender field and the same seat rotation
- only the attacker changes.

Usage: python verify_shooter.py --clone shooter_agg_v1.pth --mode agg
                                [--matches 120 --seed 130000000]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from hearts_match_env import MatchEnv, TARGET
from hearts_net import HeartsNetV5, net_from_checkpoint

PHASE_A_RATE = {'agg': 0.514792899408284, 'sel': 0.3665699782451051}
BAR_FRACTION = 0.5


def load_clone(path):
    ck = torch.load(path, weights_only=True, map_location='cpu')
    net = HeartsNetV5(obs_dim=556, d_model=ck['d_model'],
                      num_layers=ck['num_layers'],
                      num_heads=ck.get('num_heads', 6))
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, ck


def act(net, obs, legal, dev):
    o = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(dev)
    m = torch.zeros((1, 52), dtype=torch.bool)
    for a in legal:
        if a != -1:
            m[0, a] = True
    with torch.no_grad():
        lg, _ = net(o, m.to(dev))
    return int(lg.argmax(1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clone', required=True)
    ap.add_argument('--mode', required=True, choices=('agg', 'sel'))
    ap.add_argument('--defender', default='hearts_model_final.pth')
    ap.add_argument('--matches', type=int, default=120)
    ap.add_argument('--seed', type=int, default=130_000_000)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    clone, ck = load_clone(args.clone)
    clone.to(dev)
    defender = net_from_checkpoint(args.defender)
    defender.eval().to(dev)
    bar = BAR_FRACTION * PHASE_A_RATE[args.mode]
    print(f'clone {args.clone} (holdout match {ck.get("holdout_match", 0):.3f}) '
          f'vs 3x {args.defender}')
    print(f'bar: >= {bar:.4f} moons/deal ({BAR_FRACTION:.0%} of the Phase A '
          f'search-shooter rate {PHASE_A_RATE[args.mode]:.4f})')

    deals = moons = 0
    t0 = time.time()
    for mi in range(args.matches):
        seat = mi % 4
        menv = MatchEnv(seed=args.seed + mi * 1000)
        while True:
            p = menv.get_current_player()
            legal = menv.get_legal_actions()
            a = act(clone if p == seat else defender, menv.observe(), legal, dev)
            deal_done, match_done, rs = menv.step(a)
            if deal_done:
                deals += 1
                rs = np.asarray(rs)
                if rs.sum() == 78 and rs[seat] == 0:
                    moons += 1
            if match_done:
                break
        if (mi + 1) % 20 == 0:
            print(f'  match {mi+1}/{args.matches}: {deals} deals, {moons} moons '
                  f'({moons/max(1,deals):.3f}/deal, {time.time()-t0:.0f}s)')

    rate = moons / max(1, deals)
    lo = rate - 1.96 * (rate * (1 - rate) / max(1, deals)) ** 0.5
    ok = rate >= bar
    print(f'RESULT {args.mode}: {moons}/{deals} = {rate:.4f}/deal '
          f'(95% lower {lo:.4f}) vs bar {bar:.4f} -> {"PASS" if ok else "HALT"}')
    os.makedirs('equity_data/verdicts', exist_ok=True)
    json.dump({'mode': args.mode, 'clone': args.clone,
               'holdout_match': ck.get('holdout_match'),
               'matches': args.matches, 'deals': deals, 'moons': moons,
               'rate': rate, 'ci95_lower': lo, 'bar': bar, 'pass': bool(ok),
               'phase_a_rate': PHASE_A_RATE[args.mode], 'seed': args.seed},
              open(f'equity_data/verdicts/shooter_{args.mode}_v1_quality.json', 'w'),
              indent=1)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
