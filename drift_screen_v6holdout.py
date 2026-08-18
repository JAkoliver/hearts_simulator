"""League round 5 drift instrument (docs/exploiter_league_r5_prereg.md §3.5).

drift_screen_b2.py measures ordinary-state drift on the r2 b2 bank, whose
records carry the 556-dim obs only - an obs-v2 (882) candidate cannot
consume it. This instrument measures the SAME quantity - argmax agreement
with the frozen champion 8a89da90 on THREAT-DEAD states (>= 2 seats hold
round points, obs dims 156-159) - on the v6 bank's frozen full holdout
(80,220 records / 123 matches, obs v2 rows), so control (556) and adapter
(882) cells are read on one scale. Candidate nets consume 882 or 556 by
their obs_dim; the champion always sees the 556 prefix.

Calibration: run on cand_r4_A.pth (7.9% drift on the b2 bank) and report
both numbers side by side; the two banks differ, so only the ORDERING and
the band (5-15%) transfer, not the value.

Usage: python drift_screen_v6holdout.py <candidate.pth> [--json out.json]
"""
import argparse
import json

import numpy as np
import torch

from hearts_net import net_from_checkpoint
from v6_probe_eval import walk_holdout

CHAMPION = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'


@torch.no_grad()
def argmax_on(net, obs, mask, device, batch=1024):
    net.eval().to(device)
    # 882 for adapter nets; EVERY other net gets the full 556 prefix (obs
    # 550 + match ctx 6) - net_from_checkpoint builds V5 nets with the
    # default obs_dim=550, which must NOT be used to slice (it would drop
    # the match ctx and read a bit-identical net as ~1.7% 'drift').
    dim = 882 if getattr(net, 'obs_dim', 556) == 882 else 556
    out = []
    for s in range(0, len(obs), batch):
        o = obs[s:s + batch, :dim].to(device)
        m = mask[s:s + batch].to(device)
        logits, _ = net(o, m)
        out.append(logits.argmax(dim=1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hold, keys = walk_holdout()
    obs = torch.from_numpy(np.concatenate(
        [np.ascontiguousarray(hold['obs']), np.ascontiguousarray(hold['ext'])],
        axis=1)).float() / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(hold['mask'])).bool()
    play = torch.from_numpy(((hold['flags'] & 1) == 0))
    scores = obs[:, 156:160]
    dead = ((scores > 1e-6).sum(1) >= 2) & play          # threat-dead PLAY states
    alive = (~((scores > 1e-6).sum(1) >= 2)) & play      # everything else (informs)
    champ = net_from_checkpoint(CHAMPION)
    cand = net_from_checkpoint(a.candidate)
    ac = argmax_on(champ, obs, mask, device)
    ak = argmax_on(cand, obs, mask, device)
    agree_dead = float((ac[dead] == ak[dead]).float().mean())
    agree_alive = float((ac[alive] == ak[alive]).float().mean())
    res = {'candidate': a.candidate, 'candidate_obs_dim': getattr(cand, 'obs_dim', 556),
           'n_dead': int(dead.sum()), 'n_alive': int(alive.sum()),
           'agreement_threat_dead': agree_dead, 'drift_threat_dead': 1 - agree_dead,
           'agreement_other_play': agree_alive, 'drift_other_play': 1 - agree_alive,
           'band_5_15_informs': 0.05 <= 1 - agree_dead <= 0.15}
    print(f"{a.candidate}: threat-dead drift {1-agree_dead:.4f} (n={int(dead.sum())}) | "
          f"other-play drift {1-agree_alive:.4f} (n={int(alive.sum())}) | "
          f"cand obs_dim {res['candidate_obs_dim']}")
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
