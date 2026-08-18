"""League round 5, Addendum W: WARM-START the obs-v2 adapters of the champion
(docs/exploiter_league_r5_prereg.md §9).

Round 5 showed zero-init adapters never become a pathway inside the
champion-regime anchored PPO (mean |w| ~30x below the trunk's after 250k
deals). This script gives ONLY the two adapter projections a head start:
every trunk/head weight of 8a89da90 is FROZEN; `ext_card_proj` and
`ext_ctx_proj` are trained by supervised imitation of the search-chosen
action (plain CE) on the v6 bank's TRAINING records (the frozen mod-20
holdout is never touched), starting from cand_r5_ext_init.pth.

Registered acceptance (run with --accept after training; any failure ->
ONE tune of epoch/lr, else HALT):
  (a) teacher-match on the frozen full holdout >= champion's own - 0.2 pp
  (b) threat-dead drift vs the champion (drift_screen_v6holdout) <= 3 %
  (c) paired strength vs the champion (neutral_raw n=5000): UB95 <= +0.30
Also reports adapter |w| relative to the trunk (informs).

Usage:
  python warmstart_v5ext.py --epochs 1 --lr 1e-3 --out cand_r5_ext_warm.pth
  python warmstart_v5ext.py --accept cand_r5_ext_warm.pth   (a) + (b) + |w|;
      (c) is run by the driver/user via neutral_raw_eval.py (12 workers).
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import headroom
import v6_distill as vd
from hearts_net import net_from_checkpoint
from v6_probe_eval import walk_holdout

INIT = 'cand_r5_ext_init.pth'
CHAMPION = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'


def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()


def train(args):
    headroom.apply_process_priority(); headroom.banner()
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert md5(INIT)[:8] == '9a1fb86f', 'ext-init checkpoint md5 mismatch'
    train_recs, hold = vd.load_bank('expert_data/v6_bank/gen_*_t*.bin', 20)
    net = net_from_checkpoint(INIT).to(device)
    assert type(net).__name__ == 'HeartsNetV5Ext'
    for n_, p_ in net.named_parameters():
        p_.requires_grad = n_.startswith('ext_card_proj') or n_.startswith('ext_ctx_proj')
    params = [p_ for p_ in net.parameters() if p_.requires_grad]
    print(f'trainable adapter params: {sum(p.numel() for p in params)} of '
          f'{sum(p.numel() for p in net.parameters())}')
    opt = torch.optim.Adam(params, lr=args.lr)
    net.train()
    # trunk in eval mode is irrelevant (no dropout/BN in V5); keep train() for autograd
    n = len(train_recs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); perm = np.random.permutation(n); run = seen = 0
        for start in range(0, n, args.batch):
            headroom.pace()
            b = train_recs[perm[start:start + args.batch]]
            t = vd.batch_tensors(b, 'a', device)          # obs (882), mask, actions
            logits, _, _ = net.forward_all(t['obs'], t['mask'])
            logp = F.log_softmax(logits, dim=1).masked_fill(~t['mask'], 0.0)
            loss = -logp.gather(1, t['actions'].unsqueeze(1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(b); seen += len(b)
        print(f'epoch {epoch}: adapter-only CE {run / seen:.4f} | {time.time() - t0:.0f}s')
    net.eval()
    torch.save({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, args.out)
    print(f'saved {args.out} md5 {md5(args.out)[:8]}')


@torch.no_grad()
def accept(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hold, keys = walk_holdout()
    res = {'candidate': path, 'md5': md5(path)}
    # (a) teacher-match vs the champion's own (both on the frozen full holdout)
    def tm(net_path):
        net = net_from_checkpoint(net_path).to(device).eval()
        dim = 882 if getattr(net, 'obs_dim', 556) == 882 else 556
        obs = torch.from_numpy(np.concatenate([np.ascontiguousarray(hold['obs']),
                                               np.ascontiguousarray(hold['ext'])], 1)).float() / 255.0
        mask = torch.from_numpy(np.ascontiguousarray(hold['mask'])).bool()
        acts = torch.from_numpy(hold['action'].astype(np.int64))
        ok = 0; ce = 0.0
        for s in range(0, len(obs), 2048):
            o = obs[s:s + 2048, :dim].to(device); m = mask[s:s + 2048].to(device); a_ = acts[s:s + 2048].to(device)
            logits, _, _ = net.forward_all(o, m)
            logp = F.log_softmax(logits, dim=1).masked_fill(~m, 0.0)
            ce += float(-logp.gather(1, a_.unsqueeze(1)).sum()); ok += int((logits.argmax(1) == a_).sum())
        return ok / len(obs), ce / len(obs)
    tm_c, ce_c = tm(path); tm_0, ce_0 = tm(CHAMPION)
    res['teacher_match_candidate'] = tm_c; res['teacher_match_champion'] = tm_0
    res['ce_candidate'] = ce_c; res['ce_champion'] = ce_0
    res['a_pass'] = tm_c >= tm_0 - 0.002
    import os
    # (b) drift on the shared instrument
    import subprocess, sys
    out = subprocess.run([sys.executable, 'drift_screen_v6holdout.py', path, '--json', 'accept_tmp_drift.json'],
                         capture_output=True, text=True)
    d = json.load(open('accept_tmp_drift.json')); os.remove('accept_tmp_drift.json')
    res['drift_threat_dead'] = d['drift_threat_dead']; res['drift_other_play'] = d['drift_other_play']
    res['b_pass'] = d['drift_threat_dead'] <= 0.03
    # adapter magnitude vs trunk (informs)
    net = net_from_checkpoint(path)
    res['adapter_mean_abs_w'] = float((net.ext_card_proj.weight.abs().mean() + net.ext_ctx_proj.weight.abs().mean()) / 2)
    res['trunk_card_proj_mean_abs_w'] = float(net.card_proj.weight.abs().mean())
    print(json.dumps(res, indent=1))
    print('ACCEPT (a)+(b):', 'PASS' if res['a_pass'] and res['b_pass'] else 'FAIL',
          '- run (c) neutral_raw_eval.py --cand', path, '--base', CHAMPION, '--deals 5000')
    json.dump(res, open('equity_data/verdicts/r5W_warmstart_accept.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=20260818)
    ap.add_argument('--out', default='cand_r5_ext_warm.pth')
    ap.add_argument('--accept', default=None)
    a = ap.parse_args()
    if a.accept: accept(a.accept)
    else: train(a)
