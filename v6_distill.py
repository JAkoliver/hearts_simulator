"""v6 Stage-3 from-scratch distillation trainer (docs/v6_prereg.md).

One registered arm per invocation, trained from scratch on the HMR3
Stage-2 bank (expert_data/v6_bank/). The epochs {2,3,4} x lr grid is
covered by ONE 4-epoch run per lr with a snapshot + holdout metrics
after every epoch (the established .epN convention), so the full grid
is 6 trainings, not 18.

Arms (constructions declared before any training ran):
  a  v6-full:            HeartsNetV6(obs 882, d448, L8, h8)
  b  structure control:  HeartsNetV6(obs 882, d320, L6, h5)  - same
     structure as (a) at v5-M size (~7.4M params), so (a) vs (b)
     isolates scale; h5 = the net_from_checkpoint d//56 convention.
  c  data control:       HeartsNetV5(obs 556, d320, L6, h10) - v5-M
     production shape on obs v1 (the record's 556-dim obs field), no
     new aux heads (belief stays: it is part of the v5-M recipe).

Recipe (the one that BUILT v5-M, per distill.py conventions):
  hard CE on the search-chosen action + value MSE on the match-outcome
  reward + belief BCE (+ v6 aux: moon BCE + seat-points MSE, labels
  rotated from the record's absolute frame to the acting seat's
  relative frame). Fixed coefs across all arms/recipes: policy 1.0,
  value 0.5, belief 0.5, moon 0.5, points 0.5. Adam, batch 512,
  fp32/TF32.

Batch amendment (2026-08-12, before any freeze comparison was made):
the initially-declared 2048 oversubscribed the shared-desktop 4090's
VRAM under Windows/WDDM (measured 7.5 s/step of driver paging; fp32 vs
TF32 epoch metrics were bit-identical because the spill, not matmul
precision, dominated). 512 fits alongside the desktop apps; the entire
grid reruns uniformly at the new batch, so no cross-recipe comparison
ever mixes batch sizes. Epochs x lr (the registered recipe axes) are
untouched.

Holdout: BY-MATCH split, key md5(file, match_id) % holdout_mod == 0
(mod 20 = ~5% of matches), identical across arms/recipes.

Freeze criterion (declared here, before results existed): per arm,
among snapshots with holdout play-phase entropy inside [0.22, 0.87]
(2x both directions of baseline 0.434), pick the lowest holdout policy
CE; ties break toward fewer epochs. Teacher-match reported alongside.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import headroom
from hearts_net import HeartsNetV5, HeartsNetV6
from cloud.shard_check import V3_REC

COEF = {'policy': 1.0, 'value': 0.5, 'belief': 0.5,
        'moon': 0.5, 'points': 0.5}
ENTROPY_BAND = (0.22, 0.87)


def build_net(arm):
    if arm == 'a':
        return HeartsNetV6(d_model=448, num_layers=8, num_heads=8)
    if arm == 'b':
        return HeartsNetV6(d_model=320, num_layers=6, num_heads=5)
    if arm == 'c':
        return HeartsNetV5(obs_dim=556, d_model=320, num_layers=6,
                           num_heads=10)
    raise SystemExit(f'unknown arm {arm}')


def load_bank(pattern, holdout_mod):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'no bank files match {pattern}')
    parts, hold_parts = [], []
    n_match = n_hold_match = 0
    for path in files:
        raw = open(path, 'rb').read()
        assert raw[:4] == b'HMR3', path
        r = np.frombuffer(raw[32:], dtype=V3_REC)
        base = os.path.basename(path)
        hold_ids = set()
        for mid in np.unique(r['match_id']):
            h = int(hashlib.md5(f'{base}:{mid}'.encode()).hexdigest(), 16)
            n_match += 1
            if h % holdout_mod == 0:
                hold_ids.add(int(mid))
                n_hold_match += 1
        if hold_ids:
            hmask = np.isin(r['match_id'], sorted(hold_ids))
            hold_parts.append(r[hmask].copy())
            parts.append(r[~hmask].copy())
        else:
            parts.append(r.copy())
    if not hold_parts:
        # degenerate smoke-run case (tiny file set): hold out the last
        # match of the last file so eval still exercises
        r = parts[-1]
        last_mid = r['match_id'].max()
        hold_parts.append(r[r['match_id'] == last_mid].copy())
        parts[-1] = r[r['match_id'] != last_mid].copy()
        n_hold_match += 1
    train = np.concatenate(parts)
    hold = np.concatenate(hold_parts)
    print(f'bank: {len(files)} files, {n_match} matches -> '
          f'train {len(train)} records / holdout {len(hold)} records '
          f'({n_hold_match} matches)')
    return train, hold


def batch_tensors(b, arm, device):
    u8 = lambda name: torch.from_numpy(
        np.ascontiguousarray(b[name])).to(device)
    if arm == 'c':
        obs = u8('obs').float() / 255.0
    else:
        obs = torch.cat([u8('obs'), u8('ext')], dim=1).float() / 255.0
    mask = u8('mask').bool()
    labels = u8('labels').float()
    actions = torch.from_numpy(b['action'].astype(np.int64)).to(device)
    rewards = torch.from_numpy(b['reward'].astype(np.float32)).to(device)
    out = {'obs': obs, 'mask': mask, 'labels': labels,
           'actions': actions, 'rewards': rewards}
    if arm != 'c':
        seat = b['seat'].astype(np.int64)                     # absolute
        rel = (seat[:, None] + np.arange(4)[None, :]) % 4     # rel k -> abs
        fin = b['fin'].astype(np.float32)
        out['pts'] = torch.from_numpy(
            np.take_along_axis(fin, rel, axis=1) / 26.0).to(device)
        mb = b['moonby'].astype(np.int64)
        moon = (rel == mb[:, None]) & (mb[:, None] >= 0)
        out['moon'] = torch.from_numpy(moon.astype(np.float32)).to(device)
    return out


def forward_losses(net, arm, t):
    if arm == 'c':
        logits, value, belief, _oracle = net.forward_train(
            t['obs'], t['mask'], t['labels'])
        moon_l = pts_l = None
    else:
        logits, value, belief, moon_logits, seat_pts = net.forward_aux(
            t['obs'], t['mask'])
        moon_l = F.binary_cross_entropy_with_logits(moon_logits, t['moon'])
        pts_l = F.mse_loss(seat_pts, t['pts'])
    logp = F.log_softmax(logits, dim=1).masked_fill(~t['mask'], 0.0)
    per_ce = -logp.gather(1, t['actions'].unsqueeze(1)).squeeze(1)
    policy_l = per_ce.mean()
    value_l = F.mse_loss(value.squeeze(-1), t['rewards'])
    belief_l = F.binary_cross_entropy_with_logits(belief, t['labels'])
    loss = (COEF['policy'] * policy_l + COEF['value'] * value_l
            + COEF['belief'] * belief_l)
    if moon_l is not None:
        loss = loss + COEF['moon'] * moon_l + COEF['points'] * pts_l
    return loss, logits, {'ce': policy_l.item(), 'value': value_l.item(),
                          'belief': belief_l.item(),
                          **({'moon': moon_l.item(), 'pts': pts_l.item()}
                             if moon_l is not None else {})}


@torch.no_grad()
def eval_holdout(net, arm, hold, device, batch):
    net.eval()
    n = len(hold)
    sums = {}
    match = seen = 0
    ent_sum = play_seen = 0.0
    for start in range(0, n, batch):
        b = hold[start:start + batch]
        t = batch_tensors(b, arm, device)
        _, logits, parts = forward_losses(net, arm, t)
        k = len(b)
        for key, v in parts.items():
            sums[key] = sums.get(key, 0.0) + v * k
        pred = logits.argmax(dim=1)
        match += (pred == t['actions']).sum().item()
        seen += k
        # entropy band: PLAY-phase decisions only (flags bit0 = pass)
        play = torch.from_numpy(
            ((b['flags'] & 1) == 0)).to(device)
        if play.any():
            p = F.softmax(logits[play], dim=1)
            lp = F.log_softmax(logits[play], dim=1).masked_fill(
                ~t['mask'][play], 0.0)
            ent = -(p * lp).sum(dim=1)
            ent_sum += ent.sum().item()
            play_seen += int(play.sum().item())
    net.train()
    out = {k: v / seen for k, v in sums.items()}
    out['teacher_match'] = match / seen
    out['entropy_play'] = ent_sum / max(play_seen, 1)
    out['in_band'] = ENTROPY_BAND[0] <= out['entropy_play'] <= ENTROPY_BAND[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['a', 'b', 'c'])
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--data', default='expert_data/v6_bank/gen_*_t*.bin')
    ap.add_argument('--holdout-mod', type=int, default=20)
    ap.add_argument('--out', required=True,
                    help='checkpoint prefix; .epN.pth per epoch + .json')
    ap.add_argument('--seed', type=int, default=20260812)
    args = ap.parse_args()

    headroom.apply_process_priority()
    headroom.banner()
    # TF32 matmul, uniform across ALL grid runs (declared before the
    # grid ran; plain fp32 measured 89 min/epoch on arm a = ~20h grid,
    # far past the prereg cost note; TF32 changes tensor-core matmul
    # precision only, identically for every arm/recipe)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train, hold = load_bank(args.data, args.holdout_mod)
    net = build_net(args.arm).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'arm {args.arm}: {type(net).__name__} '
          f'{n_params / 1e6:.2f}M params, lr {args.lr}, device {device}')
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    results = {'arm': args.arm, 'lr': args.lr, 'batch': args.batch,
               'params': n_params, 'coefs': COEF, 'seed': args.seed,
               'train_records': len(train), 'holdout_records': len(hold),
               'epochs': {}}
    n = len(train)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(n)
        run_loss = seen = 0
        for start in range(0, n, args.batch):
            headroom.pace()
            b = train[perm[start:start + args.batch]]
            t = batch_tensors(b, args.arm, device)
            loss, _, _ = forward_losses(net, args.arm, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(b)
            seen += len(b)
        m = eval_holdout(net, args.arm, hold, device, args.batch)
        m['train_loss'] = run_loss / seen
        m['seconds'] = round(time.time() - t0, 1)
        results['epochs'][epoch] = m
        print(f"epoch {epoch}: train {m['train_loss']:.4f} | holdout CE "
              f"{m['ce']:.4f} match {m['teacher_match']:.4f} entropy "
              f"{m['entropy_play']:.3f} ({'IN' if m['in_band'] else 'OUT OF'}"
              f" band) | value {m['value']:.4f} belief {m['belief']:.4f}"
              + (f" moon {m['moon']:.4f} pts {m['pts']:.5f}"
                 if 'moon' in m else '')
              + f" | {m['seconds']}s")
        if epoch >= 2:
            torch.save(net.state_dict(), f'{args.out}.ep{epoch}.pth')
    with open(f'{args.out}.json', 'w') as f:
        json.dump(results, f, indent=1)
    print(f'done: snapshots {args.out}.ep2/3/4.pth + {args.out}.json')


if __name__ == '__main__':
    main()
