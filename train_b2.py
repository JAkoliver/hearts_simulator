"""Round-2 Phase B2 anchored distillation trainer
(docs/exploiter_league_r2_prereg.md).

Fine-tunes the PROMOTED baseline (md5-verified 8a89da90) with hard-CE
on a per-batch mixture:
  defense rows: target = the search defender's recorded choice
                (moon-alive plays + passes, from b2_data/def_train.npz)
  anchor rows:  target = the baseline's own precomputed argmax
                (ordinary decisions, from b2_data/anchor_train.npz)

Registered knobs (prereg Phase B2): --anchor-share {0.75, 0.875},
--lr {1e-5, 3e-5}, --epochs <= 3. Loss is plain masked CE over the
combined batch, so the anchor share IS the loss weighting ("mixture per
batch"). Policy head only (prereg: "hard-CE on chosen actions"); value/
belief heads move only via the trunk - the search guard owns that risk.

Epoch definition (implementation of "mixture per batch"): one epoch =
one shuffled pass over the DEFENSE stream; anchor rows are drawn from a
reshuffle-when-exhausted cycle of the anchor pool to fill each batch to
its share. Obs are raw float32 (NO /255 - that convention belongs to
selfplay_gen banks, not these records).

Per-epoch: checkpoint saved (<out>.ep<N>.pth) + quick telemetry
(defense-holdout teacher-match, drift-holdout agreement vs stored
baseline argmax). --epochs 0 writes an unmodified copy of the baseline
(null test: the drift screen must then read agreement == 1.0 exactly).
"""
import argparse
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import headroom
from hearts_net import net_from_checkpoint

BASELINE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
BASELINE_MD5_8 = '8a89da90'


def md5_8(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def batches_to_device(arr, idx, device):
    obs = torch.from_numpy(np.ascontiguousarray(arr['obs'][idx])).to(device)
    mask = torch.from_numpy(
        np.ascontiguousarray(arr['mask'][idx])).to(device).bool()
    return obs, mask


@torch.no_grad()
def eval_agreement(net, arr, target_key, device, batch=4096):
    net.eval()
    hits = 0
    n = len(arr['obs'])
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        obs, mask = batches_to_device(arr, idx, device)
        logits, _, _ = net.forward_all(obs, mask)
        tgt = torch.from_numpy(
            arr[target_key][idx].astype(np.int64)).to(device)
        hits += int((logits.argmax(1) == tgt).sum().item())
    net.train()
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='b2_data')
    ap.add_argument('--anchor-share', type=float, required=True,
                    choices=[0.75, 0.875])
    ap.add_argument('--lr', type=float, required=True,
                    choices=[1e-5, 3e-5])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--train-seed', type=int, default=20260809)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert 0 <= args.epochs <= 3, 'prereg: <= 3 epochs'

    assert md5_8(BASELINE) == BASELINE_MD5_8, \
        f'{BASELINE} md5 != {BASELINE_MD5_8} - refusing to train'
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    rng = np.random.default_rng(args.train_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    headroom.apply_process_priority()
    torch.set_float32_matmul_precision('high')

    dtr = load_npz(os.path.join(args.data_dir, 'def_train.npz'))
    atr = load_npz(os.path.join(args.data_dir, 'anchor_train.npz'))
    dho = load_npz(os.path.join(args.data_dir, 'def_holdout.npz'))
    drf = load_npz(os.path.join(args.data_dir, 'drift_holdout.npz'))

    net = net_from_checkpoint(BASELINE).to(device)
    if args.epochs == 0:   # null test: unmodified baseline copy
        torch.save(net.state_dict(), args.out)
        print(f'null run: wrote unmodified baseline to {args.out}')
        return
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    nd = len(dtr['obs'])
    na = len(atr['obs'])
    n_def_rows = max(1, round(args.batch * (1.0 - args.anchor_share)))
    n_anc_rows = args.batch - n_def_rows
    print(f'defense {nd:,} rows, anchor pool {na:,}; batch '
          f'{n_def_rows}+{n_anc_rows} (anchor share {args.anchor_share})')

    anc_perm = rng.permutation(na)
    anc_pos = 0
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(nd)
        ce_sum = seen = 0
        for start in range(0, nd, n_def_rows):
            headroom.pace()
            didx = perm[start:start + n_def_rows]
            if anc_pos + n_anc_rows > na:
                anc_perm = rng.permutation(na)
                anc_pos = 0
            aidx = anc_perm[anc_pos:anc_pos + n_anc_rows]
            anc_pos += n_anc_rows

            obs = torch.from_numpy(np.concatenate([
                np.ascontiguousarray(dtr['obs'][didx]),
                np.ascontiguousarray(atr['obs'][aidx])])).to(device)
            mask = torch.from_numpy(np.concatenate([
                np.ascontiguousarray(dtr['mask'][didx]),
                np.ascontiguousarray(atr['mask'][aidx])])).to(device).bool()
            tgt = torch.from_numpy(np.concatenate([
                dtr['action'][didx],
                atr['base_argmax'][aidx]]).astype(np.int64)).to(device)

            logits, _, _ = net.forward_all(obs, mask)
            logp = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
            loss = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            ce_sum += loss.item() * len(didx)
            seen += len(didx)

        dm = eval_agreement(net, dho, 'action', device)
        da = eval_agreement(net, drf, 'base_argmax', device)
        ck = f'{args.out}.ep{epoch}.pth'
        torch.save(net.state_dict(), ck)
        print(f'epoch {epoch}: CE {ce_sum / seen:.4f}  '
              f'defense-holdout match {dm:.4f}  '
              f'drift-holdout agreement {da:.4f}  -> {ck}')

    torch.save(net.state_dict(), args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
