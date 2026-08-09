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

EXPLORATION MODE (--exploration): halt-report probes only (the 08-09
drift-screen halt; pattern = expert-iter v2 freeze exploration:
holdout-only, pre-amendment, never gate-eligible). Unlocks values
outside the registered sets plus:
  --epoch-budget N   epoch = fixed N-sample budget; BOTH streams cycle
                     (share then truly controls defense-gradient dose)
  --anchor-loss kl   anchor rows: KL(candidate || frozen baseline full
                     distribution) instead of CE-to-argmax. At init the
                     KL and its gradient are exactly 0 (truly neutral),
                     while argmax-CE actively SHARPENS the baseline
                     toward its own argmax from step one.
  --kl-coef W        weight of the anchor-KL row sum vs defense CE
Candidates from exploration runs must never be gated; they exist to
pick the ONE registered amendment for the second (final) grid.

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
REG_SHARES = (0.75, 0.875)
REG_LRS = (1e-5, 3e-5)
# Amendment 2026-08-09 (signed): KL anchor is REGISTERED at these
# coefficients for the final grid (prereg amendment section).
REG_KL_COEFS = (4.0, 8.0)


def md5_8(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


class Cycler:
    """Reshuffle-when-exhausted index stream over a pool."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.perm = rng.permutation(n)
        self.pos = 0

    def take(self, k):
        out = []
        while k > 0:
            if self.pos >= self.n:
                self.perm = self.rng.permutation(self.n)
                self.pos = 0
            grab = min(k, self.n - self.pos)
            out.append(self.perm[self.pos:self.pos + grab])
            self.pos += grab
            k -= grab
        return np.concatenate(out)


def to_dev(arr, idx, device):
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
        obs, mask = to_dev(arr, idx, device)
        logits, _, _ = net.forward_all(obs, mask)
        tgt = torch.from_numpy(
            arr[target_key][idx].astype(np.int64)).to(device)
        hits += int((logits.argmax(1) == tgt).sum().item())
    net.train()
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='b2_data')
    ap.add_argument('--anchor-share', type=float, required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--train-seed', type=int, default=20260809)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exploration', action='store_true',
                    help='halt-report probe: unlock off-registry values '
                         'and the flags below; NEVER gate-eligible')
    ap.add_argument('--epoch-budget', type=int, default=None,
                    help='epoch = this many total samples (both streams '
                         'cycle); requires --exploration')
    ap.add_argument('--anchor-loss', choices=['ce', 'kl'], default='ce')
    ap.add_argument('--kl-coef', type=float, default=1.0)
    args = ap.parse_args()
    assert 0 <= args.epochs <= 3, 'prereg: <= 3 epochs'
    if not args.exploration:
        assert args.anchor_share in REG_SHARES, 'off-registry share needs --exploration'
        assert args.lr in REG_LRS, 'off-registry lr needs --exploration'
        assert args.epoch_budget is None, 'budget-epoch needs --exploration'
        assert args.anchor_loss == 'ce' or args.kl_coef in REG_KL_COEFS, \
            'off-registry kl-coef needs --exploration (amendment 2026-08-09 ' \
            'registers KL at coefs {4.0, 8.0})'
    else:
        print('EXPLORATION RUN (halt report only - never gate-eligible)')

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

    anchor_net = None
    if args.anchor_loss == 'kl':
        anchor_net = net_from_checkpoint(BASELINE).to(device).eval()
        for p in anchor_net.parameters():
            p.requires_grad_(False)

    nd = len(dtr['obs'])
    na = len(atr['obs'])
    n_def_rows = max(1, round(args.batch * (1.0 - args.anchor_share)))
    n_anc_rows = args.batch - n_def_rows
    if args.epoch_budget is not None:
        steps_per_epoch = max(1, args.epoch_budget // args.batch)
    else:
        steps_per_epoch = (nd + n_def_rows - 1) // n_def_rows
    print(f'defense {nd:,} rows, anchor pool {na:,}; batch '
          f'{n_def_rows}+{n_anc_rows} (share {args.anchor_share}); '
          f'{steps_per_epoch} steps/epoch '
          f'({steps_per_epoch * n_def_rows:,} defense samples/epoch); '
          f'anchor loss {args.anchor_loss}'
          + (f' coef {args.kl_coef}' if args.anchor_loss == 'kl' else ''))

    dcyc, acyc = Cycler(nd, rng), Cycler(na, rng)
    for epoch in range(1, args.epochs + 1):
        ce_sum = kl_sum = seen = 0
        for _ in range(steps_per_epoch):
            headroom.pace()
            didx = dcyc.take(n_def_rows)
            aidx = acyc.take(n_anc_rows)

            obs = torch.from_numpy(np.concatenate([
                np.ascontiguousarray(dtr['obs'][didx]),
                np.ascontiguousarray(atr['obs'][aidx])])).to(device)
            mask = torch.from_numpy(np.concatenate([
                np.ascontiguousarray(dtr['mask'][didx]),
                np.ascontiguousarray(atr['mask'][aidx])])).to(device).bool()

            logits, _, _ = net.forward_all(obs, mask)
            logp = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
            def_tgt = torch.from_numpy(
                dtr['action'][didx].astype(np.int64)).to(device)
            def_ce = -logp[:n_def_rows].gather(
                1, def_tgt.unsqueeze(1)).squeeze(1)

            if args.anchor_loss == 'ce':
                anc_tgt = torch.from_numpy(
                    atr['base_argmax'][aidx].astype(np.int64)).to(device)
                anc_term = -logp[n_def_rows:].gather(
                    1, anc_tgt.unsqueeze(1)).squeeze(1)
                coef = 1.0
            else:
                amask = mask[n_def_rows:]
                with torch.no_grad():
                    alogits, _, _ = anchor_net.forward_all(
                        obs[n_def_rows:], amask)
                alogp = F.log_softmax(alogits, dim=1).masked_fill(~amask, 0.0)
                clogp = logp[n_def_rows:]
                cp = F.softmax(logits[n_def_rows:], dim=1)
                anc_term = (cp * (clogp - alogp)).sum(dim=1)
                coef = args.kl_coef

            loss = (def_ce.sum() + coef * anc_term.sum()) / args.batch
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            ce_sum += float(def_ce.mean().item()) * len(didx)
            kl_sum += float(anc_term.mean().item()) * len(aidx)
            seen += len(didx)

        dm = eval_agreement(net, dho, 'action', device)
        da = eval_agreement(net, drf, 'base_argmax', device)
        ck = f'{args.out}.ep{epoch}.pth'
        torch.save(net.state_dict(), ck)
        print(f'epoch {epoch}: defCE {ce_sum / seen:.4f}  '
              f'anchor[{args.anchor_loss}] {kl_sum / (seen or 1):.4f}  '
              f'defense-holdout match {dm:.4f}  '
              f'drift-holdout agreement {da:.4f}  -> {ck}')

    torch.save(net.state_dict(), args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
