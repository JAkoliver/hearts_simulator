"""Supervised distillation trainer for expert iteration.

Trains the network to imitate the search teacher's action-value distribution
(policy head, soft-target cross-entropy), predict round outcomes (value head,
MSE), and predict hidden hands (belief head, BCE) from SelfPlayGen binary
records. Warm-starts from the
current baseline so each iteration refines rather than restarts.

Usage:
  python distill.py --data "selfplay_data/iter_0/*.bin" \
      --init hearts_model_final.pth --out hearts_model_candidate.pth --epochs 2
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hearts_net import HeartsNet, net_from_checkpoint

RECORD = np.dtype([
    ('obs', 'u1', 550), ('mask', 'u1', 52), ('labels', 'u1', 156),
    ('pi', 'u1', 52),
    ('action', '<u2'), ('seat', '<u2'), ('reward', '<f4'),
])
assert RECORD.itemsize == 818

# Leaf value records (SelfPlayGen --value-out): trick-boundary observations
# from EVERY seat's perspective with the seat's true opponent hands and final
# outcome - the distribution value-bootstrapped search queries at truncated
# rollout leaves. The hands train the ORACLE head (at a real leaf the
# determinized hands take their place).
LEAF_RECORD = np.dtype([('obs', 'u1', 550), ('hands', 'u1', 156), ('reward', '<f4')])
assert LEAF_RECORD.itemsize == 710

def load_data(patterns, dtype=RECORD, kind='decision'):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit(f"No data files match {patterns}")
    for f in sorted(files):
        if os.path.getsize(f) % dtype.itemsize != 0:
            raise SystemExit(
                f"{f}: size is not a multiple of {dtype.itemsize} - wrong or "
                f"stale record format? Regenerate the data.")
    chunks = [np.fromfile(f, dtype=dtype) for f in sorted(files)]
    data = np.concatenate(chunks)
    print(f"Loaded {len(data):,} {kind} records from {len(files)} files")
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--init', default='hearts_model_final.pth')
    ap.add_argument('--out', default='hearts_model_candidate.pth')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--policy-coef', type=float, default=1.0)
    ap.add_argument('--value-coef', type=float, default=0.5)
    ap.add_argument('--aux-coef', type=float, default=0.5)
    ap.add_argument('--oracle-coef', type=float, default=0.5,
                    help='weight of the oracle value head loss (predicts the '
                         'outcome GIVEN the true hands; leaf evaluator for '
                         'determinized search)')
    ap.add_argument('--width', type=int, default=512,
                    help='trunk width for fresh networks (warm starts infer '
                         'their size from the checkpoint)')
    ap.add_argument('--blocks', type=int, default=3,
                    help='residual blocks for fresh networks')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sharpen', type=float, default=1.0,
                    help='exponent applied to the teacher target (pi^s, renormalized); '
                         '>1 makes small search preferences decisive, equivalent to '
                         'lowering the generation temperature after the fact')
    ap.add_argument('--holdout', type=float, default=0.02,
                    help='fraction of records held out to report unfit teacher match')
    ap.add_argument('--leaf-data', nargs='+', default=None,
                    help='leaf value record files (SelfPlayGen --value-out); trains '
                         'the value head on the truncated-rollout leaf distribution '
                         'via interleaved value-only batches')
    ap.add_argument('--leaf-coef', type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    data = load_data(args.data)
    leaf = None
    if args.leaf_data:
        leaf = load_data(args.leaf_data, dtype=LEAF_RECORD, kind='leaf-value')
        leaf_var = float(np.var(leaf['reward'])) + 1e-8

    # Fixed held-out split: measures whether the student generalizes to
    # teacher decisions it never trained on (argmax match is only meaningful
    # unfit).
    split_rng = np.random.default_rng(12345)
    order = split_rng.permutation(len(data))
    n_hold = int(len(data) * args.holdout)
    holdout = data[order[:n_hold]]
    data = data[order[n_hold:]]
    if n_hold:
        print(f"Holding out {n_hold:,} records for unfit metrics")

    if args.init and os.path.exists(args.init):
        net = net_from_checkpoint(args.init)
        print(f"Warm start from {args.init} "
              f"(width {net.input_fc.out_features}, {len(net.blocks)} blocks)")
    else:
        net = HeartsNet(width=args.width, num_blocks=args.blocks)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"Fresh network: width {args.width}, {args.blocks} blocks, "
              f"{n_params / 1e6:.2f}M params")
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    n = len(data)
    reward_var = float(np.var(data['reward'])) + 1e-8

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        ce_sum = match_sum = err2_sum = bce_sum = oerr2_sum = 0.0
        seen = 0
        if leaf is not None:
            lperm = np.random.permutation(len(leaf))
            lpos = 0
            lerr2_sum = loerr2_sum = 0.0
            lseen = 0

        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            b = data[idx]
            # Ship the u8 fields to the device raw and convert there: 4x less
            # PCIe traffic than converting to float32 on the host first.
            # (ascontiguousarray: struct fields are strided views of the
            # 818-byte records)
            u8 = lambda name: torch.from_numpy(np.ascontiguousarray(b[name])).to(device)
            obs = u8('obs').float() / 255.0
            mask = u8('mask').bool()
            labels = u8('labels').float()
            pi = u8('pi').float()
            actions = torch.from_numpy(b['action'].astype(np.int64)).to(device)
            rewards = torch.from_numpy(b['reward'].astype(np.float32)).to(device)
            pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-6)
            if args.sharpen != 1.0:
                pi = pi.pow(args.sharpen)
                pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-6)

            logits, value, belief, oracle = net.forward_train(obs, mask, labels)
            # Soft-target cross-entropy against the teacher's value-derived
            # distribution. Illegal logits are -inf; zero them in log-space
            # (their pi is 0) so 0 * -inf can't poison the loss with NaN.
            logp = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
            policy_loss = -(pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), rewards)
            belief_loss = F.binary_cross_entropy_with_logits(belief, labels)
            oracle_loss = F.mse_loss(oracle.squeeze(-1), rewards)
            loss = (args.policy_coef * policy_loss + args.value_coef * value_loss
                    + args.aux_coef * belief_loss + args.oracle_coef * oracle_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            k = len(idx)
            seen += k
            ce_sum += policy_loss.item() * k
            match_sum += (logits.argmax(1) == actions).float().sum().item()
            err2_sum += value_loss.item() * k
            bce_sum += belief_loss.item() * k
            oerr2_sum += oracle_loss.item() * k

            # Interleaved value-only batch on the leaf distribution
            if leaf is not None:
                if lpos + args.batch > len(lperm):
                    lperm = np.random.permutation(len(leaf))
                    lpos = 0
                lidx = lperm[lpos:lpos + args.batch]
                lpos += args.batch
                lb = leaf[lidx]
                lobs = torch.from_numpy(np.ascontiguousarray(lb['obs'])).to(device).float() / 255.0
                lhands = torch.from_numpy(np.ascontiguousarray(lb['hands'])).to(device).float()
                lmask = torch.ones((len(lidx), 52), dtype=torch.bool, device=device)
                lrew = torch.from_numpy(lb['reward'].astype(np.float32)).to(device)
                _, lvalue, _, loracle = net.forward_train(lobs, lmask, lhands)
                leaf_mse = F.mse_loss(lvalue.squeeze(-1), lrew)
                leaf_oracle_mse = F.mse_loss(loracle.squeeze(-1), lrew)
                lloss = args.leaf_coef * (leaf_mse + leaf_oracle_mse)
                optimizer.zero_grad()
                lloss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                lerr2_sum += leaf_mse.item() * len(lidx)
                loerr2_sum += leaf_oracle_mse.item() * len(lidx)
                lseen += len(lidx)

        ev = 1.0 - (err2_sum / seen) / reward_var
        oev = 1.0 - (oerr2_sum / seen) / reward_var
        line = (f"epoch {epoch + 1}/{args.epochs} | policy CE {ce_sum / seen:.4f} | "
                f"teacher match {match_sum / seen * 100:.1f}% | value EV {ev:.3f} | "
                f"oracle EV {oev:.3f} | belief BCE {bce_sum / seen:.4f}")
        if leaf is not None and lseen:
            line += (f" | leaf EV {1.0 - (lerr2_sum / lseen) / leaf_var:.3f}"
                     f" | leaf oracle EV {1.0 - (loerr2_sum / lseen) / leaf_var:.3f}")
        if n_hold:
            with torch.no_grad():
                hm = 0
                herr2 = hbce = 0.0
                for start in range(0, n_hold, args.batch):
                    hb = holdout[start:start + args.batch]
                    hobs = torch.from_numpy(np.ascontiguousarray(hb['obs'])).to(device).float() / 255.0
                    hmask = torch.from_numpy(np.ascontiguousarray(hb['mask'])).to(device).bool()
                    hact = torch.from_numpy(hb['action'].astype(np.int64)).to(device)
                    hlab = torch.from_numpy(np.ascontiguousarray(hb['labels'])).to(device).float()
                    hrew = torch.from_numpy(hb['reward'].astype(np.float32)).to(device)
                    hlogits, hval, hbel = net.forward_all(hobs, hmask)
                    hm += (hlogits.argmax(1) == hact).sum().item()
                    k = len(hb)
                    herr2 += F.mse_loss(hval.squeeze(-1), hrew).item() * k
                    hbce += F.binary_cross_entropy_with_logits(hbel, hlab).item() * k
            hev = 1.0 - (herr2 / n_hold) / reward_var
            line += (f" | holdout match {hm / n_hold * 100:.1f}% "
                     f"EV {hev:.3f} BCE {hbce / n_hold:.4f}")
        print(line)

    # Save from CPU so the checkpoint stays device-neutral for the
    # orchestrator gate and export.py.
    torch.save(net.cpu().state_dict(), args.out)
    print(f"Saved candidate to {args.out}")

if __name__ == '__main__':
    main()
