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
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import headroom
from hearts_net import HeartsNet, HeartsNetV5, net_from_checkpoint

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

# Match-mode records (SelfPlayGen --match): obs is 556 wide (550 engine dims +
# the acting seat's 6 match-context dims, SearchPlayer::WriteCtx layout) and
# reward is the match-placement payoff (2.5 - place) * 4 in {+6,+2,-2,-6}
# (ties: intermediate averages). HeartsNetV5 consumes the 6-dim tail via its
# conditional match projection, so the same training loop applies unchanged.
MATCH_RECORD = np.dtype([
    ('obs', 'u1', 556), ('mask', 'u1', 52), ('labels', 'u1', 156),
    ('pi', 'u1', 52),
    ('action', '<u2'), ('seat', '<u2'), ('reward', '<f4'),
])
assert MATCH_RECORD.itemsize == 824

# Match-mode record format v2 (SelfPlayGen --match, 2026-07-31,
# docs/expert_iter_v2_prereg.md): the 824-byte v1 fields in the same order,
# then per-decision search statistics from SearchPlayer::LastStats. v2 files
# start with a 32-byte header: b"HMR2", version u16 (=2), record_size u16
# (=848), base_seed u32, thread_id u16, zero-fill to 32 bytes.
# second_action == 0xFFFF and n_dets == 0 mark decisions without per-action
# stats (pass picks, forced single-action moves). flags: bit0 = passing-phase
# decision, bit1 = tension score state at decision time.
MATCH_RECORD_V2 = np.dtype([
    ('obs', 'u1', 556), ('mask', 'u1', 52), ('labels', 'u1', 156),
    ('pi', 'u1', 52),
    ('action', '<u2'), ('seat', '<u2'), ('reward', '<f4'),
    ('eq_best', '<f4'), ('eq_second', '<f4'), ('gap_se', '<f4'),
    ('second_action', '<u2'), ('n_dets', '<u2'), ('match_id', '<u4'),
    ('flags', '<u2'), ('reserved', '<u2'),
])
assert MATCH_RECORD_V2.itemsize == 848

V2_MAGIC = b'HMR2'
V2_HEADER_BYTES = 32

def _read_records(path, dtype):
    """Read one SelfPlayGen file, auto-detecting the match v2 format.

    Files starting with the "HMR2" magic are v2: skip the 32-byte header and
    parse with MATCH_RECORD_V2. Headerless files are parsed with the caller's
    dtype after a size-divisibility check (824 -> MATCH_RECORD v1). When v1
    match records land in a v2 mix they are UPCAST to MATCH_RECORD_V2 with
    the new fields zeroed (second_action = 0xFFFF, the no-stats sentinel) so
    every caller sees one aligned dtype; returns (array, is_v1) so load_data
    can refuse v1 files where real search stats are required
    (--min-confidence / --anchor-coef)."""
    with open(path, 'rb') as fh:
        head = fh.read(V2_HEADER_BYTES)
    if head[:4] == V2_MAGIC:
        version, rec_size = struct.unpack_from('<HH', head, 4)
        if version != 2 or rec_size != MATCH_RECORD_V2.itemsize:
            raise SystemExit(f"{path}: unsupported HMR2 header "
                             f"(version {version}, record_size {rec_size})")
        if (os.path.getsize(path) - V2_HEADER_BYTES) % MATCH_RECORD_V2.itemsize:
            raise SystemExit(f"{path}: v2 payload is not a multiple of "
                             f"{MATCH_RECORD_V2.itemsize} bytes - truncated?")
        return np.fromfile(path, dtype=MATCH_RECORD_V2,
                           offset=V2_HEADER_BYTES), False
    if os.path.getsize(path) % dtype.itemsize != 0:
        raise SystemExit(
            f"{path}: size is not a multiple of {dtype.itemsize} - wrong or "
            f"stale record format? Regenerate the data.")
    arr = np.fromfile(path, dtype=dtype)
    if dtype is not MATCH_RECORD:
        return arr, False
    up = np.zeros(len(arr), dtype=MATCH_RECORD_V2)
    for name in MATCH_RECORD.names:
        up[name] = arr[name]
    up['second_action'] = 0xFFFF
    return up, True

def load_data(patterns, dtype=RECORD, kind='decision', holdout_frac=0.0):
    """Load records; optionally split off the contiguous TAIL of each file as
    holdout. Records within a file are deal-contiguous, so a tail split keeps
    (almost) whole deals out of training - a random record split leaks
    same-deal sibling records and inflates every holdout metric (measured:
    fake 0.99 EVs on the oracle head)."""
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit(f"No data files match {patterns}")
    train_chunks, hold_chunks = [], []
    n_v1 = 0
    for f in sorted(files):
        arr, is_v1 = _read_records(f, dtype)
        n_v1 += int(is_v1)
        cut = len(arr) - int(len(arr) * holdout_frac)
        train_chunks.append(arr[:cut])
        if cut < len(arr):
            hold_chunks.append(arr[cut:])
    data = np.concatenate(train_chunks)
    holdout = np.concatenate(hold_chunks) if hold_chunks else None
    n_hold = len(holdout) if holdout is not None else 0
    v1_note = f", {n_v1} v1 upcast" if (n_v1 and dtype is MATCH_RECORD) else ""
    print(f"Loaded {len(data):,} {kind} records from {len(files)} files{v1_note}"
          + (f" (+{n_hold:,} held out, per-file tails)" if n_hold else ""))
    return data, holdout, n_v1

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
    ap.add_argument('--arch', choices=['mlp', 'v5'], default='mlp',
                    help='fresh-network architecture: mlp = flat residual MLP, '
                         'v5 = card-token transformer (width = d_model, '
                         'blocks = encoder layers, heads = width // 32)')
    ap.add_argument('--width', type=int, default=512,
                    help='trunk width / d_model for fresh networks (warm starts '
                         'infer their size from the checkpoint)')
    ap.add_argument('--blocks', type=int, default=3,
                    help='residual blocks / encoder layers for fresh networks')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--hard-policy', action='store_true',
                    help='train the policy on one-hot CHOSEN actions instead '
                         'of the soft pi (required for equity-scored teachers '
                         'whose soft policies are near-uniform)')
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
    ap.add_argument('--match', action='store_true',
                    help='load SelfPlayGen --match records (v2 848-byte with '
                         'HMR2 header, or legacy v1 824-byte: 556-dim obs '
                         'with the 6 match-context dims appended, match-'
                         'placement rewards) and feed the 556-wide obs to the '
                         'net')
    ap.add_argument('--min-confidence', type=float, default=None, metavar='SIGMA',
                    help='train only on PLAY-phase records whose top-two '
                         'equity gap clears SIGMA x gap_se (v2 match records '
                         'required; errors if any v1 file is in the mix)')
    ap.add_argument('--anchor-coef', type=float, default=0.0, metavar='W',
                    help='adds W x KL(candidate || anchor policy) on a '
                         'uniformly-sampled batch of NON-confident play-phase '
                         'records each step (anti-drift anchor; 0 = off)')
    ap.add_argument('--anchor-model', default=None,
                    help='checkpoint for the frozen anchor policy '
                         '(net_from_checkpoint; required when --anchor-coef > 0)')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    headroom.apply_process_priority()
    headroom.banner()
    torch.set_float32_matmul_precision('high')  # TF32 matmuls on Ada

    # In --match mode records within a file are match-contiguous, so the
    # per-file tail holdout is effectively a by-match split.
    data, holdout, n_v1 = load_data(args.data,
                                    dtype=MATCH_RECORD if args.match else RECORD,
                                    kind='match decision' if args.match else 'decision',
                                    holdout_frac=args.holdout)

    # Confidence filtering / anchor loss (record format v2,
    # docs/expert_iter_v2_prereg.md). Both need real per-decision search
    # stats, which only v2 files carry (v1 upcasts have them zeroed).
    anchor_pool = None
    if args.min_confidence is not None or args.anchor_coef > 0:
        if not args.match:
            raise SystemExit('--min-confidence/--anchor-coef require --match '
                             '(search stats only exist in match v2 records)')
        if n_v1:
            raise SystemExit(f'--min-confidence/--anchor-coef require v2 '
                             f'records, but {n_v1} v1 file(s) are in --data')
        # Frozen confidence rule (prereg): gap > sigma x gap_se, play phase
        # only. sigma defaults to 2 for the anchor split when only
        # --anchor-coef is given.
        sigma = args.min_confidence if args.min_confidence is not None else 2.0
        def conf_split(arr):
            play = (arr['flags'] & 1) == 0
            gap = arr['eq_best'].astype(np.float32) - arr['eq_second']
            return play, play & (gap > sigma * arr['gap_se'])
        play_m, conf_m = conf_split(data)
        if args.anchor_coef > 0:
            anchor_pool = data[play_m & ~conf_m]
            print(f"Anchor pool: {len(anchor_pool):,} non-confident play-phase "
                  f"records (sigma={sigma:g})")
        if args.min_confidence is not None:
            n0 = len(data)
            data = data[conf_m]
            print(f"Confidence filter (sigma={sigma:g}): kept {len(data):,} of "
                  f"{n0:,} training records "
                  f"({play_m.sum():,} play-phase)")
            if len(data) == 0:
                raise SystemExit('confidence filter kept 0 records')
            if holdout is not None:
                _, hconf = conf_split(holdout)
                holdout = holdout[hconf]
                print(f"  holdout filtered to {len(holdout):,} confident records")
                if len(holdout) == 0:
                    holdout = None

    n_hold = len(holdout) if holdout is not None else 0
    leaf = None
    if args.leaf_data:
        leaf, _, _ = load_data(args.leaf_data, dtype=LEAF_RECORD, kind='leaf-value')
        leaf_var = float(np.var(leaf['reward'])) + 1e-8

    if args.init and os.path.exists(args.init):
        net = net_from_checkpoint(args.init)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"Warm start from {args.init} "
              f"({type(net).__name__}, {n_params / 1e6:.2f}M params)")
    else:
        if args.arch == 'v5':
            net = HeartsNetV5(d_model=args.width, num_layers=args.blocks,
                              num_heads=max(1, args.width // 32))
        else:
            net = HeartsNet(width=args.width, num_blocks=args.blocks)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"Fresh network ({args.arch}): width {args.width}, {args.blocks} "
              f"blocks/layers, {n_params / 1e6:.2f}M params")
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Frozen anchor policy for the KL term (skipped cleanly when coef == 0).
    anchor_net = None
    if args.anchor_coef > 0:
        if not args.anchor_model:
            raise SystemExit('--anchor-coef > 0 requires --anchor-model')
        if anchor_pool is None or len(anchor_pool) == 0:
            print('WARNING: anchor pool empty (no non-confident play-phase '
                  'records); anchor loss disabled')
        else:
            anchor_net = net_from_checkpoint(args.anchor_model)
            anchor_net.to(device).eval()
            for ap_ in anchor_net.parameters():
                ap_.requires_grad_(False)
            print(f"Anchor: {args.anchor_model} ({type(anchor_net).__name__}), "
                  f"coef {args.anchor_coef}")

    n = len(data)
    reward_var = float(np.var(data['reward'])) + 1e-8

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        ce_sum = match_sum = err2_sum = bce_sum = oerr2_sum = akl_sum = 0.0
        seen = 0
        if leaf is not None:
            lperm = np.random.permutation(len(leaf))
            lpos = 0
            lerr2_sum = loerr2_sum = 0.0
            lseen = 0

        for start in range(0, n, args.batch):
            headroom.pace()
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
            if args.hard_policy:
                # Equity-scored teachers (match-aware search) emit near-UNIFORM
                # soft policies - P(win) gaps between actions are a few percent,
                # so pi carries almost no preference and pow-sharpening a
                # uniform distribution is a no-op (measured 2026-07-31: entropy
                # 1.04 -> 1.08 from sharpen 2 -> 8). The teacher's signal lives
                # in its CHOSEN action: train one-hot on it.
                pi = F.one_hot(actions, num_classes=52).float()

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

            # Anchor: KL(candidate || anchor) on a uniform batch of
            # NON-confident play-phase records - keeps the flat-state policy
            # from drifting where the teacher's argmax is sampling noise.
            if anchor_net is not None:
                aidx = np.random.randint(0, len(anchor_pool), size=args.batch)
                ab = anchor_pool[aidx]
                aobs = torch.from_numpy(np.ascontiguousarray(ab['obs'])).to(device).float() / 255.0
                amask = torch.from_numpy(np.ascontiguousarray(ab['mask'])).to(device).bool()
                clogits, _, _ = net.forward_all(aobs, amask)
                with torch.no_grad():
                    alogits, _, _ = anchor_net.forward_all(aobs, amask)
                # Illegal logits are -inf on both sides: their candidate prob
                # is 0 and both logps are masked to 0, so they contribute 0.
                clogp = F.log_softmax(clogits, dim=1).masked_fill(~amask, 0.0)
                alogp = F.log_softmax(alogits, dim=1).masked_fill(~amask, 0.0)
                cp = F.softmax(clogits, dim=1)
                anchor_kl = (cp * (clogp - alogp)).sum(dim=1).mean()
                loss = loss + args.anchor_coef * anchor_kl
                akl_sum += anchor_kl.item() * len(idx)

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
        if anchor_net is not None:
            line += f" | anchor KL {akl_sum / seen:.4f}"
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
