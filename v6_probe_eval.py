"""v6 data-scaling probe: FIXED-HOLDOUT per-match evaluation
(docs/v6_data_scaling_prereg.md §3-§4).

Evaluates arm-a snapshots on the FULL-bank by-match holdout — the same
123-match / 80,220-record split Stage 3 froze on — regardless of which
subset the net trained on. Every holdout record is keyed (file,
match_id) so the analysis can cluster by match and pair across nets.

Scoring uses the frozen v6_distill functions (batch_tensors,
forward_losses, build_net) UNCHANGED; the per-record CE is the same
-log p[chosen action] that v6_distill.eval_holdout averages, recovered
here per record from the returned logits.

Self-tests (--selftest):
  1. the walk's holdout equals v6_distill.load_bank(...)[1] byte for
     byte (same records, same order);
  2. scoring the frozen Stage-3 arm-a net (v6_stage3/arma_lr1e-4.ep3.pth)
     reproduces its recorded holdout CE 0.8413 / match 0.642 to within
     GPU nondeterminism (|dCE| < 0.002, |dmatch| < 0.002).

Usage:
  python v6_probe_eval.py --selftest
  python v6_probe_eval.py --nets v6_probe/S1_s20260812.ep3.pth ... \
         --out v6_probe/holdout_by_match.csv
"""
import argparse
import csv
import glob
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F

import v6_distill as vd
from cloud.shard_check import V3_REC

BANK_GLOB = 'expert_data/v6_bank/gen_*_t*.bin'
HOLDOUT_MOD = 20
FROZEN_REF = ('v6_stage3/arma_lr1e-4.ep3.pth', 0.8413, 0.642)


def walk_holdout(pattern=BANK_GLOB, holdout_mod=HOLDOUT_MOD):
    """Same rule as v6_distill.load_bank; returns (records, keys) where
    keys[i] = (basename, match_id) for records[i]."""
    files = sorted(glob.glob(pattern))
    parts, keys = [], []
    for path in files:
        raw = open(path, 'rb').read()
        assert raw[:4] == b'HMR3', path
        r = np.frombuffer(raw[32:], dtype=V3_REC)
        base = os.path.basename(path)
        hold_ids = set()
        for mid in np.unique(r['match_id']):
            h = int(hashlib.md5(f'{base}:{mid}'.encode()).hexdigest(), 16)
            if h % holdout_mod == 0:
                hold_ids.add(int(mid))
        if hold_ids:
            hmask = np.isin(r['match_id'], sorted(hold_ids))
            hr = r[hmask].copy()
            parts.append(hr)
            keys.extend((base, int(m)) for m in hr['match_id'])
    hold = np.concatenate(parts)
    assert len(hold) == len(keys)
    return hold, keys


@torch.no_grad()
def score_net(path, hold, keys, device, batch=512, arm='a'):
    net = vd.build_net(arm).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    per = {}   # (file, mid) -> [n, ce_sum, correct, n_play, ent_sum]
    tot_ce = tot_ok = 0.0
    for start in range(0, len(hold), batch):
        b = hold[start:start + batch]
        t = vd.batch_tensors(b, arm, device)
        _, logits, _ = vd.forward_losses(net, arm, t)
        logp = F.log_softmax(logits, dim=1).masked_fill(~t['mask'], 0.0)
        ce = (-logp.gather(1, t['actions'].unsqueeze(1)).squeeze(1)).cpu().numpy()
        ok = (logits.argmax(dim=1) == t['actions']).cpu().numpy()
        p = F.softmax(logits, dim=1)
        ent = (-(p * logp).sum(dim=1)).cpu().numpy()
        play = ((b['flags'] & 1) == 0)
        tot_ce += ce.sum(); tot_ok += ok.sum()
        for j in range(len(b)):
            k = keys[start + j]
            a = per.setdefault(k, [0, 0.0, 0, 0, 0.0])
            a[0] += 1; a[1] += float(ce[j]); a[2] += int(ok[j])
            if play[j]:
                a[3] += 1; a[4] += float(ent[j])
    n = len(hold)
    return per, {'ce': tot_ce / n, 'teacher_match': tot_ok / n,
                 'records': n, 'matches': len(per)}


def selftest(device):
    hold, keys = walk_holdout()
    _, hold_ref = vd.load_bank(BANK_GLOB, HOLDOUT_MOD)
    assert len(hold) == len(hold_ref) == 80220, (len(hold), len(hold_ref))
    assert hold.tobytes() == hold_ref.tobytes(), 'holdout walk != load_bank'
    assert len(set(keys)) == 123, len(set(keys))
    print('selftest 1 PASS: holdout identical to v6_distill.load_bank '
          f'({len(hold)} records, {len(set(keys))} matches)')
    path, ce_ref, tm_ref = FROZEN_REF
    per, agg = score_net(path, hold, keys, device)
    d_ce, d_tm = abs(agg['ce'] - ce_ref), abs(agg['teacher_match'] - tm_ref)
    print(f"selftest 2: frozen arm-a ep3 -> CE {agg['ce']:.4f} (ref {ce_ref}) "
          f"match {agg['teacher_match']:.4f} (ref {tm_ref}) | "
          f"dCE {d_ce:.4f} dmatch {d_tm:.4f}")
    assert d_ce < 0.002 and d_tm < 0.002, 'frozen-net reproduction FAILED'
    # per-match aggregation must reproduce the aggregate exactly
    n = sum(v[0] for v in per.values())
    assert n == len(hold)
    ce2 = sum(v[1] for v in per.values()) / n
    assert abs(ce2 - agg['ce']) < 1e-9
    print('selftest 2 PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--nets', nargs='*', default=[])
    ap.add_argument('--out')
    args = ap.parse_args()
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.selftest:
        selftest(device); return
    if not args.nets or not args.out:
        raise SystemExit('need --nets and --out')
    hold, keys = walk_holdout()
    write_header = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['net', 'file', 'match_id', 'n', 'ce_sum',
                        'correct', 'n_play', 'ent_sum'])
        for path in args.nets:
            tag = os.path.basename(path).replace('.ep3.pth', '')
            per, agg = score_net(path, hold, keys, device)
            for (fn, mid), v in sorted(per.items()):
                w.writerow([tag, fn, mid, v[0], f'{v[1]:.6f}', v[2],
                            v[3], f'{v[4]:.6f}'])
            f.flush()
            print(f"{tag}: CE {agg['ce']:.4f} match {agg['teacher_match']:.4f} "
                  f"({agg['records']} rec / {agg['matches']} matches)")


if __name__ == '__main__':
    main()
