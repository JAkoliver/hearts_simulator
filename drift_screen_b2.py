"""Round-2 Phase B2 offline drift screen (halt-default per candidate;
docs/exploiter_league_r2_prereg.md).

REGISTERED CRITERION: candidate argmax agreement with the baseline on
the 20k held-out ordinary positions (b2_data/drift_holdout.npz, baseline
argmax precomputed by build_b2_dataset.py) must be >= 97%. Candidates
below never reach gates.

TELEMETRY (informs, never gates): defense-holdout teacher-match (vs the
search defender's choices), mean policy entropy on the drift holdout,
and the baseline's own numbers for reference.

Null contract: an unmodified baseline copy must read agreement 1.0
exactly (train_b2.py --epochs 0 produces one).

Usage: python drift_screen_b2.py <candidate.pth> [--json out.json]
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from hearts_net import net_from_checkpoint

THRESHOLD = 0.97


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


@torch.no_grad()
def screen(net, arr, target_key, device, batch=4096, want_entropy=False):
    hits = 0
    ent_sum = 0.0
    n = len(arr['obs'])
    for s in range(0, n, batch):
        b = slice(s, min(s + batch, n))
        obs = torch.from_numpy(np.ascontiguousarray(arr['obs'][b])).to(device)
        mask = torch.from_numpy(
            np.ascontiguousarray(arr['mask'][b])).to(device).bool()
        logits, _, _ = net.forward_all(obs, mask)
        tgt = torch.from_numpy(
            arr[target_key][b].astype(np.int64)).to(device)
        hits += int((logits.argmax(1) == tgt).sum().item())
        if want_entropy:
            p = F.softmax(logits, dim=1)
            logp = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
            ent_sum += float(-(p * logp).sum(dim=1).sum().item())
    return hits / n, (ent_sum / n if want_entropy else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('--data-dir', default='b2_data')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    drf = load_npz(f'{args.data_dir}/drift_holdout.npz')
    dho = load_npz(f'{args.data_dir}/def_holdout.npz')
    net = net_from_checkpoint(args.candidate).to(device).eval()

    agree, ent = screen(net, drf, 'base_argmax', device, want_entropy=True)
    def_match, _ = screen(net, dho, 'action', device)
    ok = agree >= THRESHOLD

    out = {'candidate': args.candidate,
           'drift_agreement': round(agree, 5),
           'threshold': THRESHOLD,
           'pass': bool(ok),
           'n_drift': int(len(drf['obs'])),
           'telemetry': {'defense_holdout_teacher_match': round(def_match, 5),
                         'mean_entropy_drift_holdout': round(ent, 4)}}
    print(f'{args.candidate}: drift agreement {agree:.5f} '
          f'({"PASS" if ok else "FAIL"} vs {THRESHOLD})  '
          f'defense-holdout match {def_match:.4f}  entropy {ent:.3f}')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
