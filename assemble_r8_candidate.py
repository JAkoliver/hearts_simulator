"""League r8 candidate assembly (docs/exploiter_league_r8_prereg.md §2/§4).

Bundles the trial's TRAINED specialist with the frozen champion and the
frozen original router (arm a's moon head) into one hybrid checkpoint +
one 882 traced module for the C++ instruments:

    python assemble_r8_candidate.py --tag L1r1 [--src hearts_model_mid.pth]

Verifies component md5s first (champion 8a89da90, router a9653255),
asserts trace == eager on real obs-v2 rows and that the trace rejects 556.
Note the r8 candidate carries router_sd, so its forward is the THREE-net
path (router + specialist + champion) - ~1.5x the promoted trace's cost;
quoted durations must come from the pace probe, not r7 rows.
"""
import argparse
import sys

import numpy as np
import torch

from hearts_net import net_from_checkpoint, save_hybrid
from train import file_md5_8

CHAMP = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'  # 8a89da90
ARMA = 'v6_stage3/arma_lr1e-4.ep3.pth'                        # a9653255
GATE = 'moonhead:0.1'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='e.g. L1r1, L1r1_mid')
    ap.add_argument('--src', default='hearts_model_final.pth',
                    help='trained specialist checkpoint')
    args = ap.parse_args()

    for path, want in ((CHAMP, '8a89da90'), (ARMA, 'a9653255')):
        got = file_md5_8(path)
        assert got == want, f'{path} md5 {got} != {want}'
    src_md5 = file_md5_8(args.src)
    if src_md5 == 'a9653255':
        print('WARNING: specialist == the init (untrained); assembling anyway')

    out_ckpt = f'r8_{args.tag}_ensemble.pth'
    out_trace = f'r8_{args.tag}_ensemble_882.pt'
    save_hybrid(out_ckpt, CHAMP, args.src, gate=GATE, router_ckpt=ARMA)
    hy = net_from_checkpoint(out_ckpt).eval()

    from v6_probe_eval import walk_holdout
    hold, _ = walk_holdout()
    obs = torch.from_numpy(np.concatenate(
        [np.ascontiguousarray(hold['obs']), np.ascontiguousarray(hold['ext'])],
        axis=1)).float()[:512] / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(hold['mask'])).bool()[:512]

    tr = torch.jit.trace(hy, (torch.zeros(1, 882),
                              torch.ones(1, 52, dtype=torch.bool)))
    with torch.no_grad():
        a = tr(obs, mask)
        b = hy(obs, mask)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), 'trace != eager'
    try:
        tr(torch.zeros(1, 556), torch.ones(1, 52, dtype=torch.bool))
        print('FAIL: trace accepted 556')
        sys.exit(1)
    except Exception:
        pass
    tr.save(out_trace)
    print(f'candidate {args.tag}: specialist {args.src} ({src_md5})')
    print(f'  {out_ckpt}  ({file_md5_8(out_ckpt)})')
    print(f'  {out_trace} ({file_md5_8(out_trace)}) - trace==eager, rejects 556')
