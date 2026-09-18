"""League r11 two-tier candidate assembly (docs/exploiter_league_r11_prereg.md §2).

    python assemble_r11_candidate.py --tag E50 --thi 0.37
    python assemble_r11_candidate.py --validate        # freeze validations only

Module: champion 8a89da90 (default) + arm a a9653255 (ROUTER and tier-1
specialist) + r9 B1 specialist 50492c6d (tier-2), gate
'moonhead2:0.1:<T_hi>'. Every assembly asserts trace == eager on real
obs-v2 rows and that the trace rejects 556. --validate additionally runs
the registered endpoint checks:
  (b) T_hi = 2.0  reproduces the PROMOTED ensemble 8d7816d1;
  (c) T_hi = 0.1  reproduces the r10 B1 candidate d5222b1d;
  (d) the promoted checkpoint still loads and runs (existing gate kinds
      untouched).
Argmax equality on every row is REQUIRED; bitwise equality is reported.
"""
import argparse
import os
import sys

import numpy as np
import torch

from hearts_net import net_from_checkpoint, save_hybrid
from train import file_md5_8

CHAMP = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'   # 8a89da90
ARMA = 'v6_stage3/arma_lr1e-4.ep3.pth'                         # a9653255
B1_SPEC = 'equity_data/exploiter_r9/B1/specialist_B1_end.pth'  # 50492c6d
PROMOTED = 'hybrid_champ_arma_moonhead_0p1.pth'                # 8d7816d1
B1_CAND = 'equity_data/exploiter_r9/B1/r9_B1_end_ensemble.pth'  # d5222b1d
OUT_DIR = 'equity_data/exploiter_r11'
T_LO = 0.1


def holdout_rows(n):
    from v6_probe_eval import walk_holdout
    hold, _ = walk_holdout()
    obs = torch.from_numpy(np.concatenate(
        [np.ascontiguousarray(hold['obs']), np.ascontiguousarray(hold['ext'])],
        axis=1)).float()[:n] / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(hold['mask'])).bool()[:n]
    return obs, mask


def build(path, thi, spec):
    save_hybrid(path, CHAMP, spec, gate=f'moonhead2:{T_LO}:{thi}', router_ckpt=ARMA)
    return net_from_checkpoint(path).eval()


def run(net, obs, mask, bs=256):
    """Batched CPU fp32 forward (one 8k-row batch through three 19M-param
    transformers needs >15 GB)."""
    lo, va = [], []
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            l, v = net(obs[i:i + bs], mask[i:i + bs])
            lo.append(l); va.append(v)
    return torch.cat(lo), torch.cat(va)


def compare(name, net, ref, obs, mask):
    a, b = run(net, obs, mask), run(ref, obs, mask)
    am = int((a[0].argmax(1) != b[0].argmax(1)).sum())
    bit = bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
    finite = torch.isfinite(a[0]) & torch.isfinite(b[0])
    mx = float((a[0][finite] - b[0][finite]).abs().max())
    print(f'  {name}: argmax mismatches {am}/{len(obs)}, bitwise {bit}, '
          f'max |dlogit| {mx:.3g}')
    return am == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--thi', type=float)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--spec', default=B1_SPEC)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    for path, want in ((CHAMP, '8a89da90'), (ARMA, 'a9653255'),
                       (args.spec, '50492c6d'), (PROMOTED, '8d7816d1'),
                       (B1_CAND, 'd5222b1d')):
        got = file_md5_8(path)
        assert got == want, f'{path} md5 {got} != {want}'

    if args.validate:
        obs, mask = holdout_rows(4096)
        promoted = net_from_checkpoint(PROMOTED).eval()
        b1 = net_from_checkpoint(B1_CAND).eval()
        am_p = run(promoted, obs, mask)[0].argmax(1)
        am_b = run(b1, obs, mask)[0].argmax(1)
        d = am_p != am_b
        print(f'validation rows {len(obs)}; promoted vs B1 differ on '
              f'{int(d.sum())} rows (the checks below must see through this)')
        assert int(d.sum()) > 0, 'endpoints identical on these rows - check is vacuous'
        tmp = os.path.join(OUT_DIR, '_validate_tmp.pth')
        ok = compare('(b) T_hi=2.0 vs PROMOTED', build(tmp, 2.0, args.spec),
                     promoted, obs, mask)
        ok &= compare('(c) T_hi=0.1 vs B1 candidate', build(tmp, T_LO, args.spec),
                      b1, obs, mask)
        mid = build(tmp, 0.5, args.spec)
        m = run(mid, obs, mask)[0].argmax(1)
        dp = int((m != am_p).sum())
        db = int((m != am_b).sum())
        print(f'  interior T_hi=0.5: differs from promoted on {dp} rows, from B1 on {db}')
        ok &= dp > 0 and db > 0
        os.remove(tmp)
        print('(d) promoted checkpoint loads and runs: True')
        print('FREEZE VALIDATIONS ' + ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)

    assert args.tag and args.thi is not None, '--tag and --thi required'
    out_ckpt = os.path.join(OUT_DIR, f'r11_{args.tag}_ensemble.pth')
    out_trace = os.path.join(OUT_DIR, f'r11_{args.tag}_ensemble_882.pt')
    hy = build(out_ckpt, args.thi, args.spec)
    obs, mask = holdout_rows(512)
    tr = torch.jit.trace(hy, (torch.zeros(1, 882),
                              torch.ones(1, 52, dtype=torch.bool)))
    with torch.no_grad():
        a, b = tr(obs, mask), hy(obs, mask)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), 'trace != eager'
    try:
        tr(torch.zeros(1, 556), torch.ones(1, 52, dtype=torch.bool))
        print('FAIL: trace accepted 556')
        sys.exit(1)
    except Exception:
        pass
    tr.save(out_trace)
    print(f'candidate {args.tag}: gate moonhead2:{T_LO}:{args.thi}')
    print(f'  {out_ckpt}  ({file_md5_8(out_ckpt)})')
    print(f'  {out_trace} ({file_md5_8(out_trace)}) - trace==eager, rejects 556')
