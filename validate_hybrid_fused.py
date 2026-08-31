"""Null contract for the HeartsHybrid single-aux-forward refactor
(docs/gated_ensemble_program.md §7 step 2; 2026-08-21).

The fused path must be BIT-IDENTICAL to the promoted artifacts:
  1. vs the RELEASED trace hybrid_champ_arma_moonhead_0p1_882.pt
     (md5 9d9a4f49 - built from the pre-refactor two-forward code):
     logits and value equal on real obs-v2 states, CPU fp32.
  2. Gate decisions equal to gate_mask() (the un-fused reference, still
     used by the probes/drift instruments) on the same states.
  3. A fresh trace of the refactored module: trace == eager, rejects 556.
  4. Speed: per-row cost of fused vs two-forward call pattern (report).
Exit 1 on any mismatch.
"""
import sys
import time

import numpy as np
import torch

from hearts_net import net_from_checkpoint
from v6_probe_eval import walk_holdout

CKPT = 'hybrid_champ_arma_moonhead_0p1.pth'
OLD_TRACE = 'hybrid_champ_arma_moonhead_0p1_882.pt'
N = 20000

if __name__ == '__main__':
    hy = net_from_checkpoint(CKPT).eval()
    old = torch.jit.load(OLD_TRACE).eval()
    hold, _ = walk_holdout()
    obs = torch.from_numpy(np.concatenate(
        [np.ascontiguousarray(hold['obs']), np.ascontiguousarray(hold['ext'])],
        axis=1)).float()[:N] / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(hold['mask'])).bool()[:N]

    ok_l = ok_v = True
    gates_ref = []
    gates_new = []
    with torch.no_grad():
        for s in range(0, N, 1024):
            o, m = obs[s:s + 1024], mask[s:s + 1024]
            l_new, v_new = hy(o, m)
            l_old, v_old = old(o, m)
            ok_l &= torch.equal(l_new, l_old)
            ok_v &= torch.equal(v_new, v_old)
            lc, _ = hy.champion(o[:, :556], m)
            gates_ref.append(hy.gate_mask(o, lc))
            ls, vs, _, ml, _ = hy.specialist.forward_aux(o, m)
            alive = o[:, 872:876] > 0.5
            g = (torch.sigmoid(ml[:, 1:]).max(dim=1).values > 0.1) & alive[:, 1:].any(dim=1)
            gates_new.append(g)
    assert ok_l and ok_v, 'FUSED OUTPUT != RELEASED TRACE'
    print(f'check 1 PASS: fused forward bit-identical to the released trace on {N} states')
    gr, gn = torch.cat(gates_ref), torch.cat(gates_new)
    assert torch.equal(gr, gn), 'fused gate != gate_mask reference'
    print(f'check 2 PASS: fused gate == gate_mask on {N} states (fire rate {gr.float().mean():.4f})')

    tr = torch.jit.trace(hy, (torch.zeros(1, 882), torch.ones(1, 52, dtype=torch.bool)))
    with torch.no_grad():
        a = tr(obs[:256], mask[:256]); b = hy(obs[:256], mask[:256])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), 'new trace != eager'
    try:
        tr(torch.zeros(1, 556), torch.ones(1, 52, dtype=torch.bool))
        print('FAIL: new trace accepted 556'); sys.exit(1)
    except Exception:
        pass
    tr.save('hybrid_champ_arma_moonhead_0p1_882_v2.pt')
    print('check 3 PASS: refactored trace == eager, rejects 556 -> '
          'hybrid_champ_arma_moonhead_0p1_882_v2.pt')

    # 4. speed, batch=1 (the serving shape) and batch=256
    for bs in (1, 256):
        o, m = obs[:bs], mask[:bs]
        with torch.no_grad():
            for _ in range(3): hy(o, m); old(o, m)          # warm
            t0 = time.time()
            for _ in range(50): hy(o, m)
            t_new = (time.time() - t0) / 50
            t0 = time.time()
            for _ in range(50): old(o, m)
            t_old = (time.time() - t0) / 50
        print(f'check 4 (batch {bs}): old {1000*t_old:.2f} ms -> fused {1000*t_new:.2f} ms '
              f'({t_old/t_new:.2f}x)')
    print('ALL FUSED-HYBRID NULL-CONTRACT CHECKS PASS')
