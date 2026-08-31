"""Null contract (b) + gate A/A for the league r8 ensemble-learner trainer
(docs/exploiter_league_r8_prereg.md §3).

With the specialist at its INIT (arm a, untrained), the trainer's ensemble
decision function EnsembleCtx.decide - the exact code path
run_cycle_vec_match uses - must reproduce the PROMOTED ensemble:

  1. GATE A/A + gate-fires: ens.gate() bits equal the promoted eager
     hybrid's gate_mask() (CPU fp32 reference) on every decision state of a
     self-play rollout, and the gate fires (> 0 gated rows; the r6 rule).
  2. ACTION A/A: decide() in deterministic mode equals the argmax of BOTH
     the promoted eager hybrid and the served v2 trace on every state
     (n >= 6,000 decisions, >= 400 of them gated).

The trainer side runs on the TRAINING device (cuda if available, fp32 -
the fp32=True branch); references run CPU fp32 (the serving substrate), so
this also certifies the device gap is behaviourally nil. Exit 1 on any
mismatch. Read-only: drives a throwaway MatchVecEnv, writes nothing.
"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault('ENSEMBLE_DETERMINISTIC', '1')

from train import EnsembleCtx, file_md5_8                     # noqa: E402
from hearts_net import net_from_checkpoint                    # noqa: E402
from hearts_match_env import MatchVecEnv                      # noqa: E402

CHAMP = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'  # 8a89da90
ARMA = 'v6_stage3/arma_lr1e-4.ep3.pth'                        # a9653255
PROMOTED = 'hybrid_champ_arma_moonhead_0p1.pth'               # 8d7816d1
TRACE = 'hybrid_champ_arma_moonhead_0p1_882_v2.pt'            # 4f6d396a
N_MIN, N_GATED_MIN = 6000, 400

if __name__ == '__main__':
    for path, want in ((CHAMP, '8a89da90'), (ARMA, 'a9653255'),
                       (PROMOTED, '8d7816d1'), (TRACE, '4f6d396a')):
        got = file_md5_8(path)
        assert got == want, f'{path} md5 {got} != {want}'
    print('artifact md5s verified')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'trainer side on {device}; references on cpu fp32')

    def frozen(path):
        n = net_from_checkpoint(path).to(device).eval()
        for p in n.parameters():
            p.requires_grad_(False)
        return n

    champion = frozen(CHAMP)
    router = frozen(ARMA)
    specialist = frozen(ARMA)          # init composition == promoted ensemble
    ens = EnsembleCtx(champion, router, 'moonhead:0.1', device)
    assert ens.deterministic, 'A/A requires ENSEMBLE_DETERMINISTIC=1'

    ref = net_from_checkpoint(PROMOTED).eval()                # cpu fp32 eager
    trace = torch.jit.load(TRACE).eval()

    vec = MatchVecEnv(64, 20260821)
    ids = np.arange(64, dtype=np.int64)
    total = gated = gate_mm = act_mm_eager = act_mm_trace = 0
    while total < N_MIN or gated < N_GATED_MIN:
        obs = vec.observe_v2_batch(ids)
        mask = vec.legal_mask_batch(ids)
        actions, _, _, g = ens.decide(obs, mask, specialist, deterministic=True)

        ot = torch.from_numpy(obs)
        mt = torch.from_numpy(mask)
        with torch.no_grad():
            lref, _ = ref(ot, mt)
            ltr, _ = trace(ot, mt)
            lc, _ = ref.champion(ot[:, :556], mt)
            gref = ref.gate_mask(ot, lc).numpy()
        act_mm_eager += int((actions != lref.argmax(1).numpy()).sum())
        act_mm_trace += int((actions != ltr.argmax(1).numpy()).sum())
        gate_mm += int((g != gref).sum())
        total += len(ids)
        gated += int(g.sum())
        vec.step_batch(ids, actions.astype(np.int64))

    print(f'decisions {total} (gated {gated}, {100.0 * gated / total:.2f}%)')
    print(f'gate mismatches vs gate_mask reference: {gate_mm}')
    print(f'action mismatches vs promoted eager:    {act_mm_eager}')
    print(f'action mismatches vs served v2 trace:   {act_mm_trace}')
    assert gated > 0, 'GATE NEVER FIRED (r6 rule) - instrument invalid'
    if gate_mm or act_mm_eager or act_mm_trace:
        print('ENSEMBLE-LEARNER A/A FAIL')
        sys.exit(1)
    print('ENSEMBLE-LEARNER A/A PASS: trainer decision function == promoted '
          'ensemble (gate and actions), gate fires')
