"""Post-halt diagnostic (step 1): how much behavioral change does a
defense-gate pass actually cost?

Measures, for the round-1 PPO trials (defense-gate deltas -0.031 /
-0.047 / -0.250, t3 = the only pass ever) and the round-2 screened
candidates (defense-gate dead nulls), on the SAME b2 holdout banks:
  - ordinary-state agreement with the baseline (the drift screen's
    quantity, bank: drift_holdout vs stored baseline argmax)
  - moon-alive play + pass agreement with the baseline (how much the
    defense-relevant behavior moved; baseline argmax computed here)
  - teacher-match vs the search defender's recorded choices (does a
    defense pass even correlate with imitating the teacher?)
Diagnostic only: no gates, no training, archived artifacts.
"""
import numpy as np
import torch

from hearts_net import net_from_checkpoint

BASELINE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
CANDS = [
    ('r1_t1  (def -0.031 ns)', 'cand_exploiter_r1_t1.pth'),
    ('r1_t2  (def -0.047 ns)', 'cand_exploiter_r1_t2.pth'),
    ('r1_t3  (def -0.250 PASS)', 'cand_exploiter_r1_t3.pth'),
    ('b2_kl4ep1 (def +0.094)', 'cand_b2f_kl4.pth.ep1.pth'),
    ('b2_kl8ep1 (def +0.016)', 'cand_b2f_kl8.pth.ep1.pth'),
]

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_npz(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


drf = load_npz('b2_data/drift_holdout.npz')
dho = load_npz('b2_data/def_holdout.npz')
is_pass = (dho['flags'] & 1) != 0


@torch.no_grad()
def argmaxes(net, arr, batch=4096):
    out = np.empty(len(arr['obs']), dtype=np.int64)
    for s in range(0, len(arr['obs']), batch):
        obs = torch.from_numpy(np.ascontiguousarray(arr['obs'][s:s+batch])).to(dev)
        mask = torch.from_numpy(np.ascontiguousarray(arr['mask'][s:s+batch])).to(dev).bool()
        logits, _, _ = net.forward_all(obs, mask)
        out[s:s+batch] = logits.argmax(1).cpu().numpy()
    return out


base = net_from_checkpoint(BASELINE).to(dev).eval()
base_drf = drf['base_argmax'].astype(np.int64)
base_dho = argmaxes(base, dho)
teacher = dho['action'].astype(np.int64)
bp = base_dho[~is_pass]
bq = base_dho[is_pass]

print(f"{'candidate':26s} {'ordinary':>9s} {'alive-play':>10s} "
      f"{'pass':>7s} {'tm-play':>8s} {'tm-pass':>8s}")
print(f"{'baseline (self)':26s} {'1.000':>9s} {'1.000':>10s} {'1.000':>7s} "
      f"{(base_dho[~is_pass] == teacher[~is_pass]).mean():8.3f} "
      f"{(base_dho[is_pass] == teacher[is_pass]).mean():8.3f}")
for label, path in CANDS:
    net = net_from_checkpoint(path).to(dev).eval()
    a_drf = argmaxes(net, drf)
    a_dho = argmaxes(net, dho)
    row = {
        'ord': (a_drf == base_drf).mean(),
        'alive': (a_dho[~is_pass] == bp).mean(),
        'pass': (a_dho[is_pass] == bq).mean(),
        'tmp': (a_dho[~is_pass] == teacher[~is_pass]).mean(),
        'tmq': (a_dho[is_pass] == teacher[is_pass]).mean(),
    }
    print(f"{label:26s} {row['ord']:9.3f} {row['alive']:10.3f} "
          f"{row['pass']:7.3f} {row['tmp']:8.3f} {row['tmq']:8.3f}")
