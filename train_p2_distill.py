"""Phase 2 Stage D — visit-count distillation + recipe freeze
(docs/phase2_visitcount_prereg.md, signed).

- Warm start from the md5-verified 8a89da90 milestone (NEVER the
  working file, which holds a rejected candidate).
- Loss on non-forced play decisions: soft-target cross-entropy against
  the tree's visit distribution (= KL(teacher||student) + const) -
  never argmax CE. POLICY HEAD ONLY: the bank carries no belief labels
  and the value head's GAE-return scale is not reconstructible from
  final placements - both heads stay frozen (registered deviation from
  the prereg's aspirational value/belief line; the entropy diagnostic
  and Stage E gates judge net behavior).
- Recipe grid: lr {1e-5, 3e-5} x epoch checkpoints {1,2,3} = 2 runs,
  6 candidates. Freeze on HOLDOUT ONLY (the last 17 matches of each
  shard, ~10%, never trained on): holdout teacher-KL ranks; entropy
  must stay within 2x of the baseline in both directions; the freeze
  picks <= 2 for the gates.

Writes cand_p2_lr{L}_ep{E}.pth and
equity_data/verdicts/p2_stageD_freeze.json.
"""
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from hearts_net import net_from_checkpoint
from validate_p2_records import DT

MILESTONE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
MILESTONE_MD5 = '8a89da90'
BANKS = ['expert_data/p2/bank_s0.hvt', 'expert_data/p2/bank_s1.hvt']
HOLDOUT_FROM_MATCH = 153     # per shard: matches 153..169 = holdout
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 1024


def load_bank():
    tr, ho = [], []
    for p in BANKS:
        r = np.fromfile(p, dtype=DT)
        r = r[(r['kind'] == 0) & (r['mask'].sum(1) > 1) & (r['visits'] > 0)]
        ho.append(r[r['match'] >= HOLDOUT_FROM_MATCH])
        tr.append(r[r['match'] < HOLDOUT_FROM_MATCH])
    return np.concatenate(tr), np.concatenate(ho)


def tensors(r):
    obs = torch.from_numpy(np.ascontiguousarray(r['obs']))
    mask = torch.from_numpy(np.ascontiguousarray(r['mask'])).bool()
    pi = torch.from_numpy(np.ascontiguousarray(r['pi']))
    pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-8)
    return obs, mask, pi


@torch.no_grad()
def holdout_stats(net, obs, mask, pi):
    """(teacher-KL, mean policy entropy) on the holdout."""
    net.eval()
    kls, ents = [], []
    for i in range(0, len(obs), 4096):
        o = obs[i:i + 4096].to(DEVICE)
        m = mask[i:i + 4096].to(DEVICE)
        p = pi[i:i + 4096].to(DEVICE)
        logits = net.forward_all(o, m)[0]
        # illegal actions carry -inf logits and exactly-zero teacher
        # mass: 0 * -inf = NaN unless the illegal terms are zeroed
        logq = F.log_softmax(logits, dim=1).masked_fill(~m, 0.0)
        q = F.softmax(logits, dim=1)
        logp = (p.clamp_min(1e-12)).log().masked_fill(~m, 0.0)
        kls.append(((p * (logp - logq)).sum(1)).cpu())
        ents.append((-(q * logq.clamp(-30, 0)).sum(1)).cpu())
    return float(torch.cat(kls).mean()), float(torch.cat(ents).mean())


def main():
    md5 = hashlib.md5(open(MILESTONE, 'rb').read()).hexdigest()[:8]
    assert md5 == MILESTONE_MD5, f'milestone md5 {md5} != {MILESTONE_MD5}'
    tr, ho = load_bank()
    print(f'train {len(tr)} decisions | holdout {len(ho)} '
          f'(matches >= {HOLDOUT_FROM_MATCH})', flush=True)
    obs_t, mask_t, pi_t = tensors(tr)
    obs_h, mask_h, pi_h = tensors(ho)

    base = net_from_checkpoint(MILESTONE).to(DEVICE)
    base_kl, base_ent = holdout_stats(base, obs_h, mask_h, pi_h)
    print(f'baseline: teacher-KL {base_kl:.4f} | entropy {base_ent:.4f}',
          flush=True)

    out = {'baseline': {'teacher_kl': base_kl, 'entropy': base_ent},
           'milestone_md5': md5, 'candidates': {}}
    for lr in (1e-5, 3e-5):
        net = net_from_checkpoint(MILESTONE).to(DEVICE)
        for p in net.value_head.parameters():
            p.requires_grad = False
        for p in net.belief_head.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad], lr=lr)
        for ep in range(1, 4):
            net.train()
            perm = torch.randperm(len(obs_t))
            tot = n = 0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                o = obs_t[idx].to(DEVICE)
                m = mask_t[idx].to(DEVICE)
                p = pi_t[idx].to(DEVICE)
                logits = net.forward_all(o, m)[0]
                # zero the illegal (-inf) slots - teacher mass is 0
                # there, but 0 * -inf poisons the loss with NaN
                logq = F.log_softmax(logits, dim=1).masked_fill(~m, 0.0)
                loss = -(p * logq).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
                n += len(idx)
            kl, ent = holdout_stats(net, obs_h, mask_h, pi_h)
            tag = f'lr{lr:.0e}_ep{ep}'
            path = f'cand_p2_{tag}.pth'
            torch.save(net.state_dict(), path)
            ent_ok = (base_ent / 2) <= ent <= (base_ent * 2)
            out['candidates'][tag] = {
                'path': path, 'train_ce': round(tot / n, 4),
                'holdout_teacher_kl': round(kl, 4),
                'holdout_entropy': round(ent, 4), 'entropy_ok': ent_ok}
            print(f'{tag}: train-CE {tot / n:.4f} | holdout KL {kl:.4f} '
                  f'| entropy {ent:.4f} {"OK" if ent_ok else "OUT-OF-BAND"}',
                  flush=True)

    # freeze: entropy-banded, best holdout teacher-KL, <= 2 picks
    ok = [(v['holdout_teacher_kl'], k) for k, v in out['candidates'].items()
          if v['entropy_ok']]
    ok.sort()
    out['picks'] = [k for _, k in ok[:2]]
    out['note'] = ('policy-head-only distillation: bank has no belief '
                   'labels; value head GAE scale not reconstructible - '
                   'both heads frozen (registered deviation)')
    json.dump(out, open('equity_data/verdicts/p2_stageD_freeze.json', 'w'),
              indent=1)
    print('FREEZE PICKS:', out['picks'], flush=True)


if __name__ == '__main__':
    main()
