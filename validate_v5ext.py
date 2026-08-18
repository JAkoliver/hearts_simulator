"""League round 5 null contract (docs/exploiter_league_r5_prereg.md §3.3):
HeartsNetV5Ext = the v5 champion + ZERO-INIT obs-v2 adapters must be
BIT-IDENTICAL to HeartsNetV5 on the 556 prefix at init, its adapters must
receive gradient, its checkpoint must round-trip through
net_from_checkpoint, and it must trace at 882 for the C++ path.

Checks (all must pass; exit 1 otherwise):
  1. On N recorded obs-v2 states from the v6 bank (real 882 rows), the
     ext net (champion weights + zero adapters) and HeartsNetV5 (champion
     weights, 556 prefix) return IDENTICAL logits, value, belief
     (torch.equal, fp32, CPU).
  2. Adapter gradients are non-zero under a policy loss (they can learn).
  3. Save -> net_from_checkpoint -> HeartsNetV5Ext, identical outputs.
  4. torch.jit.trace at (1, 882) succeeds, rejects 556, and the traced
     output equals the eager output on the same batch.

Usage: python validate_v5ext.py [--n 4096]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

from hearts_net import HeartsNetV5, HeartsNetV5Ext, net_from_checkpoint
from cloud.shard_check import V3_REC

CHAMPION = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'


def load_states(n):
    files = sorted(glob.glob('expert_data/v6_bank/gen_*_t*.bin'))[:4]
    recs = []
    for p in files:
        raw = open(p, 'rb').read()
        assert raw[:4] == b'HMR3'
        recs.append(np.frombuffer(raw[32:], dtype=V3_REC))
    r = np.concatenate(recs)[:n]
    obs = torch.from_numpy(np.concatenate(
        [np.ascontiguousarray(r['obs']), np.ascontiguousarray(r['ext'])],
        axis=1)).float() / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(r['mask'])).bool()
    acts = torch.from_numpy(r['action'].astype(np.int64))
    return obs, mask, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4096)
    a = ap.parse_args()
    torch.manual_seed(0)
    obs, mask, acts = load_states(a.n)
    assert obs.shape[1] == 882, obs.shape

    sd = torch.load(CHAMPION, weights_only=True, map_location='cpu')
    v5 = HeartsNetV5(obs_dim=556, d_model=320, num_layers=6, num_heads=10)
    v5.load_state_dict(sd, strict=False); v5.eval()
    ext = HeartsNetV5Ext(d_model=320, num_layers=6, num_heads=10)
    missing, unexpected = ext.load_state_dict(sd, strict=False)
    assert set(missing) == {'ext_card_proj.weight', 'ext_card_proj.bias',
                            'ext_ctx_proj.weight', 'ext_ctx_proj.bias'}, missing
    assert not unexpected, unexpected
    ext.eval()

    # 1. bit-identity on the 556 prefix
    with torch.no_grad():
        l5, v5v, b5 = v5.forward_all(obs[:, :556], mask)
        le, ve, be = ext.forward_all(obs, mask)
    assert torch.equal(l5, le), 'logits differ'
    assert torch.equal(v5v, ve), 'value differs'
    assert torch.equal(b5, be), 'belief differs'
    print(f'check 1 PASS: bit-identical logits/value/belief on {len(obs)} obs-v2 states')

    # 2. adapters get gradient
    ext.train()
    logits, _, _ = ext.forward_all(obs[:256], mask[:256])
    logp = torch.log_softmax(logits, dim=1)
    loss = -logp.gather(1, acts[:256].unsqueeze(1)).mean()
    loss.backward()
    g1 = ext.ext_card_proj.weight.grad.abs().sum().item()
    g2 = ext.ext_ctx_proj.weight.grad.abs().sum().item()
    assert g1 > 0 and g2 > 0, (g1, g2)
    print(f'check 2 PASS: adapter grads non-zero (card {g1:.3g}, ctx {g2:.3g})')
    ext.zero_grad(); ext.eval()

    # 3. checkpoint round trip through the dispatcher
    tmp = 'v5ext_selftest.pth'
    torch.save(ext.state_dict(), tmp)
    net2 = net_from_checkpoint(tmp); os.remove(tmp)
    assert type(net2).__name__ == 'HeartsNetV5Ext', type(net2)
    net2.eval()
    with torch.no_grad():
        l2, _, _ = net2.forward_all(obs, mask)
    assert torch.equal(l2, le), 'round-trip outputs differ'
    print('check 3 PASS: net_from_checkpoint -> HeartsNetV5Ext, identical outputs')

    # 4. trace at 882
    traced = torch.jit.trace(ext, (torch.zeros(1, 882), torch.ones(1, 52, dtype=torch.bool)))
    with torch.no_grad():
        lt, vt = traced(obs[:64], mask[:64])
        le64, ve64 = ext(obs[:64], mask[:64])
    assert torch.equal(lt, le64) and torch.equal(vt, ve64), 'trace != eager'
    try:
        traced(torch.zeros(1, 556), torch.ones(1, 52, dtype=torch.bool))
        print('check 4 FAIL: trace accepted 556'); sys.exit(1)
    except Exception:
        pass
    print('check 4 PASS: traces at 882, rejects 556, trace == eager')
    print('ALL V5EXT NULL-CONTRACT CHECKS PASS')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print('FAIL:', e); sys.exit(1)
