"""Stage 1 validator for HeartsNetV6 (docs/v6_prereg.md) — halt-default.

 1. Contract: 556-dim input raises (no silent v1 feeding).
 2. Mask honoring: -inf exactly on illegal, finite on legal.
 3. Determinism: repeated forwards identical.
 4. Derived seat features (points, leads-now) match env ground truth.
 5. Aux isolation: randomizing aux-head weights leaves policy/value/
    belief bit-identical (aux heads never feed the trunk).
 6. Gradient flow: a loss over all five outputs reaches every
    parameter EXCEPT the deliberately-unused ones (none expected).
 7. Parameter count report (prereg size: d448 L8 ~ 2.6x v5-M).
"""
import numpy as np
import torch

from hearts_match_env import MatchEnv
from hearts_net import HeartsNetV5, HeartsNetV6

PEN = np.zeros(52, dtype=np.float64)
PEN[39:52] = 1.0
PEN[36] = 13.0


def collect_states(seed, n):
    rng = np.random.default_rng(seed)
    menv = MatchEnv(seed)
    out = []
    while len(out) < n and not menv.match_over:
        seat = menv.get_current_player()
        legal = [a for a in menv.get_legal_actions() if a != -1]
        if not menv.is_passing():
            mask = np.zeros(52, dtype=bool)
            mask[legal] = True
            rs = np.array(menv.env.get_round_scores(), dtype=np.float64)
            rel_pts = np.array([rs[(seat + k) % 4] for k in range(4)]) / 26.0
            out.append((menv.observe_v2(), mask, rel_pts))
        menv.step(int(rng.choice(legal)))
    return out


def main():
    torch.manual_seed(0)
    net = HeartsNetV6()
    net.eval()
    n_par = sum(p.numel() for p in net.parameters())
    v5 = sum(p.numel() for p in HeartsNetV5(d_model=320, num_layers=6,
                                            num_heads=10).parameters())
    print(f'params: v6 {n_par/1e6:.2f}M vs v5-M {v5/1e6:.2f}M '
          f'({n_par/v5:.2f}x)')

    states = collect_states(51_000, 256)
    obs = torch.tensor(np.stack([s[0] for s in states]))
    mask = torch.tensor(np.stack([s[1] for s in states]))
    rel_pts = torch.tensor(np.stack([s[2] for s in states]),
                           dtype=torch.float64)

    # 1. contract
    try:
        net(obs[:2, :556], mask[:2])
        raise AssertionError('556-dim input did not raise')
    except ValueError:
        print('input contract (556 raises): PASS')

    # 2/3. masking + determinism
    with torch.no_grad():
        l1, v1 = net(obs, mask)
        l2, v2 = net(obs, mask)
    assert torch.equal(l1, l2) and torch.equal(v1, v2), 'nondeterministic'
    assert torch.isinf(l1[~mask]).all() and (l1[~mask] < 0).all(), 'mask leak'
    assert torch.isfinite(l1[mask]).all(), 'legal logits not finite'
    print(f'masking + determinism on {obs.shape[0]} states: PASS')

    # 4. derived seat features vs env ground truth
    taken = obs[:, 660:868].reshape(-1, 4, 52).double()
    pts = taken @ torch.tensor(PEN / 26.0)
    assert torch.allclose(pts, rel_pts, atol=1e-9), \
        'derived points != env round_scores (relative)'
    led_now = (obs[:, 608:660] * obs[:, 52:104]).double()
    who = obs[:, 290:498].reshape(-1, 4, 52).double()
    leads = (who * led_now.unsqueeze(1)).sum(-1)
    trick_live = obs[:, 52:104].sum(-1) > 0
    assert torch.all(leads.sum(-1)[trick_live] == 1.0), \
        'live trick must have exactly one leader'
    assert torch.all(leads.sum(-1)[~trick_live] == 0.0), \
        'fresh trick must have no leader'
    print('derived seat features (points, leads-now): PASS')

    # 5. aux isolation
    with torch.no_grad():
        base = net.forward_all(obs[:64], mask[:64])
        for p in list(net.moon_head.parameters()) \
                + list(net.points_head.parameters()):
            p.add_(torch.randn_like(p))
        after = net.forward_all(obs[:64], mask[:64])
    assert all(torch.equal(a, b) for a, b in zip(base, after)), \
        'aux heads leaked into standard outputs'
    print('aux isolation: PASS')

    # 6. gradient flow
    net.train()
    ml, val, bel, moon, spts = net.forward_aux(obs[:32], mask[:32])
    probs = torch.softmax(ml.masked_fill(~torch.isfinite(ml), -1e9), -1)
    loss = probs.mean() + val.mean() + bel.mean() + moon.mean() + spts.mean()
    loss.backward()
    missing = [n for n, p in net.named_parameters()
               if p.grad is None or p.grad.abs().sum() == 0]
    allowed = set()   # nothing is expected to be grad-dead
    bad = [m for m in missing if m not in allowed]
    assert not bad, f'grad-dead parameters: {bad}'
    print('gradient flow to every parameter: PASS')
    print('ALL V6 STAGE-1 CHECKS PASS')


if __name__ == '__main__':
    main()
