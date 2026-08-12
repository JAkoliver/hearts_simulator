"""Stage 0 validator for obs v2 (docs/v6_prereg.md) — halt-default.

Checks, all hard assertions:
 1. A/A determinism: same seed + same action stream -> identical
    observe_v2 at every decision.
 2. Prefix identity: observe_v2()[:556] == observe() exactly.
 3. Capture invariants at every non-terminal play decision, against an
    INDEPENDENT rules re-implementation (winner recursion):
      - taken-by planes reproduce each seat's round_scores exactly
      - taken-by matches the recursively computed trick winners
      - tricks-won counts match
      - position/led channels match the recorded play order
      - moon-alive flags consistent with round_scores
      - hearts-unseen and QS one-hot match the play record
 4. v5 bit-identity: the 8a89da90 milestone produces IDENTICAL logits
    and value on 556-dim and 882-dim inputs (the extension must be
    invisible to v1 consumers).

Usage: python validate_obs_v2.py [n_matches]
"""
import sys

import numpy as np
import torch

from hearts_match_env import MatchEnv
from hearts_net import net_from_checkpoint

MILESTONE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
PEN = {36: 13, **{c: 1 for c in range(39, 52)}}   # QS + hearts
QS = 36


def run_match(seed, collect):
    """Drive one match with a seeded random-legal policy; call
    `collect(menv, tracker)` at every non-terminal play decision."""
    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    menv = MatchEnv(seed)
    # independent play record for the recursion check, per deal:
    trk = {'plays': [], 'deal_done': False}

    def fresh_deal():
        trk['plays'] = []          # [(seat, card)] in play order

    fresh_deal()
    guard = 0
    while not menv.match_over and guard < 60000:
        guard += 1
        seat = menv.get_current_player()
        legal = [a for a in menv.get_legal_actions() if a != -1]
        a = int(rng.choice(legal))
        if not menv.is_passing():
            trk['plays'].append((seat, a))
            collect(menv, trk)     # observe BEFORE the action applies
        deal_done, match_done, _ = menv.step(a)
        if deal_done:
            fresh_deal()
    assert guard < 60000, 'runaway match'
    return menv


def recursion_state(plays):
    """Independent winner recursion over the play record so far.
    Returns (taken_by{card: seat}, tricks_won[4], pos{card}, led{card})."""
    taken, won = {}, [0, 0, 0, 0]
    pos, led = {}, set()
    trick = []
    for seat, card in plays:
        pos[card] = len(trick)
        if len(trick) == 0:
            led.add(card)
        trick.append((seat, card))
        if len(trick) == 4:
            led_suit = trick[0][1] // 13
            best_rank, winner = -1, trick[0][0]
            for s, c in trick:
                if c // 13 == led_suit and c % 13 > best_rank:
                    best_rank, winner = c % 13, s
            for s, c in trick:
                taken[c] = winner
            won[winner] += 1
            trick = []
    return taken, won, pos, led


def main():
    n_matches = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    checks = {'decisions': 0, 'obs': []}

    def collect(menv, trk):
        # the play just appended has NOT been stepped yet - drop it for
        # the "state before acting" recursion
        plays = trk['plays'][:-1]
        obs = menv.observe_v2()
        v1 = menv.observe()
        assert obs.shape == (882,), obs.shape
        assert np.array_equal(obs[:556], v1), 'prefix identity broken'
        ext = obs[556:]
        taken, won, pos, led = recursion_state(plays)
        rs = list(menv.env.get_round_scores())
        me = menv.get_current_player()
        # taken-by planes + per-seat sums
        for k in range(4):
            seat = (me + k) % 4
            plane = ext[104 + 52 * k:104 + 52 * (k + 1)]
            want = np.zeros(52, dtype=np.float32)
            pts = 0
            for c, s in taken.items():
                if s == seat:
                    want[c] = 1.0
                    pts += PEN.get(c, 0)
            assert np.array_equal(plane, want), f'taken-by plane k={k}'
            assert pts == rs[seat], f'points sum {pts} != {rs[seat]}'
            assert ext[312 + k] == np.float32(won[seat] / 13.0), 'tricks won'
            alive = all(rs[t] == 0 for t in range(4) if t != seat)
            assert ext[316 + k] == (1.0 if alive else 0.0), 'moon-alive'
        # position + led channels (include the current partial trick,
        # which the env HAS recorded for cards already on the table)
        full_pos = dict(pos)
        cur = plays[len(plays) - len(plays) % 4:]
        for i, (s, c) in enumerate(cur):
            full_pos[c] = i
            if i == 0:
                led.add(c)
        for c in range(52):
            want_p = np.float32((full_pos[c] + 1) / 4.0) if c in full_pos else 0.0
            assert ext[c] == want_p, f'position ch card {c}'
            assert ext[52 + c] == (1.0 if c in led else 0.0), f'led ch {c}'
        # hearts unseen + QS one-hot
        seen_h = sum(1 for _, c in plays if c >= 39)
        assert ext[320] == np.float32((13 - seen_h) / 13.0), 'hearts unseen'
        qs = np.zeros(5, dtype=np.float32)
        if QS in taken:
            qs[1 + ((taken[QS] - me) % 4)] = 1.0
        else:
            qs[0] = 1.0
        assert np.array_equal(ext[321:326], qs), 'QS one-hot'
        checks['decisions'] += 1
        if len(checks['obs']) < 512:
            checks['obs'].append(obs)

    for m in range(n_matches):
        run_match(31_000 + m, collect)
    print(f'invariants: {checks["decisions"]} decisions across '
          f'{n_matches} matches, all checks PASS')

    # A/A determinism: identical seeds -> identical obs streams
    stream = []
    run_match(77_001, lambda e, t: stream.append(e.observe_v2()))
    stream2 = []
    run_match(77_001, lambda e, t: stream2.append(e.observe_v2()))
    assert len(stream) == len(stream2) and all(
        np.array_equal(a, b) for a, b in zip(stream, stream2)), 'A/A broken'
    print(f'A/A determinism: {len(stream)} decisions identical PASS')

    # v5 bit-identity on 556 vs 882
    net = net_from_checkpoint(MILESTONE)
    net.eval()
    obs = torch.tensor(np.stack(checks['obs']))
    mask = torch.ones(obs.shape[0], 52, dtype=torch.bool)
    with torch.no_grad():
        l556, v556 = net(obs[:, :556], mask)
        l882, v882 = net(obs, mask)
    assert torch.equal(l556, l882) and torch.equal(v556, v882), \
        'v5 outputs differ between 556 and 882 inputs'
    print(f'v5 bit-identity (8a89da90) on {obs.shape[0]} states: PASS')
    print('ALL OBS-V2 STAGE-0 CHECKS PASS')


if __name__ == '__main__':
    main()
