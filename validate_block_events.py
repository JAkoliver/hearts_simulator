"""Addendum R null contract #1: independent validation of the engine's
BLOCK-EVENT register (docs/exploiter_league_r4_prereg.md §9.2/§9.4).

Registered rule. Before a trick resolves, let holders = seats with
round points > 0. A block event (W, pts) is recorded iff
  holders == 1 (call that seat S)  AND  points[S] >= 6
  AND trick index >= 3 (0-based; the 4th trick or later)
  AND the resolving trick carries pts >= 1 penalty points
  AND its winner W != S.

This script re-implements the rule in Python from the OBSERVABLE game
(round scores before each trick, the four cards played, the led suit)
and compares, after EVERY step, the engine's take_block_event() with the
expected value, over many random-policy deals with passing on and off.
It also asserts the two registered corner cases across the sample:
  - a deal in which a moon COMPLETES fired no event (by construction: no
    other seat ever won a point trick), and
  - no event ever fires on trick index < 3 (pre-4th-trick blocks) or when
    two or more seats already hold points.
Exit code 0 = PASS; any mismatch = FAIL with the offending deal printed.

Usage: python validate_block_events.py [--deals 4000] [--seed 5]
"""
import argparse
import random
import sys

import numpy as np

import hearts_env

HEART = 3            # Suit enum: Clubs=0, Diamonds=1, Spades=2, Hearts=3
QS_ID = 2 * 13 + 10  # spades, rank 12


def card(aid):
    return aid // 13, aid % 13 + 2   # (suit, rank)


def trick_points(cards):
    return sum(1 for a in cards if card(a)[0] == HEART) + (13 if QS_ID in cards else 0)


def trick_winner(seq):
    """seq: list of (seat, action_id) in play order."""
    led = card(seq[0][1])[0]
    best_seat, best_rank = seq[0][0], card(seq[0][1])[1]
    for seat, a in seq[1:]:
        s, r = card(a)
        if s == led and r > best_rank:
            best_seat, best_rank = seat, r
    return best_seat


def run_deal(env, rng, stats):
    """Play one deal (random legal actions) from a fresh env state; compare
    engine events to the recomputed rule at every step."""
    trick_seq = []           # (seat, action) for the trick in progress
    trick_idx = 0
    pre_scores = list(env.get_round_scores())
    moon_completed = False
    events_this_deal = 0
    while True:
        # passing phase: pick 3 random cards per seat (events impossible)
        legal = [a for a in env.get_legal_actions() if a != -1]
        seat = env.get_current_player()
        a = rng.choice(legal)
        was_passing = env.is_passing()
        res = env.step(a)
        ev = env.take_block_event()
        if was_passing:
            assert ev == (-1, 0), f'event during passing: {ev}'
            if not env.is_passing():
                pre_scores = list(env.get_round_scores())
            continue
        trick_seq.append((seat, a))
        expected = (-1, 0)
        if len(trick_seq) == 4:
            pts = trick_points([x[1] for x in trick_seq])
            w = trick_winner(trick_seq)
            holders = [i for i in range(4) if pre_scores[i] > 0]
            if (len(holders) == 1 and pre_scores[holders[0]] >= 6
                    and trick_idx >= 3 and pts >= 1 and w != holders[0]):
                expected = (w, pts)
            # bookkeeping for corner-case assertions
            if expected != (-1, 0):
                stats['events'] += 1
                events_this_deal += 1
                assert trick_idx >= 3
                assert len(holders) == 1
            # advance
            trick_idx += 1
            trick_seq = []
            if res.done:
                sc = list(env.get_round_scores())
                srt = sorted(sc)
                if srt[0] == 0 and srt[1:] == [26, 26, 26]:
                    moon_completed = True
                    stats['moons'] += 1
                    assert events_this_deal == 0, 'event fired in a completed-moon deal'
                stats['deals'] += 1
                break
            pre_scores = list(env.get_round_scores())
        assert ev == expected, (f'MISMATCH deal {stats["deals"]} trick {trick_idx} '
                                f'engine {ev} expected {expected} pre {pre_scores} '
                                f'seq {trick_seq}')
    return moon_completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deals', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    stats = {'deals': 0, 'events': 0, 'moons': 0}
    for passing in (False, True):
        env = hearts_env.HeartsEnv(seed=args.seed + int(passing), enable_passing=passing)
        env.reset()
        for _ in range(args.deals // 2):
            run_deal(env, rng, stats)
            env.reset()
    # vec surface: events must be surfaced through block_events_batch too
    vec = hearts_env.HeartsVecEnv(4, args.seed + 100)
    seen = 0
    for _ in range(3000):
        cp = vec.current_players()
        masks = vec.legal_mask_batch(np.arange(4))
        acts = np.array([rng.choice(np.flatnonzero(masks[i])) for i in range(4)])
        vec.step_batch(np.arange(4), acts)
        ev = vec.block_events_batch(np.arange(4))
        seen += int((ev[:, 0] >= 0).sum())
        assert ((ev[:, 0] >= 0) == (ev[:, 1] >= 1)).all(), 'vec event without points'
        ev2 = vec.block_events_batch(np.arange(4))
        assert (ev2[:, 0] == -1).all(), 'vec events not cleared on read'
    print(f'PASS: {stats["deals"]} random deals (passing off+on), engine events == '
          f'independent rule at every step; {stats["events"]} block events, '
          f'{stats["moons"]} completed moons (0 events in those, as required); '
          f'vec surface: {seen} events read-and-cleared correctly.')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print('FAIL:', e)
        sys.exit(1)
