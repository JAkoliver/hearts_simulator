"""Replay + analysis of web matches from match_logs.jsonl (log format v2).

The v2 replay contract: MatchEnv(seed) + the logged per-deal action
sequences reproduce a match bit-exactly (verified per deal against the
logged round scores). On top of the replay this reports per-deal point
flow, human moons, and MISSED-BLOCK opportunities - tricks the human was
winning where an AI held a higher card of the led suit and chose not to
beat it (the moon-defense hole's per-decision signature; see the
2026-08-02 exploit-session analysis).

Caveats: in-suit blocks only (discard-choice failures are invisible);
ducking is CORRECT normal play - judge blocks by the human_pts context.

Usage:
  python hearts_web/analyze_matches.py            # latest human match
  python hearts_web/analyze_matches.py --sid SID  # a specific match
  python hearts_web/analyze_matches.py --all      # every v2 match
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hearts_match_env import MatchEnv  # noqa: E402

SUITS = 'CDSH'
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'match_logs.jsonl')


def name(c):
    return RANKS[c % 13] + SUITS[c // 13]


def pts(c):
    return 1 if c // 13 == 3 else (13 if c == 36 else 0)


def analyze_match(match, deals):
    h = match['human_seat']
    print(f"match {match['sid']}: seed {match['seed']}, human seat {h}, "
          f"{match['deals']} deals, {match['n_actions']} actions, "
          f"{match['duration_s']}s, model {match['model']}")
    print(f"final {match['final']} placements {match['placements']}\n")

    menv = MatchEnv(seed=match['seed'])
    all_blocks = []
    for d in sorted(deals, key=lambda x: x['deal_no']):
        play_seq = []
        rs = None
        for seat, card, ms in d['actions']:
            assert menv.get_current_player() == seat, 'replay desync'
            if not menv.is_passing():
                play_seq.append((seat, card))
            _, _, rs = menv.step(card)
        ok = list(map(int, rs)) == d['round_scores']
        remaining = {s: set(c for ss, c in play_seq if ss == s)
                     for s in range(4)}
        tricks = [play_seq[i:i + 4] for i in range(0, 52, 4)]
        taken = [0, 0, 0, 0]
        timeline, dblocks = [], []
        for ti, tr in enumerate(tricks):
            lead = tr[0][1] // 13
            cur_ws, cur_wc = tr[0]
            for s, c in tr[1:]:
                # Block check BEFORE updating the winner: at this seat's
                # decision the human holds the trick - could it be beaten?
                if s != h and cur_ws == h:
                    higher = sorted(x for x in remaining[s]
                                    if x // 13 == lead and x % 13 > cur_wc % 13)
                    beat = c // 13 == lead and c % 13 > cur_wc % 13
                    if higher and not beat:
                        dblocks.append({'trick': ti + 1, 'ai': s,
                                        'played': name(c),
                                        'could_have': [name(x) for x in higher],
                                        'human_pts_before': taken[h],
                                        'trick_pts': sum(pts(c2) for _, c2 in tr)})
                if c // 13 == lead and c % 13 > cur_wc % 13:
                    cur_ws, cur_wc = s, c
            tpts = sum(pts(c) for _, c in tr)
            for s, c in tr:
                remaining[s].discard(c)
            taken[cur_ws] += tpts
            if tpts:
                who = 'YOU' if cur_ws == h else f'P{cur_ws + 1}'
                timeline.append(f"T{ti + 1}:{who}+{tpts}")
        moon = d['round_scores'][h] == 0 and all(
            v == 26 for i, v in enumerate(d['round_scores']) if i != h)
        print(f"deal {d['deal_no']} [{'replay OK' if ok else 'REPLAY FAIL'}] "
              f"round={d['round_scores']}{' HUMAN MOON' if moon else ''}")
        print(f"  points: {' '.join(timeline) if timeline else 'none'}")
        for b in dblocks:
            print(f"  MISSED BLOCK trick {b['trick']}: Player {b['ai'] + 1} "
                  f"played {b['played']}, held {','.join(b['could_have'])} "
                  f"(human had {b['human_pts_before']} pts, trick worth "
                  f"{b['trick_pts']})")
        all_blocks += [dict(b, deal=d['deal_no'], moon=moon) for b in dblocks]

    late = [b for b in all_blocks if b['moon'] and b['human_pts_before'] >= 8]
    print(f"\nTOTALS: {len(all_blocks)} missed-block opportunities; "
          f"{len([b for b in all_blocks if b['moon']])} in moon deals; "
          f"{len(late)} with the human already holding >=8 points.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=LOG_PATH)
    ap.add_argument('--sid', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--include-test', action='store_true',
                    help='include smoke-test matches')
    args = ap.parse_args()

    lines = [json.loads(l) for l in open(args.log)]
    v2 = [l for l in lines if l.get('v') == 2]
    if not args.include_test:
        v2 = [l for l in v2 if l.get('pid') != 'smoke-test-pid']
    matches = [l for l in v2 if l['kind'] == 'match']
    if args.sid:
        matches = [m for m in matches if m['sid'] == args.sid]
    elif not args.all:
        matches = matches[-1:]
    if not matches:
        raise SystemExit('no matching v2 match lines found')
    for m in matches:
        deals = [l for l in v2 if l['kind'] == 'deal' and l['sid'] == m['sid']]
        analyze_match(m, deals)


if __name__ == '__main__':
    main()
