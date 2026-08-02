"""Replay + analysis of web matches from match_logs.jsonl (log format v2).

The v2 replay contract: MatchEnv(seed) + the logged per-deal action
sequences reproduce a match bit-exactly (verified per deal against the
logged round scores). On top of the replay this reports, per deal:

- point flow and moon detection (human AND AI moons);
- MISSED BLOCKS: the human holds the trick, an AI still holding a higher
  lead-suit card ducks under it;
- PRE-DUCKS (counterfactual): the AI acted BEFORE the human in a trick
  the human won, holding a card that would have beaten the human's
  eventual winner - labeled counterfactual because the human might have
  played differently;
- FED POINTS: a void AI discards a point card (heart / QS) into a trick
  the human won while holding a pointless discard - normal-Hearts
  dumping instinct inverted into moon fuel;
- PASS FEEDS: cards each AI passed that ended up in the human's played
  set (receiver identified with no direction convention needed - a card
  can only be played by the seat that holds it).

Every finding carries human_pts_before and a moon_alive flag (the human
holds ALL points taken so far) - ducking/dumping is CORRECT normal play,
so only moon-alive findings indict the defense. See the 2026-08-02
exploit-session analysis.

Usage:
  python hearts_web/analyze_matches.py            # latest human match
  python hearts_web/analyze_matches.py --sid SID  # a specific match
  python hearts_web/analyze_matches.py --all      # every v2 match
                                                  # (+ abandoned sessions)
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


def moon_shooter(round_scores):
    """Seat index of a successful moon, else None."""
    for s, v in enumerate(round_scores):
        if v == 0 and all(x == 26 for i, x in enumerate(round_scores) if i != s):
            return s
    return None


def analyze_match(match, deals):
    h = match['human_seat']
    print(f"match {match['sid']}: seed {match['seed']}, human seat {h}, "
          f"{match['deals']} deals, {match['n_actions']} actions, "
          f"{match['duration_s']}s, model {match['model']}")
    print(f"final {match['final']} placements {match['placements']}\n")

    menv = MatchEnv(seed=match['seed'])
    tot = {'block': 0, 'block_alive': 0, 'pre': 0, 'pre_alive': 0,
           'fed': 0, 'fed_alive': 0}
    for d in sorted(deals, key=lambda x: x['deal_no']):
        play_seq, pass_by = [], {s: [] for s in range(4)}
        rs = None
        for seat, card, ms in d['actions']:
            assert menv.get_current_player() == seat, 'replay desync'
            if menv.is_passing():
                pass_by[seat].append(card)
            else:
                play_seq.append((seat, card))
            _, _, rs = menv.step(card)
        ok = list(map(int, rs)) == d['round_scores']
        human_played = set(c for s, c in play_seq if s == h)
        remaining = {s: set(c for ss, c in play_seq if ss == s)
                     for s in range(4)}
        tricks = [play_seq[i:i + 4] for i in range(0, 52, 4)]
        taken = [0, 0, 0, 0]
        timeline, findings = [], []

        for ti, tr in enumerate(tricks):
            lead = tr[0][1] // 13
            alive = sum(taken) == taken[h]  # human holds ALL points so far
            # Final winner of the trick.
            fw, fwc = tr[0]
            for s, c in tr[1:]:
                if c // 13 == lead and c % 13 > fwc % 13:
                    fw, fwc = s, c
            hpos = next((i for i, (s, _) in enumerate(tr) if s == h), None)

            cur_ws, cur_wc = tr[0]
            for i, (s, c) in enumerate(tr):
                if i == 0:
                    continue
                if s != h and cur_ws == h:
                    # MISSED BLOCK: human holds the trick at this decision.
                    higher = sorted(x for x in remaining[s]
                                    if x // 13 == lead and x % 13 > cur_wc % 13)
                    beat = c // 13 == lead and c % 13 > cur_wc % 13
                    if higher and not beat:
                        findings.append(('block', ti, s, alive, taken[h],
                                         f"played {name(c)}, held "
                                         f"{','.join(name(x) for x in higher)}"))
                if c // 13 == lead and c % 13 > cur_wc % 13:
                    cur_ws, cur_wc = s, c

            if fw == h:
                for i, (s, c) in enumerate(tr):
                    if s == h:
                        continue
                    if hpos is not None and i < hpos:
                        # PRE-DUCK (counterfactual): acted before the human,
                        # held a card beating the human's eventual winner.
                        higher = sorted(x for x in remaining[s]
                                        if x // 13 == lead and x % 13 > fwc % 13)
                        beat = c // 13 == lead and c % 13 > fwc % 13
                        if higher and not beat:
                            findings.append(('pre', ti, s, alive, taken[h],
                                             f"played {name(c)} before you; "
                                             f"held {','.join(name(x) for x in higher)}"))
                    if c // 13 != lead and pts(c) > 0:
                        # FED POINTS: void discard of a point card into the
                        # human's trick, with a pointless discard in hand.
                        safe = sorted(x for x in remaining[s]
                                      if x != c and pts(x) == 0)
                        findings.append(('fed', ti, s, alive, taken[h],
                                         f"threw {name(c)} into your trick"
                                         + (f"; safe discards held: "
                                            f"{','.join(name(x) for x in safe[:6])}"
                                            if safe else " (held only points)")))

            tpts = sum(pts(c) for _, c in tr)
            for s, c in tr:
                remaining[s].discard(c)
            taken[fw] += tpts
            if tpts:
                who = 'YOU' if fw == h else f'P{fw + 1}'
                timeline.append(f"T{ti + 1}:{who}+{tpts}")

        shooter = moon_shooter(d['round_scores'])
        moon_tag = ('' if shooter is None else
                    ' HUMAN MOON' if shooter == h else
                    f' PLAYER {shooter + 1} (AI) MOON')
        print(f"deal {d['deal_no']} [{'replay OK' if ok else 'REPLAY FAIL'}] "
              f"round={d['round_scores']}{moon_tag}")
        fed_to_h = {s: [c for c in pass_by[s] if c in human_played]
                    for s in range(4) if s != h and pass_by[s]}
        for s, cards in fed_to_h.items():
            if cards:
                hot = [name(c) for c in cards if pts(c) > 0 or c % 13 >= 10]
                print(f"  pass: Player {s + 1} passed you "
                      f"{','.join(name(c) for c in cards)}"
                      + (f"  <- high/point cards: {','.join(hot)}" if hot else ''))
        print(f"  points: {' '.join(timeline) if timeline else 'none'}")
        label = {'block': 'MISSED BLOCK', 'pre': 'PRE-DUCK (counterfactual)',
                 'fed': 'FED POINTS'}
        for kind, ti, s, alive, hp, desc in findings:
            tag = ' [MOON ALIVE]' if alive and shooter == h else ''
            print(f"  {label[kind]} trick {ti + 1}: Player {s + 1} {desc} "
                  f"(human pts {hp}){tag}")
            tot[kind] += 1
            if alive and shooter == h:
                tot[kind + '_alive'] += 1

    print(f"\nTOTALS: blocks {tot['block']} ({tot['block_alive']} moon-alive) | "
          f"pre-ducks {tot['pre']} ({tot['pre_alive']} moon-alive) | "
          f"fed points {tot['fed']} ({tot['fed_alive']} moon-alive)")
    print("moon-alive counts = failures while the human held every point of "
          "a deal they went on to sweep; the rest is normal point avoidance.\n")


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
    if not matches and not args.all:
        raise SystemExit('no matching v2 match lines found')
    for m in matches:
        deals = [l for l in v2 if l['kind'] == 'deal' and l['sid'] == m['sid']]
        try:
            analyze_match(m, deals)
        except Exception as e:  # keep going: one bad line != dead dataset
            print(f"WARNING: match {m['sid']} failed to analyze: {e}\n")
    if args.all:
        done = {m['sid'] for m in matches}
        orphans = {}
        for l in v2:
            if l['kind'] == 'deal' and l['sid'] not in done:
                orphans.setdefault(l['sid'], []).append(l)
        for sid, ds in orphans.items():
            print(f"abandoned session {sid}: {len(ds)} completed deal(s) "
                  f"logged (pid {ds[0].get('pid')}, seed {ds[0]['seed']})")


if __name__ == '__main__':
    main()
