"""One-time log migration: back-fill 'hands' into v2 deal lines.

WHY: reviews/insights/progress replay a match by re-dealing from the
logged seed and stepping the logged actions. Seed-based dealing is
std::shuffle-IMPLEMENTATION-bound (MSVC, libstdc++ and libc++ all
consume the RNG differently - HeartsEnv.hpp SetDeal comment), so logs
written by the Windows server desync when replayed by the Linux build.
Fix: every deal line carries its dealt hands ('hands' key); replay
installs them via set_deal. New lines get hands at write time
(server.py); THIS script back-fills history.

MUST RUN ON THE TOOLCHAIN THAT WROTE THE LOGS (the Windows dev
machine): only that build's dealer reproduces the original hands.

Usage (from repo root):
    python hearts_web/migrate_logs_add_hands.py [path/to/match_logs.jsonl]
Writes <path>.migrated + prints a report; original file untouched.
Verification: every migrated group is re-replayed through the
set_deal path (the exact path the Linux server will take).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hearts_match_env import MatchEnv  # noqa: E402


def hand_of(menv, seat):
    """Mirror of server.py hand_of (kept dependency-free)."""
    import numpy as np
    p = menv.get_current_player()
    if seat == p:
        obs = np.asarray(menv.env.observe(), dtype=np.float32)
        return np.flatnonzero(obs[:52] > 0).tolist()
    rows = np.asarray(menv.env.observe_opponent_hands(),
                      dtype=np.float32).reshape(3, 52)
    k = (seat - p) % 4
    return np.flatnonzero(rows[k - 1] > 0).tolist()


def replay_group(lines, use_logged_hands):
    """Replay one match's deal lines; returns per-line start hands.
    use_logged_hands=True verifies the set_deal path instead."""
    menv = MatchEnv(seed=lines[0]['seed'])
    out = []
    for d in lines:
        if use_logged_hands:
            menv.env.set_deal([[int(c) for c in h] for h in d['hands']])
        hands = [sorted(hand_of(menv, s)) for s in range(4)]
        out.append(hands)
        for s, card, ms in d['actions']:
            if menv.get_current_player() != s:
                raise RuntimeError(
                    f"desync sid={d.get('sid')} deal={d.get('deal_no')}")
            menv.step(card)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'match_logs.jsonl')
    raw = []
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                raw.append(json.loads(ln))

    # group v2 deal lines needing hands by (sid, match_no)
    groups = {}
    for i, line in enumerate(raw):
        if (line.get('v') == 2 and line.get('kind') == 'deal'
                and 'seed' in line and 'actions' in line
                and 'hands' not in line):
            groups.setdefault((line['sid'], line.get('match_no')),
                              []).append(i)

    migrated = skipped = 0
    for key, idxs in sorted(groups.items()):
        lines = sorted((raw[i] for i in idxs), key=lambda d: d['deal_no'])
        if [d['deal_no'] for d in lines] != list(
                range(lines[0]['deal_no'], lines[0]['deal_no'] + len(lines))) \
                or lines[0]['deal_no'] != 1:
            print(f'SKIP {key}: non-contiguous deals '
                  f'{[d["deal_no"] for d in lines]}')
            skipped += len(idxs)
            continue
        try:
            hands_per_deal = replay_group(lines, use_logged_hands=False)
        except RuntimeError as e:
            print(f'SKIP {key}: {e}')
            skipped += len(idxs)
            continue
        for d, hands in zip(lines, hands_per_deal):
            d['hands'] = hands
        # verify: the set_deal path (what the other toolchain will run)
        replay_group(lines, use_logged_hands=True)
        migrated += len(idxs)

    out_path = path + '.migrated'
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in raw:
            f.write(json.dumps(line, separators=(',', ':')) + '\n')
    print(f'{migrated} deal lines migrated, {skipped} skipped, '
          f'{len(raw)} total lines -> {out_path}')


if __name__ == '__main__':
    main()
