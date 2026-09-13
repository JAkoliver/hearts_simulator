"""League r9 null contract (f): the trainer's per-deal moon detector
(round scores sum to 78 with exactly one 0 - train.py cell-B block) agrees
EXACTLY with the established accounting (match_eval._play_match: sorted
round scores == [0, 26, 26, 26]) on >= 2,000 random-play deals of
MatchVecEnv, and the moon seat matches. Exit 1 on any disagreement."""
import sys

import numpy as np

from hearts_match_env import MatchVecEnv

N_DEALS = 2500

if __name__ == '__main__':
    rng = np.random.default_rng(20260913)
    vec = MatchVecEnv(64, 990_000_000)
    ids = np.arange(64, dtype=np.int64)
    deals = moons = mismatches = 0
    while deals < N_DEALS:
        mask = vec.legal_mask_batch(ids)
        actions = np.array([int(rng.choice(np.flatnonzero(m))) for m in mask], dtype=np.int64)
        deal_dones, _, _, rs = vec.step_batch(ids, actions)
        for j in np.flatnonzero(deal_dones):
            r = np.asarray(rs[j], dtype=np.float64)
            # trainer detector
            t_moon = bool(r.sum() == 78.0 and (r == 0.0).sum() == 1)
            t_seat = int(np.flatnonzero(r == 0.0)[0]) if t_moon else -1
            # established accounting
            srt = np.sort(r)
            e_moon = bool(srt[0] == 0 and np.all(srt[1:] == 26))
            e_seat = int(np.argmin(r)) if e_moon else -1
            deals += 1
            moons += int(e_moon)
            if (t_moon, t_seat) != (e_moon, e_seat):
                mismatches += 1
                print(f'MISMATCH deal {deals}: scores {r} trainer ({t_moon},{t_seat}) '
                      f'established ({e_moon},{e_seat})')
    print(f'{deals} random-play deals, {moons} moons, {mismatches} detector mismatches')
    if mismatches:
        print('R9 CONTRACT (f) FAIL'); sys.exit(1)
    if moons == 0:
        print('R9 CONTRACT (f) INCONCLUSIVE: no moons occurred'); sys.exit(1)
    print('R9 CONTRACT (f) PASS: trainer moon detector == established accounting')
