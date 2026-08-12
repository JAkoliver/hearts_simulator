"""Stage 2 validator for record format v3 (docs/v6_prereg.md) — halt-default.

Parses .hmr3 files (HMR3 header + 1180-byte records) and hard-asserts:
 - header magic/version/record_size; body size divisible by 1180
 - action legality (mask set, card in hand)
 - obs/ext cross-consistency at the QUANTIZED level: on-table cards
   have position but no capturer; history cards have exactly one
   capturer; unseen cards have neither; QS one-hot matches the planes;
   moon-alive flags match penalty points computed FROM the planes;
   hearts-unseen matches the position channel
 - belief labels disjoint from own hand and from played cards,
   per-opponent counts <= 13
 - deal outcome labels: fin sums to 26 (no moon) or 78 (moon, with
   mooned_by holding 0 and everyone else 26)
 - reward in the (2.5 - placement) * 4 value set
 - shooter matches (flags bit2) contain records from EXACTLY 3 seats
   (the attacker is never recorded); natural matches from 4
 - schedule check: bit2 exactly on matches where (m + tid) % 8 == 7

Usage: python validate_v3_records.py <file.bin> [more files...]
"""
import sys

import numpy as np

REC = np.dtype([
    ('obs', 'u1', 556), ('ext', 'u1', 326), ('mask', 'u1', 52),
    ('labels', 'u1', 156), ('pi', 'u1', 52), ('action', '<u2'),
    ('seat', '<u2'), ('reward', '<f4'), ('eq_best', '<f4'),
    ('eq_second', '<f4'), ('gap_se', '<f4'), ('second_action', '<u2'),
    ('n_dets', '<u2'), ('match_id', '<u4'), ('flags', '<u2'),
    ('reserved', '<u2'), ('fin', 'u1', 4), ('moonby', 'i1'), ('pad', 'u1')])
assert REC.itemsize == 1180, REC.itemsize

PEN = np.zeros(52)
PEN[39:52] = 1.0
PEN[36] = 13.0
REWARDS = {6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0}


def check_file(path):
    raw = open(path, 'rb').read()
    assert raw[:4] == b'HMR3', f'{path}: bad magic {raw[:4]}'
    version = int.from_bytes(raw[4:6], 'little')
    rsize = int.from_bytes(raw[6:8], 'little')
    tid = int.from_bytes(raw[12:14], 'little')
    assert version == 3 and rsize == 1180, (version, rsize)
    body = raw[32:]
    assert len(body) % 1180 == 0, f'{path}: torn tail {len(body) % 1180}'
    r = np.frombuffer(body, dtype=REC)
    n = len(r)
    assert n > 0, 'empty file'

    on_table = r['obs'][:, 52:104] > 127
    history = r['obs'][:, 104:156] > 127
    hand = r['obs'][:, 0:52] > 127
    pos = r['ext'][:, 0:52]
    led = r['ext'][:, 52:104] > 127
    planes = (r['ext'][:, 104:312].reshape(n, 4, 52) > 127)
    n_capt = planes.sum(axis=1)   # capturers per card (0 or 1)

    # action legality
    idx = np.arange(n)
    assert (r['mask'][idx, r['action']] == 1).all(), 'action not in mask'
    assert hand[idx, r['action']].all(), 'action not from hand'
    assert (r['seat'] < 4).all()

    # obs/ext cross-consistency
    assert (n_capt <= 1).all(), 'card captured by two seats'
    assert (n_capt[on_table] == 0).all(), 'on-table card already captured'
    assert (pos[on_table] > 0).all(), 'on-table card lacks position'
    assert (n_capt[history] == 1).all(), 'history card lacks capturer'
    assert (pos[history] > 0).all(), 'history card lacks position'
    unseen = ~(on_table | history)
    assert (n_capt[unseen] == 0).all(), 'unseen card captured'
    assert (pos[unseen] == 0).all(), 'unseen card has position'
    assert (~led | (pos > 0)).all(), 'led flag on positionless card'

    # QS one-hot vs planes
    qs = r['ext'][:, 321:326] > 127
    assert (qs.sum(axis=1) == 1).all(), 'QS one-hot not one-hot'
    qs_capt_plane = planes[:, :, 36]          # (n, 4) rel-seat capturer
    has_qs = qs_capt_plane.any(axis=1)
    assert (qs[:, 0] == ~has_qs).all(), 'QS uncaptured slot wrong'
    rel = qs_capt_plane.argmax(axis=1)
    ok = ~has_qs | (qs[idx, 1 + rel])
    assert ok.all(), 'QS captured slot mismatch'

    # moon-alive vs plane-derived points (relative frame throughout)
    pts = planes @ PEN                        # (n, 4) points per rel seat
    others = pts.sum(axis=1, keepdims=True) - pts
    alive = r['ext'][:, 316:320] > 127
    assert (alive == (others == 0)).all(), 'moon-alive mismatch'

    # hearts unseen vs position channel
    seen_h = (pos[:, 39:52] > 0).sum(axis=1)
    want = np.rint((13 - seen_h) / 13.0 * 255.0)
    assert (np.abs(r['ext'][:, 320].astype(float) - want) <= 1).all(), \
        'hearts-unseen mismatch'

    # belief labels
    lab = r['labels'].reshape(n, 3, 52) > 0
    assert (lab.sum(axis=2) <= 13).all(), 'opponent hand > 13'
    assert not (lab.any(axis=1) & hand).any(), 'belief overlaps own hand'
    assert not (lab.any(axis=1) & history).any(), 'belief overlaps played'

    # deal outcome labels
    fin_sum = r['fin'].sum(axis=1)
    assert np.isin(fin_sum, (26, 78)).all(), 'fin sum not 26/78'
    moon = fin_sum == 78
    assert (r['moonby'][~moon] == -1).all(), 'moonby set without moon'
    mb = r['moonby'][moon].astype(int)
    assert (mb >= 0).all() and (mb < 4).all(), 'bad moonby'
    assert (r['fin'][moon, mb] == 0).all(), 'mooning seat has points'

    # rewards
    assert np.isin(r['reward'], list(REWARDS)).all(), 'bad reward value'

    # shooter schedule + attacker exclusion (per-thread match order is
    # file order; match_id = match_base + m with m increasing)
    mids = r['match_id']
    order = {mid: i for i, mid in enumerate(dict.fromkeys(mids.tolist()))}
    moons_sh = moons_nat = 0
    for mid in order:
        rows = r[mids == mid]
        m = order[mid]
        sched = (m + tid) % 8 == 7
        bit2 = (rows['flags'] & 4) > 0
        assert bit2.all() == sched and bit2.any() == sched, \
            f'match {mid}: shooter flag != schedule'
        seats = set(rows['seat'].tolist())
        assert len(seats) == (3 if sched else 4), \
            f'match {mid}: {len(seats)} recorded seats (shooter={sched})'
        dm = (rows['fin'].sum(axis=1) == 78).any()
        if sched:
            moons_sh += dm
        else:
            moons_nat += dm
    pass_frac = ((r['flags'] & 1) > 0).mean()
    assert 0.05 < pass_frac < 0.30, f'pass fraction {pass_frac:.3f} off'
    return (n, len(order), sum((m + tid) % 8 == 7 for m in order.values()),
            moons_sh, moons_nat)


def main():
    tot = np.zeros(5, dtype=int)
    for path in sys.argv[1:]:
        res = check_file(path)
        tot += res
        print(f'{path}: {res[0]} records, {res[1]} matches '
              f'({res[2]} shooter), moons sh/nat {res[3]}/{res[4]} — PASS')
    print(f'TOTAL: {tot[0]} records, {tot[1]} matches ({tot[2]} shooter), '
          f'matches-with-moons shooter/natural {tot[3]}/{tot[4]}')
    print('ALL V3 RECORD CHECKS PASS')


if __name__ == '__main__':
    main()
