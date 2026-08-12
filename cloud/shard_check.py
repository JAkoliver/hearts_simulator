"""Record-level sanity validation for self-play shards (REQUIREMENTS R1/R2).

A shard is only marked complete once it passes these checks - run by the
worker before upload (fail fast) and again by the orchestrator on receipt
(authoritative; workers are untrusted).
"""
import os

import numpy as np

RECORD = np.dtype([
    ('obs', 'u1', 550),
    ('mask', 'u1', 52),
    ('labels', 'u1', 156),
    ('pi', 'u1', 52),
    ('action', '<u2'),
    ('seat', '<u2'),
    ('reward', '<f4'),
])
assert RECORD.itemsize == 818

# Each deal's four seat-rewards are (avg - own): they sum to 0 exactly in
# spirit, to fp round-off in practice. A shard is many whole deals, so the
# grand mean must sit near 0; 0.05 is ~two orders above observed round-off
# and ~two below any real corruption signature.
REWARD_MEAN_TOL = 0.05
MIN_RECORDS_PER_DEAL = 20   # a 4-player deal always logs >= ~52 decisions;
                            # pass-phase-less deals still exceed 20


def validate_shard(path, expect_deals=None):
    """Returns (ok: bool, detail: str). Never raises on bad data."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as e:
        return False, f"unreadable: {e}"
    if data.size == 0:
        return False, "empty file"
    if data.size % RECORD.itemsize != 0:
        return False, f"size {data.size} not a multiple of {RECORD.itemsize}"
    rec = data.view(RECORD)
    n = len(rec)

    if (rec['action'] >= 52).any():
        return False, "action out of range"
    if (rec['seat'] >= 4).any():
        return False, "seat out of range"
    legal = rec['mask'][np.arange(n), rec['action']]
    if not legal.all():
        bad = int((~legal.astype(bool)).sum())
        return False, f"{bad} recorded actions illegal under their own mask"
    if (rec['mask'].sum(axis=1) == 0).any():
        return False, "record with empty legal mask"
    rmean = float(rec['reward'].mean())
    if not np.isfinite(rec['reward']).all():
        return False, "non-finite rewards"
    if abs(rmean) > REWARD_MEAN_TOL:
        return False, f"reward mean {rmean:+.4f} exceeds {REWARD_MEAN_TOL}"
    if expect_deals is not None and n < expect_deals * MIN_RECORDS_PER_DEAL:
        return False, f"only {n} records for {expect_deals} deals"
    return True, f"ok: {n} records, reward mean {rmean:+.5f}"


# ---------------------------------------------------------------------------
# Record format v3 (docs/v6_prereg.md stage 2): HMR3 header + 1180-byte
# records. The full invariant battery lives in validate_v3_records.py at the
# repo root; this is the SAME battery, importable on workers and the
# orchestrator (both re-validate; workers are untrusted).
# ---------------------------------------------------------------------------
V3_REC = np.dtype([
    ('obs', 'u1', 556), ('ext', 'u1', 326), ('mask', 'u1', 52),
    ('labels', 'u1', 156), ('pi', 'u1', 52), ('action', '<u2'),
    ('seat', '<u2'), ('reward', '<f4'), ('eq_best', '<f4'),
    ('eq_second', '<f4'), ('gap_se', '<f4'), ('second_action', '<u2'),
    ('n_dets', '<u2'), ('match_id', '<u4'), ('flags', '<u2'),
    ('reserved', '<u2'), ('fin', 'u1', 4), ('moonby', 'i1'), ('pad', 'u1')])
assert V3_REC.itemsize == 1180

_PEN = np.zeros(52)
_PEN[39:52] = 1.0
_PEN[36] = 13.0
_V3_REWARDS = np.array([6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0])


def validate_v3_file(path):
    """One per-thread .hmr3 file. Returns (ok, n_matches, detail)."""
    try:
        raw = open(path, 'rb').read()
    except OSError as e:
        return False, 0, f"unreadable: {e}"
    if len(raw) < 32 or raw[:4] != b'HMR3':
        return False, 0, "bad header magic"
    version = int.from_bytes(raw[4:6], 'little')
    rsize = int.from_bytes(raw[6:8], 'little')
    tid = int.from_bytes(raw[12:14], 'little')
    if version != 3 or rsize != 1180:
        return False, 0, f"bad version/size {version}/{rsize}"
    body = raw[32:]
    if len(body) == 0 or len(body) % 1180 != 0:
        return False, 0, f"torn body ({len(body) % 1180} trailing bytes)"
    r = np.frombuffer(body, dtype=V3_REC)
    n = len(r)
    idx = np.arange(n)

    if not (r['mask'][idx, r['action']] == 1).all():
        return False, 0, "action not in mask"
    hand = r['obs'][:, 0:52] > 127
    if not hand[idx, r['action']].all():
        return False, 0, "action not from hand"
    if (r['seat'] >= 4).any():
        return False, 0, "seat out of range"

    on_table = r['obs'][:, 52:104] > 127
    history = r['obs'][:, 104:156] > 127
    pos = r['ext'][:, 0:52]
    planes = (r['ext'][:, 104:312].reshape(n, 4, 52) > 127)
    n_capt = planes.sum(axis=1)
    unseen = ~(on_table | history)
    if (n_capt > 1).any():
        return False, 0, "card captured twice"
    if (n_capt[on_table] != 0).any() or (pos[on_table] == 0).any():
        return False, 0, "on-table capture state wrong"
    if (n_capt[history] != 1).any() or (pos[history] == 0).any():
        return False, 0, "history capture state wrong"
    if (n_capt[unseen] != 0).any() or (pos[unseen] != 0).any():
        return False, 0, "unseen capture state wrong"

    pts = planes @ _PEN
    others = pts.sum(axis=1, keepdims=True) - pts
    alive = r['ext'][:, 316:320] > 127
    if (alive != (others == 0)).any():
        return False, 0, "moon-alive mismatch"

    lab = r['labels'].reshape(n, 3, 52) > 0
    if (lab.sum(axis=2) > 13).any():
        return False, 0, "opponent hand > 13"
    if (lab.any(axis=1) & hand).any() or (lab.any(axis=1) & history).any():
        return False, 0, "belief labels overlap hand/played"

    fin_sum = r['fin'].sum(axis=1)
    if not np.isin(fin_sum, (26, 78)).all():
        return False, 0, "fin sum not 26/78"
    moon = fin_sum == 78
    if (r['moonby'][~moon] != -1).any():
        return False, 0, "moonby set without moon"
    if moon.any():
        mb = r['moonby'][moon].astype(int)
        if (mb < 0).any() or (mb > 3).any() or \
                (r['fin'][moon, mb] != 0).any():
            return False, 0, "moonby inconsistent"

    if not np.isin(r['reward'], _V3_REWARDS).all():
        return False, 0, "bad reward value"

    # shooter schedule + attacker exclusion (file order = match order)
    mids = list(dict.fromkeys(r['match_id'].tolist()))
    for m, mid in enumerate(mids):
        rows = r[r['match_id'] == mid]
        sched = (m + tid) % 8 == 7
        bit2 = (rows['flags'] & 4) > 0
        if bool(bit2.all()) != sched or bool(bit2.any()) != sched:
            return False, 0, f"match {mid}: shooter flag != schedule"
        n_seats = len(set(rows['seat'].tolist()))
        if n_seats != (3 if sched else 4):
            return False, 0, f"match {mid}: {n_seats} recorded seats"
    return True, len(mids), f"ok: {n} records, {len(mids)} matches"


def validate_v3_tar(path, expect_matches=None):
    """A chunk upload = tar of per-thread .hmr3 files. Returns (ok, detail)."""
    import tarfile
    import tempfile
    total_m = total_r = 0
    try:
        with tarfile.open(path) as tf, tempfile.TemporaryDirectory() as td:
            names = [m for m in tf.getmembers() if m.isfile()]
            if not names:
                return False, "empty tar"
            for m in names:
                tf.extract(m, td, filter='data')
                ok, nm, detail = validate_v3_file(os.path.join(td, m.name))
                if not ok:
                    return False, f"{m.name}: {detail}"
                total_m += nm
                total_r += int(detail.split()[1])
    except Exception as e:
        return False, f"tar error: {e}"
    if expect_matches is not None and total_m != expect_matches:
        return False, f"{total_m} matches != expected {expect_matches}"
    return True, f"ok: {total_r} records, {total_m} matches"


if __name__ == '__main__':
    import sys
    if sys.argv[1].endswith('.tar'):
        ok, detail = validate_v3_tar(sys.argv[1],
                                     int(sys.argv[2]) if len(sys.argv) > 2
                                     else None)
    else:
        ok, detail = validate_shard(sys.argv[1],
                                    int(sys.argv[2]) if len(sys.argv) > 2
                                    else None)
    print(("PASS " if ok else "FAIL ") + detail)
    sys.exit(0 if ok else 1)
