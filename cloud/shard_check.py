"""Record-level sanity validation for self-play shards (REQUIREMENTS R1/R2).

A shard is only marked complete once it passes these checks - run by the
worker before upload (fail fast) and again by the orchestrator on receipt
(authoritative; workers are untrusted).
"""
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


if __name__ == '__main__':
    import sys
    ok, detail = validate_shard(sys.argv[1],
                                int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(("PASS " if ok else "FAIL ") + detail)
    sys.exit(0 if ok else 1)
