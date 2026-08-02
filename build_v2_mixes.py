"""Build expert-iteration v2 mix banks (docs/expert_iter_v2_prereg.md,
mix-selection stage).

For each candidate mix, selects the prescribed composition of CONFIDENT
play-phase records (gap > 2 x gap_se, the frozen prereg rule) from the
expert_data/v2_*.bin reserves, plus a non-confident play-phase anchor
sample at the same family proportions (--anchor-ratio), and writes:

  <out-dir>/<mix>.bin           HMR2 bank, records grouped by source
                                match and match groups shuffled -- so
                                distill.py's per-file tail holdout stays
                                a by-match split (rule #4)
  <out-dir>/<mix>.index.csv     (source file, record index, family,
                                confident) per selected record
  <out-dir>/<mix>.manifest.json composition counts, selection seed,
                                sha256 of the index list and of the bank

Selection is deterministic from --seed-base (per-mix seed = base + mix
index). Mixes share reserves (max-not-sum governs feasibility, prereg).
Run --dry-run for a feasibility/composition report without writing.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

from distill import MATCH_RECORD_V2, V2_MAGIC

FAMILIES = ['knife', 'mid', 'leader', 'asym', 'trail', 'early']

# (natural share, seeded share); seeded share is spread evenly over the
# six families by largest remainder. Names follow the prereg's (a)..(e).
MIXES = {
    'a_nat60':     (0.60, 0.40),
    'b_even50':    (0.50, 0.50),
    'c_seed65':    (0.35, 0.65),
    'd_natonly':   (1.00, 0.00),
    'e_seedspread': (0.00, 1.00),
}

CONF_SIGMA = 2.0  # frozen (prereg); distill.py --min-confidence must match


def family_of(path):
    b = os.path.basename(path)
    for fam in FAMILIES:
        if f'_{fam}_' in b:
            return fam
    return 'nat' if '_nat_' in b else None


def load_bank(pattern):
    """Load every v2 file; returns list of dicts with per-file arrays."""
    files = []
    for f in sorted(glob.glob(pattern)):
        fam = family_of(f)
        if fam is None:
            continue
        with open(f, 'rb') as fh:
            if fh.read(4) != V2_MAGIC:
                continue
        n = (os.path.getsize(f) - 32) // MATCH_RECORD_V2.itemsize
        if n <= 0:
            continue
        a = np.fromfile(f, dtype=MATCH_RECORD_V2, offset=32, count=n)
        play = (a['flags'] & 1) == 0
        valid = a['second_action'] != 0xFFFF
        gap = a['eq_best'].astype(np.float32) - a['eq_second']
        conf = play & valid & (gap > CONF_SIGMA * a['gap_se'])
        files.append({'path': f, 'family': fam, 'records': a,
                      'conf_idx': np.flatnonzero(conf),
                      'nonconf_idx': np.flatnonzero(play & ~conf)})
    return files


def split_even(total, k):
    """Largest-remainder split of `total` into k near-equal integers."""
    base = total // k
    parts = [base] * k
    for i in range(total - base * k):
        parts[i] += 1
    return parts


def mix_requirements(mix, target):
    nat_share, seed_share = MIXES[mix]
    n_nat = round(target * nat_share)
    n_seed = target - n_nat
    req = {'nat': n_nat}
    for fam, cnt in zip(FAMILIES, split_even(n_seed, len(FAMILIES))):
        req[fam] = cnt
    return req


def select(files, req, rng, pool_key):
    """Sample per-family counts from pool_key ('conf_idx'/'nonconf_idx').
    Returns {family: [(file_i, record_idx array)]} or raises on shortfall."""
    out = {}
    for fam, need in req.items():
        if need == 0:
            continue
        pools = [(i, f[pool_key]) for i, f in enumerate(files)
                 if f['family'] == fam]
        avail = sum(len(p) for _, p in pools)
        if avail < need:
            raise SystemExit(f"SHORTFALL {fam}: need {need}, have {avail} "
                             f"({pool_key})")
        # Global uniform draw across this family's files.
        flat = np.concatenate([np.full(len(p), i) for i, p in pools]) \
            if pools else np.empty(0, dtype=int)
        offs = np.concatenate([p for _, p in pools])
        pick = rng.choice(len(flat), size=need, replace=False)
        sel = {}
        for j in pick:
            sel.setdefault(int(flat[j]), []).append(int(offs[j]))
        out[fam] = [(i, np.array(sorted(v), dtype=np.int64))
                    for i, v in sel.items()]
    return out


def write_mix(files, mix, conf_sel, anch_sel, out_dir, seed):
    """Write bank (match-grouped, groups shuffled), index csv, manifest."""
    # Collect (file_i, record_idx, confident) then group by (file, match_id).
    entries = []
    for sel, is_conf in ((conf_sel, 1), (anch_sel, 0)):
        for fam, parts in sel.items():
            for fi, idxs in parts:
                for ri in idxs:
                    entries.append((fi, int(ri), is_conf))
    groups = {}
    for fi, ri, is_conf in entries:
        mid = int(files[fi]['records'][ri]['match_id'])
        groups.setdefault((fi, mid), []).append((ri, is_conf))
    keys = list(groups.keys())
    rng = np.random.default_rng(seed + 7)  # ordering rng, distinct stream
    rng.shuffle(keys)

    bank_path = os.path.join(out_dir, f'{mix}.bin')
    index_path = os.path.join(out_dir, f'{mix}.index.csv')
    header = V2_MAGIC + np.array([2, MATCH_RECORD_V2.itemsize],
                                 dtype='<u2').tobytes() \
        + np.array([seed % 2**32], dtype='<u4').tobytes() \
        + np.array([0], dtype='<u2').tobytes()
    header += b'\x00' * (32 - len(header))
    n_written = 0
    with open(bank_path, 'wb') as bf, open(index_path, 'w', newline='') as xf:
        bf.write(header)
        xf.write('source_file,record_index,family,confident\n')
        for key in keys:
            fi, _ = key
            f = files[fi]
            for ri, is_conf in sorted(groups[key]):
                bf.write(f['records'][ri].tobytes())
                xf.write(f"{f['path']},{ri},{f['family']},{is_conf}\n")
                n_written += 1
    return bank_path, index_path, n_written


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default='expert_data/v2_*.bin')
    ap.add_argument('--target', type=int, default=50000,
                    help='confident records per mix (prereg: 50,000)')
    ap.add_argument('--anchor-ratio', type=float, default=1.0,
                    help='non-confident anchor records as a fraction of '
                         '--target, same family proportions (recipe-freeze '
                         'knob; recorded in the manifest)')
    ap.add_argument('--seed-base', type=int, default=20260801)
    ap.add_argument('--out-dir', default='expert_data/mixes')
    ap.add_argument('--mix', nargs='+', default=list(MIXES),
                    choices=list(MIXES))
    ap.add_argument('--dry-run', action='store_true',
                    help='feasibility/composition report only, write nothing')
    args = ap.parse_args()

    files = load_bank(args.data_glob)
    if not files:
        raise SystemExit(f'no v2 files match {args.data_glob}')
    conf_avail = {k: 0 for k in ['nat'] + FAMILIES}
    nonconf_avail = {k: 0 for k in ['nat'] + FAMILIES}
    for f in files:
        conf_avail[f['family']] += len(f['conf_idx'])
        nonconf_avail[f['family']] += len(f['nonconf_idx'])
    print(f"Bank: {len(files)} files")
    print("  confident:     " + '  '.join(f"{k}={v:,}" for k, v in conf_avail.items()))
    print("  non-conf play: " + '  '.join(f"{k}={v:,}" for k, v in nonconf_avail.items()))

    feasible = True
    for mix in args.mix:
        # Seed keyed to the CANONICAL mix index, not the invocation's
        # subset order -- selection must reproduce regardless of which
        # --mix subset a run asks for.
        mi = list(MIXES).index(mix)
        req = mix_requirements(mix, args.target)
        anchor_total = round(args.target * args.anchor_ratio)
        areq = {}
        if anchor_total:
            nat_a = round(anchor_total * MIXES[mix][0])
            areq = {'nat': nat_a}
            for fam, cnt in zip(FAMILIES,
                                split_even(anchor_total - nat_a, len(FAMILIES))):
                areq[fam] = cnt
            if MIXES[mix][1] == 0:
                areq = {'nat': anchor_total}
        seed = args.seed_base + mi
        print(f"\n== {mix} (selection seed {seed}) ==")
        print("  confident req: " + '  '.join(f"{k}={v:,}" for k, v in req.items() if v))
        short = [f"{k} (need {v:,}, have {conf_avail[k]:,})"
                 for k, v in req.items() if v > conf_avail[k]]
        short += [f"{k} anchor (need {v:,}, have {nonconf_avail[k]:,})"
                  for k, v in areq.items() if v > nonconf_avail[k]]
        if short:
            print("  INFEASIBLE: " + '; '.join(short))
            feasible = False
            continue
        if args.dry_run:
            print("  feasible (dry-run: not written)")
            continue

        os.makedirs(args.out_dir, exist_ok=True)
        rng = np.random.default_rng(seed)
        conf_sel = select(files, req, rng, 'conf_idx')
        anch_sel = select(files, areq, rng, 'nonconf_idx') if areq else {}
        bank, index, n = write_mix(files, mix, conf_sel, anch_sel,
                                   args.out_dir, seed)
        manifest = {
            'mix': mix, 'target_confident': args.target,
            'composition_confident': req,
            'composition_anchor': areq, 'anchor_ratio': args.anchor_ratio,
            'selection_seed': seed, 'confidence_sigma': CONF_SIGMA,
            'records_written': n,
            'ordering': 'grouped by (source file, match_id); group order '
                        f'shuffled with seed {seed + 7} (per-file tail '
                        'holdout in distill.py = by-match split)',
            'bank': bank, 'bank_sha256': sha256(bank),
            'index': index, 'index_sha256': sha256(index),
            'sources': [{'path': f['path'], 'family': f['family'],
                         'n_records': int(len(f['records']))}
                        for f in files],
        }
        mpath = os.path.join(args.out_dir, f'{mix}.manifest.json')
        with open(mpath, 'w') as mf:
            json.dump(manifest, mf, indent=1)
        print(f"  wrote {bank} ({n:,} records), manifest {mpath}")

    if not feasible:
        print("\nNOT ALL REQUESTED MIXES FEASIBLE" +
              (" (dry-run)" if args.dry_run else ""))
        sys.exit(0 if args.dry_run else 1)


if __name__ == '__main__':
    main()
