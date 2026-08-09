"""Round-2 Phase B2 dataset builder (docs/exploiter_league_r2_prereg.md).

Reads the validated A2 corpus (expert_data/defender_v1/r2_*.sdrec) and
produces the anchored-distillation training banks:

  defense stream = moon-alive defender PLAY decisions + defender passes
                   (all passes carry alive=1 by construction: passing
                   happens at deal start, before any point is taken)
                   -> target = the search defender's recorded choice
  anchor stream  = ordinary decisions (threat-dead play decisions)
                   -> target = the BASELINE's own argmax (self-distill),
                   computed here in one preprocessing pass
  drift holdout  = ordinary decisions from HELD-OUT MATCHES (>=20k,
                   match-keyed split so no game leaks across the split;
                   stratified round-robin across the 6 shard files),
                   with the baseline argmax stored for the drift screen
  defense holdout= defense-stream decisions from the same held-out
                   matches (telemetry only: teacher-match reporting)

Pipeline safety checks (all hard-fail):
  - baseline is the PROMOTED MILESTONE (md5 verified 8a89da90), NOT
    hearts_model_final.pth (the working file currently holds a rejected
    candidate);
  - obs are stored as raw float32 in these records - NO /255
    dequantization (that convention belongs to selfplay_gen banks);
  - every recorded action and every computed argmax target is inside
    its legal mask;
  - stream counts must reconcile exactly with the validated corpus
    verdict (equity_data/verdicts/exploiter_r2_corpusA2.json).

Outputs b2_data/{def_train,anchor_train,drift_holdout,def_holdout}.npz
+ manifest.json (counts, seed, per-file held matches, md5s).
"""
import glob
import hashlib
import json
import os

import numpy as np
import torch

from hearts_net import net_from_checkpoint

REC_DTYPE = np.dtype([('obs', '<f4', 556), ('mask', 'u1', 52),
                      ('action', '<i4'), ('flags', 'u1'), ('seat', 'u1'),
                      ('match', '<u2')])
F_PASS, F_ALIVE, F_DEF = 1, 2, 8

CORPUS_DIR = 'expert_data/defender_v1'
OUT_DIR = 'b2_data'
BASELINE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
BASELINE_MD5_8 = '8a89da90'
VERDICT = 'equity_data/verdicts/exploiter_r2_corpusA2.json'
HOLDOUT_TARGET = 20000     # drift-screen ordinary positions (prereg)
SPLIT_SEED = 20260809


def md5_8(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert md5_8(BASELINE) == BASELINE_MD5_8, \
        f'{BASELINE} md5 != {BASELINE_MD5_8} - refusing to build'

    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'r2_*.sdrec')))
    assert len(files) == 6, f'expected 6 shard files, found {len(files)}'

    recs_by_file = []
    for fi, p in enumerate(files):
        size = os.path.getsize(p)
        assert size % REC_DTYPE.itemsize == 0, f'{p}: partial record'
        r = np.fromfile(p, dtype=REC_DTYPE)
        assert ((r['flags'] & F_DEF) != 0).all(), f'{p}: non-defender record'
        recs_by_file.append(r)

    # Match-keyed holdout: round-robin one match at a time across files
    # (stratifies agg/sel evenly) until held ordinary count >= target.
    rng = np.random.default_rng(SPLIT_SEED)
    per_file_matches = []
    for r in recs_by_file:
        m = np.unique(r['match'])
        rng.shuffle(m)
        per_file_matches.append(list(m))
    held = [set() for _ in files]
    held_ord = 0
    ord_by_file = [((r['flags'] & (F_ALIVE | F_PASS)) == 0)
                   for r in recs_by_file]
    ord_count_by_match = []
    for r, om in zip(recs_by_file, ord_by_file):
        cnt = {}
        for mid in np.unique(r['match']):
            cnt[int(mid)] = int((om & (r['match'] == mid)).sum())
        ord_count_by_match.append(cnt)
    fi = 0
    while held_ord < HOLDOUT_TARGET:
        if per_file_matches[fi]:
            mid = int(per_file_matches[fi].pop())
            held[fi].add(mid)
            held_ord += ord_count_by_match[fi][mid]
        fi = (fi + 1) % len(files)
        if all(not lst for lst in per_file_matches):
            break
    assert held_ord >= HOLDOUT_TARGET, f'only {held_ord} held ordinary'

    def split(pred):
        tr, ho = [], []
        for r, hset in zip(recs_by_file, held):
            m = pred(r)
            in_held = np.isin(r['match'], list(hset)) if hset else \
                np.zeros(len(r), dtype=bool)
            tr.append(r[m & ~in_held])
            ho.append(r[m & in_held])
        return np.concatenate(tr), np.concatenate(ho)

    is_def_stream = lambda r: ((r['flags'] & F_ALIVE) != 0) | \
                              ((r['flags'] & F_PASS) != 0)
    is_ordinary = lambda r: (r['flags'] & (F_ALIVE | F_PASS)) == 0

    def_tr, def_ho = split(is_def_stream)
    ord_tr, ord_ho = split(is_ordinary)

    # Reconcile against the validated corpus verdict.
    with open(VERDICT) as f:
        v = json.load(f)
    v_alive = v['volume']['alive_play_decisions']
    v_pass = v['volume']['defender_pass_decisions']
    v_total = sum(m['records'] for m in v['modes'].values())
    got_def = len(def_tr) + len(def_ho)
    got_ord = len(ord_tr) + len(ord_ho)
    assert got_def == v_alive + v_pass, (got_def, v_alive + v_pass)
    assert got_def + got_ord == v_total, (got_def + got_ord, v_total)

    # In-mask assertions for recorded actions.
    for name, arr in (('def_train', def_tr), ('def_holdout', def_ho),
                      ('anchor_train', ord_tr), ('drift_holdout', ord_ho)):
        act = arr['action']
        assert (act >= 0).all() and (act < 52).all(), name
        assert (arr['mask'][np.arange(len(arr)), act] == 1).all(), \
            f'{name}: recorded action outside mask'

    # Baseline argmax targets (raw float obs - NO /255) for the anchor
    # stream and the drift holdout.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net_from_checkpoint(BASELINE).to(device).eval()

    def base_argmax(arr, batch=4096):
        out = np.empty(len(arr), dtype=np.int32)
        with torch.no_grad():
            for s in range(0, len(arr), batch):
                b = arr[s:s + batch]
                obs = torch.from_numpy(np.ascontiguousarray(b['obs'])).to(device)
                mask = torch.from_numpy(np.ascontiguousarray(b['mask'])).to(device).bool()
                logits, _, _ = net.forward_all(obs, mask)
                out[s:s + batch] = logits.argmax(1).cpu().numpy()
        return out

    anchor_tgt = base_argmax(ord_tr)
    drift_tgt = base_argmax(ord_ho)
    for name, arr, tgt in (('anchor', ord_tr, anchor_tgt),
                           ('drift', ord_ho, drift_tgt)):
        assert (arr['mask'][np.arange(len(arr)), tgt] == 1).all(), \
            f'{name}: baseline argmax outside mask (wiring bug)'

    # Diagnostics: baseline-vs-search-defender agreement per stream.
    # Sane values are moderate-to-high on ordinary states; near-random
    # (~10-20%) would indicate an obs-scaling/wiring bug.
    agree_ord = float((anchor_tgt == ord_tr['action']).mean())
    da = base_argmax(def_tr)
    play = (def_tr['flags'] & F_PASS) == 0
    agree_def_play = float((da[play] == def_tr['action'][play]).mean())
    agree_def_pass = float((da[~play] == def_tr['action'][~play]).mean())

    def save(name, arr, extra=None):
        d = {'obs': np.ascontiguousarray(arr['obs']),
             'mask': np.ascontiguousarray(arr['mask']),
             'action': arr['action'].astype(np.int32),
             'flags': arr['flags']}
        if extra is not None:
            d['base_argmax'] = extra
        np.savez(os.path.join(OUT_DIR, name + '.npz'), **d)

    save('def_train', def_tr)
    save('def_holdout', def_ho)
    save('anchor_train', ord_tr, anchor_tgt)
    save('drift_holdout', ord_ho, drift_tgt)

    manifest = {
        'split_seed': SPLIT_SEED,
        'baseline': BASELINE, 'baseline_md5_8': BASELINE_MD5_8,
        'files': {os.path.basename(p): md5_8(p) for p in files},
        'held_matches_per_file': {os.path.basename(p): sorted(h)
                                  for p, h in zip(files, held)},
        'counts': {'def_train': len(def_tr), 'def_holdout': len(def_ho),
                   'anchor_train': len(ord_tr),
                   'drift_holdout': len(ord_ho)},
        'diagnostics': {'base_vs_search_ordinary': round(agree_ord, 4),
                        'base_vs_search_alive_play': round(agree_def_play, 4),
                        'base_vs_search_pass': round(agree_def_pass, 4)},
    }
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    print(f"def_train {len(def_tr):,}  def_holdout {len(def_ho):,}  "
          f"anchor_train {len(ord_tr):,}  drift_holdout {len(ord_ho):,}")
    print(f"baseline-vs-search agreement: ordinary {agree_ord:.3f}  "
          f"alive-play {agree_def_play:.3f}  pass {agree_def_pass:.3f}")
    print(f"wrote {OUT_DIR}/ + manifest.json")


if __name__ == '__main__':
    main()
