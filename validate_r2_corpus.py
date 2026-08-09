"""Round-2 Phase A2 corpus validator (docs/exploiter_league_r2_prereg.md).

Executes the pre-registered instrument HALT checks on the defended-game
corpus plus the volume stop condition, from the generator's own outputs
(expert_data/defender_v1/r2_{agg,sel}_*.{csv,sdrec}):

  1. RECORDER SELF-CONSISTENCY: every .sdrec record's action is inside
     its legal mask; every record is a defender record (clone attackers
     are never recorded), seated off the attacker chair (match % 4).
  2. ATTACKER COMPLETION BAND (2026-08-08 amendment): per mode, clone
     moon completion/deal in [0.05, certified-vs-raw] (agg 0.291,
     sel 0.224).
  3. DEFENDER-SIDE MOONS: < 0.05/deal per mode.
  4. VOLUME (stop condition): >= 30,000 moon-alive defender PLAY
     decisions banked across modes (defender passes reported alongside:
     total, and from threat games = matches with >= 1 alive decision).

Safe to run while generation is live (per-deal flush; a trailing
partial record is tolerated and reported). HALT verdicts are only
meaningful on the COMPLETE corpus - live runs are informational.

Usage: python validate_r2_corpus.py [--dir expert_data/defender_v1]
                                    [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REC_DTYPE = np.dtype([('obs', '<f4', 556), ('mask', 'u1', 52),
                      ('action', '<i4'), ('flags', 'u1'), ('seat', 'u1'),
                      ('match', '<u2')])
assert REC_DTYPE.itemsize == 2284

F_PASS, F_ALIVE, F_SHOOT, F_DEF = 1, 2, 4, 8

# Completion bands per the pre-generation amendment: [0.05/deal,
# Phase B certified rate vs raw defenders].
BANDS = {'agg': (0.05, 0.291), 'sel': (0.05, 0.224)}
DEF_MOON_MAX = 0.05
VOLUME_TARGET = 30000

# deal-CSV column indices (header in search_eval.cpp DHDR)
C_MATCH, C_SEAT, C_MOON, C_DMOON = 0, 2, 13, 14


def read_records(path):
    size = os.path.getsize(path)
    n, tail = divmod(size, REC_DTYPE.itemsize)
    recs = np.fromfile(path, dtype=REC_DTYPE, count=n)
    return recs, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='expert_data/defender_v1')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    out = {'modes': {}, 'volume': {}, 'halts': []}
    tot_alive_play = tot_def_pass = tot_threat_pass = 0

    for mode in ('agg', 'sel'):
        csvs = sorted(glob.glob(os.path.join(args.dir, f'r2_{mode}_*.csv')))
        csvs = [c for c in csvs if not c.endswith('.tricks.csv')]
        if not csvs:
            print(f'[{mode}] no shards found - skipped')
            continue
        deals = moons = dmoons = 0
        n_matches = 0
        for c in csvs:
            rows = np.genfromtxt(c, delimiter=',', skip_header=1,
                                 usecols=(C_MATCH, C_MOON, C_DMOON), ndmin=2)
            if rows.size == 0:
                continue
            deals += rows.shape[0]
            moons += int(rows[:, 1].sum())
            dmoons += int(rows[:, 2].sum())
            n_matches += len(np.unique(rows[:, 0]))
        comp = moons / deals if deals else 0.0
        drate = dmoons / deals if deals else 0.0

        alive_play = def_pass = threat_pass = bad_act = nondef = 0
        recs_total = 0
        partial_bytes = 0
        for c in csvs:
            rp = c[:-4] + '.sdrec'
            if not os.path.exists(rp):
                out['halts'].append(f'{mode}: missing {rp}')
                continue
            recs, tail = read_records(rp)
            partial_bytes += tail
            recs_total += len(recs)
            # 1) self-consistency
            act = recs['action']
            ok = (act >= 0) & (act < 52)
            picked = np.zeros(len(recs), dtype=bool)
            picked[ok] = recs['mask'][np.nonzero(ok)[0], act[ok]] == 1
            bad_act += int((~picked).sum())
            is_def = (recs['flags'] & F_DEF) != 0
            nondef += int((~is_def).sum())
            wrong_seat = int((recs['seat'][is_def]
                              == (recs['match'][is_def] % 4)).sum())
            if wrong_seat:
                out['halts'].append(
                    f'{mode}: {wrong_seat} defender records on the '
                    f'attacker seat in {os.path.basename(rp)}')
            # 4) volume slices
            is_pass = (recs['flags'] & F_PASS) != 0
            is_alive = (recs['flags'] & F_ALIVE) != 0
            alive_play += int((is_def & is_alive & ~is_pass).sum())
            def_pass += int((is_def & is_pass).sum())
            # threat games: matches with >= 1 alive defender decision
            alive_matches = np.unique(recs['match'][is_def & is_alive])
            threat_pass += int((is_def & is_pass
                                & np.isin(recs['match'], alive_matches)).sum())

        m = {'shards': len(csvs), 'matches': n_matches, 'deals': deals,
             'attacker_moons': moons, 'completion_per_deal': round(comp, 4),
             'band': BANDS[mode], 'defender_moons': dmoons,
             'defender_moon_per_deal': round(drate, 4),
             'records': recs_total, 'bad_action_records': bad_act,
             'non_defender_records': nondef,
             'alive_play_decisions': alive_play,
             'defender_pass_decisions': def_pass,
             'threat_game_pass_decisions': threat_pass,
             'trailing_partial_bytes': partial_bytes}
        out['modes'][mode] = m
        tot_alive_play += alive_play
        tot_def_pass += def_pass
        tot_threat_pass += threat_pass

        lo, hi = BANDS[mode]
        if bad_act:
            out['halts'].append(f'{mode}: {bad_act} records with action '
                                f'outside the legal mask')
        if nondef:
            out['halts'].append(f'{mode}: {nondef} non-defender records '
                                f'(clone attackers must not record)')
        if not (lo <= comp <= hi):
            out['halts'].append(f'{mode}: completion {comp:.4f}/deal outside '
                                f'band [{lo}, {hi}]')
        if drate >= DEF_MOON_MAX:
            out['halts'].append(f'{mode}: defender moons {drate:.4f}/deal '
                                f'>= {DEF_MOON_MAX}')
        print(f'[{mode}] shards {len(csvs)}  matches {n_matches}  '
              f'deals {deals}  completion {comp:.4f} (band [{lo}, {hi}])  '
              f'defender-moons {drate:.4f}  records {recs_total}  '
              f'alive-play {alive_play}  passes {def_pass} '
              f'(threat-games {threat_pass})')

    out['volume'] = {'alive_play_decisions': tot_alive_play,
                     'target': VOLUME_TARGET,
                     'met': tot_alive_play >= VOLUME_TARGET,
                     'defender_pass_decisions': tot_def_pass,
                     'threat_game_pass_decisions': tot_threat_pass}
    print(f'VOLUME: {tot_alive_play}/{VOLUME_TARGET} moon-alive defender '
          f'play decisions ({"MET" if tot_alive_play >= VOLUME_TARGET else "not met"})  '
          f'+ {tot_def_pass} defender passes ({tot_threat_pass} in threat games)')
    if out['halts']:
        print('HALT CHECKS FAILING:')
        for h in out['halts']:
            print(f'  - {h}')
    else:
        print('HALT CHECKS: all clean')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'wrote {args.json}')
    sys.exit(1 if out['halts'] else 0)


if __name__ == '__main__':
    main()
