"""Exploiter league Phase A: instrument validation + base rates.

Reads the shooter driver's RAW event shards (per-deal rows + per-trick
rows, SearchEval --shooter) and computes the pre-registered validity
checks and gate base rates (docs/exploiter_league_prereg.md). Metric
definitions live HERE, not in the C++ driver, so they can be refined
without re-running search-speed matches.

Shard naming convention (see ops/run_phaseA_shooter.sh):
    <prefix>_<combo>_<shard>.csv        combo in {agg_base, sel_base, sel_v4}
    <prefix>_<combo>_<shard>.tricks.csv

Registered checks:
  1. AGG moon-success rate vs baseline >= 3x background. Background is
     operationalized as the DEFENDER seats' accidental-moon rate in the
     same matches (same nets, same conditions, not attacking) - reported
     explicitly so the operationalization is visible.
  2. SEL success vs v4-m10 field < SEL success vs baseline field
     (concession-rate ordering; mechanism-agnostic per the amendment).
  3. SEL attempt rate in [2%, 60%] of deals.
Plus: AGG-vs-SEL mechanism split, defense-cost accounting from trick
events, and the defense-gate power computation (n for 80% power to see
a 25% relative concession reduction, one-sided alpha=.05, unpaired-
conservative since Phase A has one arm).
"""
import argparse
import glob
import json
import math
import os
import sys
import csv as csvmod
from collections import defaultdict


def read_csv(path):
    with open(path, newline='') as f:
        return list(csvmod.DictReader(f))


def load_combo(prefix, combo):
    deals, tricks = [], []
    dpaths = sorted(glob.glob(f"{prefix}_{combo}_*.csv"))
    dpaths = [p for p in dpaths if not p.endswith('.tricks.csv')]
    if not dpaths:
        return None
    for p in dpaths:
        shard = p
        for r in read_csv(p):
            r['_shard'] = shard
            deals.append(r)
        tp = p[:-4] + '.tricks.csv'
        if not os.path.exists(tp):
            print(f"HALT: missing trick shard {tp}", file=sys.stderr)
            sys.exit(2)
        for r in read_csv(tp):
            r['_shard'] = shard
            tricks.append(r)
    return deals, tricks


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - w) / den, (c + w) / den)


def analyze_combo(name, deals, tricks):
    n_deals = len(deals)
    n_matches = len({(d['_shard'], d['match']) for d in deals})
    succ = sum(int(d['moon_success']) for d in deals)
    dmoon = sum(int(d['defender_moon']) for d in deals)
    attempts = sum(1 for d in deals
                   if d['pass_committed'] == '1' or int(d['shoot_dec']) > 0)
    succ_in_att = sum(int(d['moon_success']) for d in deals
                      if d['pass_committed'] == '1' or int(d['shoot_dec']) > 0)

    # Trick-event pass: threat-alive accounting per deal. Threat alive at a
    # trick = before it, the shooter had taken every point so far AND at
    # least one point (a live, visible threat). Defense intervention = a
    # defender wins a point-trick while the threat is alive (the block).
    by_deal = defaultdict(list)
    for t in tricks:
        by_deal[(t['_shard'], t['match'], t['deal'])].append(t)
    interventions = 0
    block_points = 0            # points defenders paid making blocks
    threat_deals = 0            # deals where a live threat existed
    for key, ts in by_deal.items():
        ts.sort(key=lambda r: int(r['trick']))
        shooter = int(ts[0]['shooter_seat'])
        cum_sh, cum_other, threat_seen = 0, 0, False
        for t in ts:
            w, pts = int(t['winner']), int(t['points'])
            alive = (cum_other == 0 and cum_sh >= 1)
            if alive:
                threat_seen = True
                if w != shooter and pts > 0:
                    interventions += 1
                    block_points += pts
            if w == shooter:
                cum_sh += pts
            else:
                cum_other += pts
        threat_deals += threat_seen

    # Per-match concession counts -> defense-gate power inputs
    per_match = defaultdict(int)
    for d in deals:
        per_match[(d['_shard'], d['match'])] += int(d['moon_success'])
    counts = list(per_match.values())
    mean_c = sum(counts) / len(counts) if counts else 0.0
    var_c = (sum((c - mean_c) ** 2 for c in counts) / (len(counts) - 1)
             if len(counts) > 1 else 0.0)

    out = {
        'combo': name,
        'matches': n_matches,
        'deals': n_deals,
        'moons': succ,
        'moon_rate_per_deal': succ / max(1, n_deals),
        'moon_rate_ci': wilson(succ / max(1, n_deals), n_deals),
        'attempt_deals': attempts,
        'attempt_rate': attempts / max(1, n_deals),
        'success_given_attempt': succ_in_att / max(1, attempts),
        'defender_moons': dmoon,
        'defender_moon_rate': dmoon / max(1, n_deals),
        'threat_deals': threat_deals,
        'interventions': interventions,
        'block_points': block_points,
        'moons_per_match_mean': mean_c,
        'moons_per_match_var': var_c,
    }
    return out


def power_n(mean_c, var_c, rel_reduction=0.25, alpha=0.05, power=0.80):
    """Matches per arm for a one-sided two-sample test of a rel_reduction
    drop in per-match concessions (unpaired-conservative: CRN pairing in
    the real gate only helps)."""
    if mean_c <= 0 or var_c <= 0:
        return None
    delta = mean_c * rel_reduction
    za, zb = 1.645, 0.842
    return int(math.ceil(2 * var_c * (za + zb) ** 2 / (delta * delta)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='equity_data/exploiter_r1/phaseA')
    ap.add_argument('--write', action='store_true',
                    help='write docs/exploiter_league_phaseA.md + verdict JSON')
    args = ap.parse_args()

    results = {}
    for combo in ('agg_base', 'sel_base', 'sel_v4'):
        loaded = load_combo(args.prefix, combo)
        if loaded is None:
            print(f"(combo {combo}: no shards yet)")
            continue
        results[combo] = analyze_combo(combo, *loaded)

    checks = {}
    if 'agg_base' in results:
        r = results['agg_base']
        bg = r['defender_moon_rate']
        checks['check1_agg_attacks'] = {
            'agg_success_rate': r['moon_rate_per_deal'],
            'background_defender_rate': bg,
            'ratio': (r['moon_rate_per_deal'] / bg) if bg > 0 else float('inf'),
            'bar': '>= 3x background',
            'pass': r['moon_rate_per_deal'] >= 3 * bg if bg > 0
                    else r['moons'] > 0,
        }
    if 'sel_base' in results and 'sel_v4' in results:
        b, v = results['sel_base'], results['sel_v4']
        checks['check2_ordering'] = {
            'sel_vs_baseline': b['moon_rate_per_deal'],
            'ci_baseline': b['moon_rate_ci'],
            'sel_vs_v4m10': v['moon_rate_per_deal'],
            'ci_v4': v['moon_rate_ci'],
            'bar': 'v4 rate < baseline rate',
            'pass': v['moon_rate_per_deal'] < b['moon_rate_per_deal'],
        }
    if 'sel_base' in results:
        r = results['sel_base']
        # Band widened [2%,60%] -> [2%,90%] by user-approved amendment
        # 2026-08-06 (prereg): the ceiling presumed a punishing defender;
        # at 51.5% success/attempt, high attempt rates are rational.
        checks['check3_sel_attempt_band'] = {
            'attempt_rate': r['attempt_rate'],
            'bar': '[0.02, 0.90] (amended 2026-08-06)',
            'pass': 0.02 <= r['attempt_rate'] <= 0.90,
        }
        checks['gate_power'] = {
            'moons_per_match_mean': r['moons_per_match_mean'],
            'moons_per_match_var': r['moons_per_match_var'],
            'n_per_arm_80pct_25rel': power_n(r['moons_per_match_mean'],
                                             r['moons_per_match_var']),
            'note': 'unpaired-conservative; CRN pairing in the gate only helps',
        }
    if 'agg_base' in results and 'sel_base' in results:
        checks['mechanism_split'] = {
            'agg_success': results['agg_base']['moon_rate_per_deal'],
            'sel_success': results['sel_base']['moon_rate_per_deal'],
            'sel_success_given_attempt':
                results['sel_base']['success_given_attempt'],
            'note': ('AGG-vs-SEL gap decomposes cannot-block (AGG succeeds) '
                     'vs cannot-detect-selected-threats (SEL-specific)'),
        }

    all_checked = [v for k, v in checks.items() if 'pass' in v]
    verdict = ('PASS' if all_checked and all(v['pass'] for v in all_checked)
               else 'HALT' if all_checked else 'INCOMPLETE')

    print(json.dumps({'results': results, 'checks': checks,
                      'verdict': verdict}, indent=2))

    if args.write:
        os.makedirs('equity_data/verdicts', exist_ok=True)
        with open('equity_data/verdicts/exploiter_r1_phaseA.json', 'w') as f:
            json.dump({'results': results, 'checks': checks,
                       'verdict': verdict}, f, indent=2)
        lines = ["# Exploiter league Phase A - instrument validation + base rates\n",
                 "Written by analyze_phasea.py from raw event shards, before "
                 "narrative interpretation.\n",
                 f"\nVERDICT: **{verdict}** (halt-is-default: any failed check "
                 "stops Phase B)\n",
                 "\n| combo | matches | deals | moons | rate/deal | attempts | "
                 "P(success|attempt) | defender moons |\n|---|---|---|---|---|---|---|---|\n"]
        for c, r in results.items():
            lines.append(
                f"| {c} | {r['matches']} | {r['deals']} | {r['moons']} | "
                f"{r['moon_rate_per_deal']:.4f} | {r['attempt_deals']} | "
                f"{r['success_given_attempt']:.3f} | {r['defender_moons']} |\n")
        lines.append("\n## Checks\n```json\n"
                     + json.dumps(checks, indent=2) + "\n```\n")
        with open('docs/exploiter_league_phaseA.md', 'w') as f:
            f.writelines(lines)
        print("wrote docs/exploiter_league_phaseA.md + verdict JSON")


if __name__ == '__main__':
    main()
