"""v6 data-scaling probe: registered inference (docs/v6_data_scaling_prereg.md §4).

Input: the per-match CSV from v6_probe_eval.py (rows: net, file,
match_id, n, ce_sum, correct, n_play, ent_sum) for the six nets
{S1,S2,S3} x {s20260812, s20260813}.

Per match m and size S: metric = mean over the two seeds of the per-
match value (CE = ce_sum/n; teacher-match = correct/n). Paired by-match
differences (123 clusters) for the two registered steps:
  control  S1 -> S2   decision  S2 -> S3
Improvement = CE decreases / match increases. One-sided paired t at
alpha 0.05, exact Wilcoxon signed-rank as robustness, Holm over the two
metrics within each step, 95% two-sided t CIs. Verdict per §4:
  control fails (neither metric q<.05)      -> UNINFORMATIVE
  decision: both q<.05 -> DATA-BOUND; neither -> SATURATED; one -> MIXED

Diagnostics only: per-net aggregate CE / match / play entropy.

--selftest: synthetic CSV with a planted improvement at the control step
and a planted null at the decision step must return SATURATED, and a
planted improvement at both steps must return DATA-BOUND.
"""
import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

SIZES = ('S1', 'S2', 'S3')
SEEDS = ('s20260812', 's20260813')
STEPS = [('control', 'S1', 'S2'), ('decision', 'S2', 'S3')]
ALPHA = 0.05


def load(csv_path):
    per = defaultdict(dict)   # (size, seed) -> {(file, mid): (ce, match, n, ent, nplay)}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            size, seed = r['net'].split('_')
            n = int(r['n'])
            per[(size, seed)][(r['file'], int(r['match_id']))] = (
                float(r['ce_sum']) / n, int(r['correct']) / n, n,
                float(r['ent_sum']), int(r['n_play']))
    return per


def per_match_means(per):
    """size -> {match: (ce, tm)} averaged over the two seeds."""
    out = {}
    for s in SIZES:
        a, b = per[(s, SEEDS[0])], per[(s, SEEDS[1])]
        assert set(a) == set(b), f'{s}: seeds cover different matches'
        out[s] = {m: ((a[m][0] + b[m][0]) / 2, (a[m][1] + b[m][1]) / 2)
                  for m in a}
    keys = sorted(out['S1'])
    for s in SIZES:
        assert sorted(out[s]) == keys, 'sizes cover different matches'
    return out, keys


def paired(d, better_if_negative):
    d = np.asarray(d, float)
    n = len(d)
    mean = d.mean(); se = d.std(ddof=1) / math.sqrt(n)
    t = mean / se
    p_t = stats.t.cdf(t, n - 1) if better_if_negative else stats.t.sf(t, n - 1)
    alt = 'less' if better_if_negative else 'greater'
    nz = d[d != 0]
    p_w = stats.wilcoxon(nz, alternative=alt, method='exact').pvalue \
        if len(nz) else 1.0
    h = stats.t.ppf(0.975, n - 1) * se
    return {'n': n, 'mean': mean, 'se': se, 't': t, 'p_t': p_t,
            'p_wilcoxon': p_w, 'ci95': [mean - h, mean + h]}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); q = [0] * m; run = 0
    for rank, i in enumerate(order):
        run = max(run, pvals[i] * (m - rank)); q[i] = min(1.0, run)
    return q


def analyze(per):
    means, keys = per_match_means(per)
    out = {'n_matches': len(keys), 'steps': {}}
    for name, lo, hi in STEPS:
        d_ce = [means[hi][m][0] - means[lo][m][0] for m in keys]
        d_tm = [means[hi][m][1] - means[lo][m][1] for m in keys]
        r_ce = paired(d_ce, better_if_negative=True)
        r_tm = paired(d_tm, better_if_negative=False)
        q_ce, q_tm = holm([r_ce['p_t'], r_tm['p_t']])
        r_ce['q_holm'] = q_ce; r_tm['q_holm'] = q_tm
        out['steps'][name] = {'from': lo, 'to': hi, 'ce': r_ce,
                              'teacher_match': r_tm,
                              'ce_improves': q_ce < ALPHA,
                              'match_improves': q_tm < ALPHA}
    c = out['steps']['control']; dd = out['steps']['decision']
    if not (c['ce_improves'] or c['match_improves']):
        verdict = 'UNINFORMATIVE'
    else:
        k = int(dd['ce_improves']) + int(dd['match_improves'])
        verdict = {2: 'DATA-BOUND', 0: 'SATURATED', 1: 'MIXED'}[k]
    out['verdict'] = verdict
    # diagnostics
    diag = {}
    for (s, seed), rec in sorted(per.items()):
        n = sum(v[2] for v in rec.values())
        diag[f'{s}_{seed}'] = {
            'ce': sum(v[0] * v[2] for v in rec.values()) / n,
            'teacher_match': sum(v[1] * v[2] for v in rec.values()) / n,
            'entropy_play': sum(v[3] for v in rec.values())
                            / max(1, sum(v[4] for v in rec.values()))}
    out['diagnostics'] = diag
    return out


def synth(effect_control, effect_decision, seed=0):
    rng = np.random.default_rng(seed)
    per = {}
    base = rng.normal(0.85, 0.08, 123); tm0 = rng.normal(0.64, 0.05, 123)
    shift = {'S1': 0.0, 'S2': effect_control,
             'S3': effect_control + effect_decision}
    for s in SIZES:
        for sd in SEEDS:
            rec = {}
            for i in range(123):
                ce = base[i] - shift[s] + rng.normal(0, 0.01)
                tm = tm0[i] + shift[s] + rng.normal(0, 0.008)
                n = 650
                rec[(f'f{i}', 0)] = (ce, tm, n, 0.83 * n, n)
            per[(s, sd)] = rec
    return per


def selftest():
    a = analyze(synth(0.02, 0.0));  assert a['verdict'] == 'SATURATED', a['verdict']
    b = analyze(synth(0.02, 0.02)); assert b['verdict'] == 'DATA-BOUND', b['verdict']
    c = analyze(synth(0.0, 0.02));  assert c['verdict'] == 'UNINFORMATIVE', c['verdict']
    print('selftest PASS: SATURATED / DATA-BOUND / UNINFORMATIVE all recovered')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    res = analyze(load(a.csv))
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        open(a.out, 'w').write(txt)
    for name, st in res['steps'].items():
        for k in ('ce', 'teacher_match'):
            r = st[k]
            print(f"{name:8s} {st['from']}->{st['to']} {k:13s} "
                  f"d={r['mean']:+.5f} SE {r['se']:.5f} CI [{r['ci95'][0]:+.5f},"
                  f" {r['ci95'][1]:+.5f}] p_t={r['p_t']:.4f} q={r['q_holm']:.4f} "
                  f"wilcoxon={r['p_wilcoxon']:.4f}")
    print('VERDICT:', res['verdict'])


if __name__ == '__main__':
    main()
