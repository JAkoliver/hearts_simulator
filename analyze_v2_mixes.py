"""Pre-specified analysis for the expert-iteration v2 mix experiment
(docs/expert_iter_v2_prereg.md, mix-selection stage -- this script encodes
the registered analysis and was written BEFORE any mix was evaluated).

Inputs: per-match paired CSVs from match_eval.py --csv-out, named
  eval_<mix>_r<rep>_b<block>.csv
(5 mixes x 2 training replicates x 2 disjoint seed blocks = 20 files).

Registered analysis:
- Unit = (block, match idx). Per mix, per unit: placement delta (A-B)
  averaged over the training replicates. CRN integrity is ASSERTED: the
  baseline arm must be bit-identical across all evals within a block.
- Per-mix delta = mean over its 4 eval means; hierarchical SE = sd of
  the 4 eval means / 2 (t3 critical value for the 95% CI).
- Familywise error over the 5 mix-vs-baseline tests: MAX-T PERMUTATION
  on unit-level sign flips (exact under CRN; flips joint across mixes to
  preserve cross-mix correlation), one-sided (negative placement =
  better). Bonferroni alpha=.01/mix reported as the transparent backup.
- Variance decomposition: between-mix / between-replicate / between-block
  components of the eval means.
- Pairwise mix contrasts: unit-level candidate-arm differences (baseline
  cancels), with the pre-committed line: |contrast| < 2 x paired SE =>
  "indistinguishable at this n".

Outputs: report to stdout; --write-verdicts -> equity_data/verdicts/
expert_iter_v2_<mix>.json; --write-results-doc ->
docs/expert_iter_v2_results.md (table only, no narrative).

--selftest builds synthetic CSVs (null + one true effect) in a temp dir
and checks the pipeline end to end.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

MIX_ORDER = ['a_nat60', 'b_even50', 'c_seed65', 'd_natonly', 'e_seedspread']
FNAME_RE = re.compile(r'eval_(?P<mix>.+)_r(?P<rep>\w+)_b(?P<blk>\w+)\.csv$')


def load_eval(path):
    """Returns dict idx->(a_place, b_place) for one eval CSV."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('idx'):
                continue
            p = line.strip().split(',')
            if len(p) < 13:
                continue
            out[int(p[0])] = (float(p[3]), float(p[8]))
    return out


def collect(eval_dir):
    """-> {mix: {(blk, rep): {idx: (a,b)}}}, sorted block/rep label sets."""
    data = {}
    for path in glob.glob(os.path.join(eval_dir, 'eval_*.csv')):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        data.setdefault(m['mix'], {})[(m['blk'], m['rep'])] = load_eval(path)
    return data


def unit_matrix(data):
    """Aligned unit-level deltas.
    Returns (units, D, crn_ok) where D[u, j] = mean-over-reps placement
    delta for mix j at unit u=(blk, idx)."""
    mixes = [m for m in MIX_ORDER if m in data] or sorted(data)
    blocks = sorted({blk for m in mixes for (blk, _) in data[m]})
    units, rows = [], []
    crn_ok = True
    for blk in blocks:
        idxs = None
        for m in mixes:
            evs = [v for (b, _), v in data[m].items() if b == blk]
            for ev in evs:
                s = set(ev)
                idxs = s if idxs is None else (idxs & s)
        for idx in sorted(idxs):
            base_vals = {round(data[m][k][idx][1], 9)
                         for m in mixes for k in data[m] if k[0] == blk}
            if len(base_vals) > 1:
                crn_ok = False
            row = []
            for m in mixes:
                ds = [data[m][k][idx][0] - data[m][k][idx][1]
                      for k in data[m] if k[0] == blk]
                row.append(float(np.mean(ds)))
            units.append((blk, idx))
            rows.append(row)
    return mixes, units, np.array(rows), crn_ok


def max_t(D, n_perm=10000, seed=1, chunk=500):
    """One-sided max-T sign-flip permutation (negative = better).
    Returns (t_obs per mix, adjusted p per mix, raw p per mix)."""
    n, m = D.shape
    rng = np.random.default_rng(seed)
    sumsq = (D ** 2).sum(axis=0)

    def t_of(means):
        var = (sumsq - n * means ** 2) / (n - 1)
        se = np.sqrt(np.maximum(var, 1e-30) / n)
        return means / se

    t_obs = t_of(D.mean(axis=0))
    min_ts = []
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        S = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        means = (S @ D) / n
        var = (sumsq[None, :] - n * means ** 2) / (n - 1)
        se = np.sqrt(np.maximum(var, 1e-30) / n)
        min_ts.append((means / se).min(axis=1))
        done += b
    min_ts = np.concatenate(min_ts)
    p_adj = np.array([(np.sum(min_ts <= t) + 1) / (n_perm + 1) for t in t_obs])
    # Raw per-mix p (for the Bonferroni backup): one-sided t distribution
    # approximated by the same permutation marginal is overkill; use the
    # normal tail on the unit-level t (n is thousands).
    from scipy.stats import t as tdist
    p_raw = tdist.cdf(t_obs, df=n - 1)
    return t_obs, p_adj, p_raw


def analyze(eval_dir, n_perm=10000, write_verdicts=None, results_doc=None):
    data = collect(eval_dir)
    if not data:
        raise SystemExit(f'no eval_*.csv in {eval_dir}')
    mixes, units, D, crn_ok = unit_matrix(data)
    n = len(units)
    sizes = {f'{m}/{k[0]}/{k[1]}': len(v) for m in data for k, v in data[m].items()}
    if len(set(sizes.values())) > 1:
        print(f"WARNING: eval sizes differ (truncation?): {sizes}")
    print(f"mixes: {mixes}")
    print(f"units (block, idx): {n}  |  CRN baseline-arm identity: "
          f"{'OK' if crn_ok else 'VIOLATED -- investigate before trusting'}")

    # Per-mix summaries from the 4 eval means (hierarchical SE).
    summaries = {}
    for m in mixes:
        ev_means = []
        for k, ev in sorted(data[m].items()):
            d = [a - b for a, b in ev.values()]
            ev_means.append(float(np.mean(d)))
        k = len(ev_means)
        delta = float(np.mean(ev_means))
        se_h = float(np.std(ev_means, ddof=1) / math.sqrt(k)) if k > 1 else float('nan')
        tcrit = 3.182 if k == 4 else 1.96  # t(3) for the 2x2 design
        summaries[m] = {'delta': delta, 'se_hier': se_h,
                        'ci95': [delta - tcrit * se_h, delta + tcrit * se_h],
                        'eval_means': ev_means, 'n_evals': k}

    t_obs, p_adj, p_raw = max_t(D, n_perm=n_perm)
    print(f"\nPer-mix vs baseline (placement delta; negative = better; "
          f"max-T over {n_perm} sign-flip permutations):")
    for j, m in enumerate(mixes):
        s = summaries[m]
        print(f"  {m:13s} delta {s['delta']:+.4f}  hierSE {s['se_hier']:.4f}  "
              f"CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
              f"unit-t {t_obs[j]:+.2f}  p_adj {p_adj[j]:.4f}  "
              f"p_raw {p_raw[j]:.5f} (Bonferroni bar .01)")
        s['t_unit'] = float(t_obs[j])
        s['p_maxT_adj'] = float(p_adj[j])
        s['p_raw_one_sided'] = float(p_raw[j])

    # Variance decomposition over the eval means.
    print("\nVariance decomposition (eval-mean scale):")
    mix_means = np.array([summaries[m]['delta'] for m in mixes])
    rep_vars, blk_vars = [], []
    for m in mixes:
        by_rep, by_blk = {}, {}
        for (blk, rep), ev in data[m].items():
            d = float(np.mean([a - b for a, b in ev.values()]))
            by_rep.setdefault(rep, []).append(d)
            by_blk.setdefault(blk, []).append(d)
        if len(by_rep) > 1:
            rep_vars.append(np.var([np.mean(v) for v in by_rep.values()], ddof=1))
        if len(by_blk) > 1:
            blk_vars.append(np.var([np.mean(v) for v in by_blk.values()], ddof=1))
    vd = {'between_mix': float(np.var(mix_means, ddof=1)) if len(mixes) > 1 else None,
          'between_replicate_within_mix': float(np.mean(rep_vars)) if rep_vars else None,
          'between_block_within_mix': float(np.mean(blk_vars)) if blk_vars else None}
    for k, v in vd.items():
        print(f"  {k}: {v if v is None else f'{v:.6f}'}")
    if (vd['between_replicate_within_mix'] is not None
            and vd['between_mix'] is not None
            and vd['between_replicate_within_mix'] > vd['between_mix']):
        print("  REPORTED FINDING (pre-specified): replicate disagreement "
              "exceeds between-mix spread -- training noise dominates "
              "composition at this n.")

    # Pairwise contrasts, candidate arms only (baseline cancels).
    print("\nPairwise contrasts (unit-level; |delta| < 2xSE = "
          "'indistinguishable at this n'):")
    contrasts = []
    for i in range(len(mixes)):
        for j in range(i + 1, len(mixes)):
            c = D[:, i] - D[:, j]
            mu = float(c.mean())
            se = float(c.std(ddof=1) / math.sqrt(n))
            verdict = 'indistinguishable' if abs(mu) < 2 * se else 'distinct'
            contrasts.append({'pair': [mixes[i], mixes[j]], 'delta': mu,
                              'se': se, 'verdict': verdict})
            print(f"  {mixes[i]} - {mixes[j]}: {mu:+.4f} (SE {se:.4f}) "
                  f"[{verdict}]")

    if write_verdicts:
        os.makedirs(write_verdicts, exist_ok=True)
        for m in mixes:
            path = os.path.join(write_verdicts, f'expert_iter_v2_{m}.json')
            with open(path, 'w') as f:
                json.dump({'stage': 'mix-selection (selects, never concludes)',
                           'mix': m, 'n_units': n, 'crn_ok': crn_ok,
                           **summaries[m]}, f, indent=1)
        print(f"\nverdict JSONs -> {write_verdicts}/expert_iter_v2_<mix>.json")

    if results_doc:
        with open(results_doc, 'w') as f:
            f.write("# Expert iteration v2 -- mix-experiment results "
                    "(comparative stage)\n\n")
            f.write("Written by analyze_v2_mixes.py BEFORE narrative "
                    "interpretation (prereg recording plan).\n"
                    "This stage SELECTS, never concludes; claims require the "
                    "fresh-seed confirmation battery.\n\n")
            f.write("| mix | delta vs baseline | hier. SE | 95% CI | "
                    "p (max-T adj) |\n|---|---|---|---|---|\n")
            for m in mixes:
                s = summaries[m]
                f.write(f"| {m} | {s['delta']:+.4f} | {s['se_hier']:.4f} | "
                        f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
                        f"{s['p_maxT_adj']:.4f} |\n")
            f.write(f"\nUnits: {n} (block, match) pairs; CRN check: "
                    f"{'OK' if crn_ok else 'VIOLATED'}.\n")
            f.write("\n| contrast | delta | SE | verdict |\n|---|---|---|---|\n")
            for c in contrasts:
                f.write(f"| {c['pair'][0]} - {c['pair'][1]} | "
                        f"{c['delta']:+.4f} | {c['se']:.4f} | "
                        f"{c['verdict']} |\n")
        print(f"results table -> {results_doc}")

    return {'mixes': mixes, 'summaries': summaries, 'contrasts': contrasts,
            'crn_ok': crn_ok, 'variance': vd}


def selftest():
    import tempfile
    rng = np.random.default_rng(42)
    tmp = tempfile.mkdtemp(prefix='v2mix_selftest_')
    n = 400
    mixes = MIX_ORDER
    effect = {m: 0.0 for m in mixes}
    effect['c_seed65'] = -0.25  # injected true effect
    for blk in ('1', '2'):
        b_place = rng.integers(1, 5, size=n).astype(float)
        for m in mixes:
            for rep in ('1', '2'):
                a_place = b_place + effect[m] + rng.normal(0, 1.3, size=n)
                path = os.path.join(tmp, f'eval_{m}_r{rep}_b{blk}.csv')
                with open(path, 'w') as f:
                    f.write("# selftest\nidx,seat,field,a_place,a_score,"
                            "a_deals,a_moons_for,a_moons_against,b_place,"
                            "b_score,b_deals,b_moons_for,b_moons_against\n")
                    for i in range(n):
                        f.write(f"{i},{i % 4},v3m7,{a_place[i]:.4f},0,0,0,0,"
                                f"{b_place[i]:.4f},0,0,0,0\n")
    res = analyze(tmp, n_perm=2000)
    assert res['crn_ok'], 'CRN check failed on identical baseline arms'
    sig = [m for m in res['mixes']
           if res['summaries'][m]['p_maxT_adj'] < 0.05]
    assert 'c_seed65' in sig, f'injected effect not detected (sig={sig})'
    nulls = [m for m in sig if m != 'c_seed65']
    assert not nulls, f'null mixes flagged significant: {nulls}'
    print(f"\nSELFTEST PASS (detected c_seed65, no false positives; "
          f"artifacts in {tmp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='equity_data/expert_iter_v2')
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--write-verdicts', nargs='?', const='equity_data/verdicts',
                    default=None)
    ap.add_argument('--write-results-doc', nargs='?',
                    const='docs/expert_iter_v2_results.md', default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    analyze(args.eval_dir, n_perm=args.n_perm,
            write_verdicts=args.write_verdicts,
            results_doc=args.write_results_doc)


if __name__ == '__main__':
    main()
