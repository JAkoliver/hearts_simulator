"""Run and analyze SearchEval matchups.

Invokes the C++ SearchEval.exe (paired duplicate deals: 1 search seat vs 3
opponent seats, against an all-opponent reference table on identical deals)
and reports the per-deal differential with CI and one-sided p-value.

Matchups:
  uplift   search(v4-m10) vs raw v4-m10   - the direct value of search
  payoff   search(v4-m10) vs raw v3-m7    - the standing anchor
  ablation uplift with uniform sampling   - isolates the belief head's value
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

EXE = os.path.join('build', 'Release', 'SearchEval.exe')
SEARCH_MODEL = 'hearts_ai_search.pt'
RAW_V4 = 'hearts_ai_grandmaster.pt'
V3_M7 = os.path.join('legacy_v3_pass238', 'hearts_ai_grandmaster_v3_milestone7.pt')

DEALS = 300
K = 32
SEED = 42

def run_matchup(name, opponent, deals=DEALS, uniform=False):
    out = f'search_eval_{name}.csv'
    cmd = [EXE, '--search-model', SEARCH_MODEL, '--opponent-model', opponent,
           '--deals', str(deals), '--k', str(K), '--seed', str(SEED), '--out', out]
    if uniform:
        cmd.append('--uniform-sampling')
    print(f'\n=== {name} ===', flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f'{name}: SearchEval failed with code {res.returncode}')
        return None

    d = pd.read_csv(out)['diff'].to_numpy(dtype=np.float64)
    mean = d.mean()
    half = 1.96 * d.std(ddof=1) / len(d) ** 0.5
    _, p = stats.ttest_1samp(d, 0.0, alternative='less')
    print(f'{name}: mean diff {mean:+.3f} +/- {half:.3f} pts/deal '
          f'(n={len(d)}, p={p:.5f} for "search seat is better")', flush=True)
    return mean, half, p

def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else DEALS
    results = {}
    results['uplift'] = run_matchup('uplift', RAW_V4, deals)
    results['payoff_v3m7'] = run_matchup('payoff_v3m7', V3_M7, deals)
    results['ablation_uniform'] = run_matchup('ablation_uniform', RAW_V4, deals, uniform=True)

    print('\n=== SUMMARY (negative = search seat better) ===')
    for name, r in results.items():
        if r:
            print(f'{name:>18}: {r[0]:+.3f} +/- {r[1]:.3f}  (p={r[2]:.5f})')

if __name__ == '__main__':
    main()
