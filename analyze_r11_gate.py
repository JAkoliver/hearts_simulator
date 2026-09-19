"""League r11 defense gate: gate-fires check and the registered readout
(docs/exploiter_league_r11_prereg.md section 5 item 2).

    python analyze_r11_gate.py fires    # after `run_r11_gate.sh ref` + E25 s0m00
    python analyze_r11_gate.py gate     # after all 320 units

fires: (1) the promoted trace's one-match run on shard-0 match-0 seeds must
equal the PROMOTED ensemble's r7 row set for that match (same engine, same
convention: a process's first match has a fresh search stream either way);
(2) E25's s0m00 row set must DIFFER from that promoted row set AND from the
champion's r7 base row set. Identical to promoted = the B1 tier never fired.
gate: PRIMARY = paired E25 - promoted (r7 cand rows) moons per match on
shards 4..11 (n=256), one-sided t, alpha 0.05, fewer moons; pooled n=320
over shards 0,1,4..11; champion baseline (r7 base rows) delta reported.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

R7 = 'equity_data/exploiter_r4/r7_gate'
D = 'equity_data/exploiter_r11'
COLS = ['seed', 'seat', 'deal', 'moon_success', 'defender_moon', 's0', 's1', 's2', 's3',
        't0', 't1', 't2', 't3']


def per_match(df):
    out = {}
    for seed, g in df.groupby('seed'):
        last = g.iloc[-1]
        totals = np.array([last[f't{i}'] for i in range(4)], dtype=float)
        shooter = int(last['seat'])
        order = np.argsort(np.argsort(totals)) + 1
        defs = [s for s in range(4) if s != shooter]
        out[int(seed)] = (int(g['moon_success'].sum()), float(np.mean(order[defs])))
    return out


def units(d, shards):
    files = [p for s in shards for p in sorted(glob.glob(f'{d}/s{s}m*.csv'))
             if not p.endswith('.tricks.csv')]
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def r7(kind, shards):
    return pd.concat([pd.read_csv(f'{R7}/{kind}_{s}.csv') for s in shards], ignore_index=True)


def rows(df, seed):
    return df[df['seed'] == seed][COLS].reset_index(drop=True)


def fires():
    seed = 720260806
    ref = rows(pd.read_csv(f'{D}/gate_ref/s0m00.csv'), seed)
    e25 = rows(pd.read_csv(f'{D}/gate/s0m00.csv'), seed)
    prom = rows(pd.read_csv(f'{R7}/cand_0.csv'), seed)
    champ = rows(pd.read_csv(f'{R7}/base_0.csv'), seed)
    same_conv = ref.equals(prom)
    diff_prom = not e25.equals(prom)
    diff_champ = not e25.equals(champ)
    print(f'ref (promoted, one-match run) == r7 promoted rows: {same_conv}')
    print(f'E25 s0m00 differs from promoted rows: {diff_prom}; from champion rows: {diff_champ}')
    ok = same_conv and diff_prom and diff_champ
    print('GATE-FIRES ' + ('PASS' if ok else 'FAIL'))
    return ok


def paired(c, b):
    keys = sorted(set(c) & set(b))
    d = np.array([c[k][0] - b[k][0] for k in keys], dtype=float)
    n = len(d); mean = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else 0.0
    return {'n': n, 'delta': mean, 'se': se, 't': t,
            'p_one_sided': float(stats.t.cdf(t, n - 1)),
            'cand_mean': float(np.mean([c[k][0] for k in keys])),
            'base_mean': float(np.mean([b[k][0] for k in keys])),
            'cand_def_place': float(np.mean([c[k][1] for k in keys])),
            'base_def_place': float(np.mean([b[k][1] for k in keys]))}


def gate():
    prim, allsh = [4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
    cand = per_match(units(f'{D}/gate', allsh))
    prom = per_match(r7('cand', allsh)); champ = per_match(r7('base', allsh))
    cp = {k: v for k, v in cand.items() if (k - 720260806) // 1000000 in prim}
    primary = paired(cp, prom)
    primary['pass'] = bool(primary['p_one_sided'] < 0.05 and primary['delta'] < 0)
    out = {'primary_shards': '4..11 vs promoted r7 rows', 'primary': primary,
           'pooled_n320_vs_promoted': paired(cand, prom),
           'pooled_n320_vs_champion': paired(cand, champ),
           'bar_promoted_moons_primary': primary['base_mean']}
    assert primary['n'] == 256 and out['pooled_n320_vs_promoted']['n'] == 320, 'incomplete rows'
    with open('equity_data/verdicts/r11_E25_defense_gate.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print('DEFENSE GATE ' + ('PASS' if primary['pass'] else 'FAIL'))
    return primary['pass']


if __name__ == '__main__':
    sys.exit(0 if {'fires': fires, 'gate': gate}[sys.argv[1]]() else 1)
