"""League r11 stage 1 driver (docs/exploiter_league_r11_prereg.md §3/§9.1).

census -> arms -> defense probes -> cost screen -> frontier + §4 selection.
Resumable: every unit is skipped when its output exists; a unit's output is
written only at its end (tmp name, then rename), so a kill loses at most
the unit in flight and nothing holds VRAM between units. The r9 preflight
runs before EVERY unit (so PAUSE = kill, and an `r9_STOP` file stops the
chain after the current unit). HALT-DEFAULT: any failing unit stops it.

    python scripts/r11_stage1.py --mode "full speed"      # or "headroom 0.25"
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

D = 'equity_data/exploiter_r11'
PROMOTED = 'hybrid_champ_arma_moonhead_0p1.pth'
B1_CAND = 'equity_data/exploiter_r9/B1/r9_B1_end_ensemble.pth'
ATTACKERS = ('shooter_sel_v2.pth', 'shooter_sel_v1.pth')
UNIT_SEEDS = (1_100_000_000, 1_200_000_000)
DEALS, SHARDS = 5000, 10
BAR_DEF, BAR_COST, TIE = -0.40, 0.045, 0.005


def log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(D, 'stage1.log'), 'a') as f:
        f.write(line + '\n')


def unit(name, stage, mode, cmd, outputs, workers_env=None):
    if all(os.path.exists(o) for o in outputs):
        log(f'SKIP {name} (done)')
        return
    rc = subprocess.run([sys.executable, 'scripts/r9_preflight.py', '--stage', stage,
                         '--mode', mode]).returncode
    if rc != 0:
        log(f'HALT before {name}: preflight refused'); sys.exit(2)
    tmp = ['{}.tmp{}'.format(*os.path.splitext(o)) for o in outputs]   # keep the extension (np.save appends .npy)
    full = [c if c not in outputs else tmp[outputs.index(c)] for c in cmd]
    log(f'START {name}')
    t0 = time.time()
    with open(os.path.join(D, f'{name}.log'), 'w') as lf:
        rc = subprocess.run(full, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f'HALT {name} rc={rc}'); sys.exit(rc)
    for t, o in zip(tmp, outputs):
        os.replace(t, o)
    log(f'DONE {name} {time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    if a.mode == 'full speed':
        os.environ['HEARTS_NO_LOWPRI'] = '1'; os.environ.pop('HEARTS_HEADROOM', None)
    elif a.mode.startswith('headroom'):
        os.environ['HEARTS_HEADROOM'] = a.mode.split()[1]
    else:
        sys.exit('mode must be "full speed" or "headroom <x>"')
    py = [sys.executable, '-u']

    # 1. census (outcome-blind) -> arms
    cj = f'{D}/census.json'
    unit('census', 'vecprobe', a.mode,
         py + ['r11_exposure_census.py', '--matches', '1000', '--seed', '770000000',
               '--json', cj], [cj])
    arms = {}
    for tag, thi in json.load(open(cj))['arms'].items():     # E75, E50, E25
        if thi not in arms.values() and thi > 0.1:
            arms[tag] = thi                                   # duplicates dropped
    log(f'ARMS {arms}')
    for tag, thi in arms.items():
        ck = f'{D}/r11_{tag}_ensemble.pth'
        if not os.path.exists(ck):
            subprocess.run(py + ['assemble_r11_candidate.py', '--tag', tag,
                                 '--thi', str(thi)], check=True)
    ckpt = {tag: f'{D}/r11_{tag}_ensemble.pth' for tag in arms}

    # 2. defense probes (GPU; all arms in one call per attacker, CRN seeds 760M)
    for att in ATTACKERS:
        an = att.replace('.pth', '')
        js, cs = f'{D}/vecprobe_{an}.json', f'{D}/vecprobe_{an}.csv'
        unit(f'vecprobe_{an}', 'vecprobe', a.mode,
             py + ['defense_probe_vec.py', '--nets'] + list(ckpt.values()) +
             ['--base', PROMOTED, '--attacker', att, '--matches', '1000',
              '--seed', '760000000', '--out', cs, '--json', js], [cs, js])

    # 3. cost screen (CPU; identical deals for E100 reference and every arm)
    mods = dict(E100=B1_CAND, **ckpt)
    for tag, path in mods.items():
        for u, seed in enumerate(UNIT_SEEDS):
            js, nf = f'{D}/strength_{tag}_u{u}.json', f'{D}/strength_{tag}_u{u}.npy'
            unit(f'strength_{tag}_u{u}', 'cpu', a.mode,
                 py + ['neutral_raw_eval.py', '--cand', path, '--base', PROMOTED,
                       '--deals', str(DEALS), '--seed', str(seed), '--shards', str(SHARDS),
                       '--workers', str(a.workers), '--diffs-out', nf, '--json', js],
                 [nf, js])

    # 4. frontier + mechanical selection (prereg §4)
    ref = np.concatenate([np.load(f'{D}/strength_E100_u{u}.npy') for u in range(2)])
    probes = {att: json.load(open(f'{D}/vecprobe_{att.replace(".pth", "")}.json'))['results']
              for att in ATTACKERS}
    rows = {}
    for tag, path in mods.items():
        d = np.concatenate([np.load(f'{D}/strength_{tag}_u{u}.npy') for u in range(2)])
        row = {'T_hi': arms.get(tag, 0.1), 'n': len(d), 'cost': float(d.mean()),
               'cost_se': float(d.std(ddof=1) / np.sqrt(len(d))),
               'cost_vs_E100': float((d - ref).mean()),
               'cost_vs_E100_se': float((d - ref).std(ddof=1) / np.sqrt(len(d)))}
        if tag != 'E100':
            for att in ATTACKERS:
                r = probes[att][os.path.basename(path)]
                row[att] = {'delta': r['delta'], 'se': r['se']}
            row['eligible'] = bool(all(row[att]['delta'] <= BAR_DEF for att in ATTACKERS)
                                   and row['cost'] <= BAR_COST)
        rows[tag] = row
    elig = sorted((t for t in arms if rows[t]['eligible']), key=lambda t: rows[t]['cost'])
    sel = None
    if elig:
        close = [t for t in elig if rows[t]['cost'] - rows[elig[0]]['cost'] <= TIE]
        sel = min(close, key=lambda t: np.mean([rows[t][att]['delta'] for att in ATTACKERS]))
    out = {'arms': arms, 'frontier': rows, 'eligible': elig, 'selected': sel,
           'bars': {'defense': BAR_DEF, 'cost': BAR_COST, 'tie': TIE}}
    with open('equity_data/verdicts/r11_stage1_frontier.json', 'w') as f:
        json.dump(out, f, indent=1)
    log(f'FRONTIER {json.dumps(out)}')
    log(f'STAGE1_DONE selected={sel}')


if __name__ == '__main__':
    main()
