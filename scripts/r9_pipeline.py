"""League r9 pipeline driver (docs/exploiter_league_r9_prereg.md §3.6/§9).

Idempotent, resumable stage driver: `r9_state.json` records every
completed UNIT (a chunk, a trial, a readout); re-running the driver after
PAUSE_R9.cmd or a reboot skips completed units and continues. The driver
NEVER launches a GPU stage on its own without the user's mode choice: it
is invoked per stage with --mode, runs the preflight, and refuses if the
preflight refuses.

Stages (run one at a time; each is a separate invocation):
  stage0-base      promoted base rows on transfer shards 12..15 (once)
  stage0-bank      attacker bank vs the promoted ensemble (300 matches, one per process)
  stage0-distill   train_shooter (sel, v2) + certification
  trial <A1|B1|A2|B2>   train (resumable) + archive + assemble
  readout <trial>  vec probe (both clones) -> strength -> transfer (n=128)
  battery <trial>  NI (chunked) -> defense gate -> substrate verification

Usage:
  python scripts/r9_pipeline.py <stage> [<arg>] --mode "full speed"|"headroom 0.25"
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STATE = 'r9_state.json'
LOGDIR = 'equity_data/exploiter_r9'
PROMOTED = 'hybrid_champ_arma_moonhead_0p1.pth'
PROMOTED_TRACE = 'hybrid_champ_arma_moonhead_0p1_882_v2.pt'
ARMA = 'v6_stage3/arma_lr1e-4.ep3.pth'
CHAMP = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
CELL_CONFIG = {'A': 'config_r9_A.json', 'B': 'config_r9_B.json'}


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {'done': {}, 'log': []}


def save_state(st):
    tmp = STATE + '.tmp'
    json.dump(st, open(tmp, 'w'), indent=1)
    os.replace(tmp, STATE)


def mark(st, unit, info=None):
    st['done'][unit] = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'info': info}
    st['log'].append(f"{st['done'][unit]['t']} DONE {unit}")
    save_state(st)


def done(st, unit):
    return unit in st['done']


def run(cmd, log, env=None):
    """File-logged, unbuffered child; returns rc. Never chains with &&/&."""
    e = dict(os.environ); e['PYTHONUNBUFFERED'] = '1'
    if env:
        e.update(env)
    os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
    with open(log, 'a') as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=e)
        f.write(f"=== rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return rc


def preflight(stage, mode, allow_running=False):
    cmd = [sys.executable, 'scripts/r9_preflight.py', '--stage', stage, '--mode', mode]
    if allow_running:
        cmd.append('--allow-running')
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit('PREFLIGHT REFUSED - not launching')


def mode_env(mode):
    """'full speed' -> HEARTS_NO_LOWPRI=1, no pacing; 'headroom X' -> HEARTS_HEADROOM=X."""
    m = mode.strip().lower()
    if m.startswith('full'):
        return {'HEARTS_NO_LOWPRI': '1', 'HEARTS_HEADROOM': ''}
    if m.startswith('headroom'):
        frac = m.split()[1] if len(m.split()) > 1 else '0.25'
        return {'HEARTS_HEADROOM': frac}
    raise SystemExit(f'unknown mode {mode!r}: use "full speed" or "headroom 0.25"')


def md5_8(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


# ---------------------------------------------------------------- stages
def stage0_base(st, mode):
    unit = 'stage0-base'
    if done(st, unit):
        print(f'{unit} already done'); return
    preflight('searchsel', mode)
    rc = run(['bash', 'scripts/run_r9_searchsel.sh', 'base'],
             f'{LOGDIR}/stage0_base.log', mode_env(mode))
    if rc == 0:
        mark(st, unit)
    else:
        raise SystemExit(f'{unit} incomplete (rc {rc}) - re-run to resume')


def stage0_bank(st, mode, matches=300):
    unit = 'stage0-bank'
    if done(st, unit):
        print(f'{unit} already done'); return
    preflight('searchsel', mode)
    rc = run(['bash', 'scripts/run_r9_searchsel.sh', 'bank', 'selv2', PROMOTED_TRACE, str(matches)],
             f'{LOGDIR}/stage0_bank.log', mode_env(mode))
    if rc == 0:
        mark(st, unit, {'matches': matches})
    else:
        raise SystemExit(f'{unit} incomplete (rc {rc}) - re-run to resume')


def stage0_distill(st, mode):
    """train_shooter on the bank -> shooter_sel_v2.pth; certification is a
    separate unit (verify_shooter --defender promoted --search-rate <rate>)."""
    unit = 'stage0-distill'
    if done(st, unit):
        print(f'{unit} already done'); return
    preflight('vecprobe', mode)
    rc = run([sys.executable, 'train_shooter.py', '--mode', 'sel',
              '--data', f'{LOGDIR}/bank/selv2', '--out', 'shooter_sel_v2.pth'],
             f'{LOGDIR}/stage0_distill.log', mode_env(mode))
    if rc != 0:
        raise SystemExit('distill failed')
    mark(st, unit, {'md5': md5_8('shooter_sel_v2.pth')})


def trial(st, name, mode):
    cell = name[0]
    unit = f'trial-{name}'
    if done(st, unit):
        print(f'{unit} already done'); return
    cfg = CELL_CONFIG[cell]
    ck = 'train_ckpt.pt'
    resuming = os.path.exists(ck) and st.get('active_trial') == name
    preflight('trainer', mode, allow_running=False)
    if not resuming:
        for f in (ck, ck + '.prev', 'hearts_optimizer.pth', 'hearts_model_mid.pth'):
            if os.path.exists(f):
                os.remove(f)
        st['active_trial'] = name
        save_state(st)
    import shutil
    shutil.copyfile(cfg, 'config.json')
    cmd = [sys.executable, '-u', 'train.py'] + (['--resume'] if resuming else [])
    rc = run(cmd, f'{LOGDIR}/{name}/train.log', mode_env(mode))
    if rc != 0:
        raise SystemExit(f'{unit} stopped (rc {rc}) - re-run this stage to resume from the checkpoint')
    d = f'{LOGDIR}/{name}'
    os.makedirs(d, exist_ok=True)
    shutil.copyfile('hearts_model_final.pth', f'{d}/specialist_{name}_end.pth')
    shutil.copyfile('hearts_model_mid.pth', f'{d}/specialist_{name}_mid.pth')
    for tag, src in ((f'{name}_end', 'hearts_model_final.pth'), (f'{name}_mid', 'hearts_model_mid.pth')):
        if run([sys.executable, 'assemble_r8_candidate.py', '--tag', tag, '--src', src],
               f'{d}/assemble.log') != 0:
            raise SystemExit('assemble failed')
        for ext in ('_ensemble.pth', '_ensemble_882.pt'):
            os.replace(f'r8_{tag}{ext}', f'{d}/r9_{tag}{ext}')
    st['active_trial'] = None
    mark(st, unit, {'end_md5': md5_8(f'{d}/specialist_{name}_end.pth')})


def readout(st, name, mode):
    d = f'{LOGDIR}/{name}'
    cand = f'{d}/r9_{name}_end_ensemble.pth'
    mid = f'{d}/r9_{name}_mid_ensemble.pth'
    trace = f'{d}/r9_{name}_end_ensemble_882.pt'
    # 1. vec probe, both clones
    for clone in ('shooter_sel_v2.pth', 'shooter_sel_v1.pth'):
        tag = clone.replace('.pth', '')
        unit = f'readout-{name}-vecprobe-{tag}'
        if done(st, unit):
            continue
        preflight('vecprobe', mode)
        rc = run([sys.executable, 'defense_probe_vec.py', '--nets', mid, cand,
                  '--base', PROMOTED, '--attacker', clone, '--matches', '1000',
                  '--seed', '760000000', '--out', f'{d}/vecprobe_{tag}.csv',
                  '--json', f'{d}/vecprobe_{tag}.json'],
                 f'{d}/vecprobe_{tag}.log', mode_env(mode))
        if rc != 0:
            raise SystemExit(f'{unit} failed')
        mark(st, unit)
    # 2. strength
    unit = f'readout-{name}-strength'
    if not done(st, unit):
        preflight('cpu', mode)
        rc = run([sys.executable, 'neutral_raw_eval.py', '--cand', cand, '--base', PROMOTED,
                  '--deals', '5000', '--workers', '6'], f'{d}/strength.log', mode_env(mode))
        if rc not in (0, 1):     # 1 = the promoter's own PASS/FAIL, not an error
            raise SystemExit(f'{unit} failed')
        mark(st, unit)
    # 3. transfer n=128 (resumable inside)
    unit = f'readout-{name}-transfer'
    if not done(st, unit):
        preflight('searchsel', mode)
        rc = run(['bash', 'scripts/run_r9_searchsel.sh', 'transfer', name, trace],
                 f'{d}/transfer.log', mode_env(mode))
        if rc != 0:
            raise SystemExit(f'{unit} incomplete - re-run to resume')
        mark(st, unit)
    print(f'readout {name} complete - analyze with analyze_r9_readout.py {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--mode', required=True)
    a = ap.parse_args()
    st = load_state()
    os.makedirs(LOGDIR, exist_ok=True)
    if a.stage == 'stage0-base':
        stage0_base(st, a.mode)
    elif a.stage == 'stage0-bank':
        stage0_bank(st, a.mode)
    elif a.stage == 'stage0-distill':
        stage0_distill(st, a.mode)
    elif a.stage == 'trial':
        trial(st, a.arg, a.mode)
    elif a.stage == 'readout':
        readout(st, a.arg, a.mode)
    elif a.stage == 'status':
        print(json.dumps(st, indent=1))
    else:
        raise SystemExit(f'unknown stage {a.stage}')


if __name__ == '__main__':
    main()
