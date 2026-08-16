"""v6 data-scaling probe DRIVER (docs/v6_data_scaling_prereg.md, signed
2026-08-15).

Stages the registered nested subsets S1 (t0-t9) / S2 (t0-t20) of the
frozen v6 bank as HARDLINK directories (basenames unchanged, so the
frozen v6_distill.load_bank by-match split md5(basename:match_id) % 20
is bit-identical to a filtered full-bank split), then runs the frozen
arm-a recipe (v6_distill.py, UNCHANGED: --arm a --lr 1e-4 --epochs 3
--batch 512) once per (size, seed), sequentially, each as its own
unbuffered file-logged subprocess. S3 = the original bank directory.

Nothing here trains, evaluates or decides: training is v6_distill.py,
fixed-holdout evaluation is v6_probe_eval.py, inference is
v6_probe_analysis.py.

Modes:
  --stage      create/verify the S1/S2 hardlink dirs (idempotent)
  --preflight  throwaway 2-file, 2-epoch smoke run + eval + analysis
               self-test on synthetic data (writes under v6_probe/preflight)
  --run        the six registered trainings (refuses if any output exists)
  --status     print what has completed
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

BANK = 'expert_data/v6_bank'
STAGE = 'expert_data/v6_bank_probe'
OUT = 'v6_probe'
LOGS = 'logs'
SIZES = {'S1': range(0, 10), 'S2': range(0, 21)}      # thread indices
SEEDS = [20260812, 20260813]
RECIPE = ['--arm', 'a', '--lr', '1e-4', '--epochs', '3', '--batch', '512']
EXPECT_FILES = {'S1': 240, 'S2': 504, 'S3': 768}


def thread_of(path):
    base = os.path.basename(path)              # gen_G_tT.bin
    return int(base.rsplit('_t', 1)[1].split('.')[0])


def data_glob(size):
    if size == 'S3':
        return f'{BANK}/gen_*_t*.bin'
    return f'{STAGE}/{size}/gen_*_t*.bin'


def stage():
    src = sorted(glob.glob(f'{BANK}/gen_*_t*.bin'))
    assert len(src) == EXPECT_FILES['S3'], len(src)
    for size, threads in SIZES.items():
        d = f'{STAGE}/{size}'
        os.makedirs(d, exist_ok=True)
        want = [p for p in src if thread_of(p) in threads]
        for p in want:
            dst = os.path.join(d, os.path.basename(p))
            if not os.path.exists(dst):
                os.link(p, dst)                # hardlink, same basename
            # identity check: same inode/size as the source
            assert os.path.getsize(dst) == os.path.getsize(p), dst
        have = sorted(glob.glob(f'{d}/gen_*_t*.bin'))
        assert len(have) == EXPECT_FILES[size] == len(want), \
            (size, len(have), len(want))
        assert [os.path.basename(x) for x in have] == \
               [os.path.basename(x) for x in want]
        print(f'staged {size}: {len(have)} files -> {d}')
    return True


def train_one(size, seed, out_prefix, log_path, epochs=None):
    cmd = [sys.executable, '-u', 'v6_distill.py'] + RECIPE + [
        '--data', data_glob(size), '--out', out_prefix,
        '--seed', str(seed)]
    if epochs is not None:                     # preflight only
        i = cmd.index('--epochs'); cmd[i + 1] = str(epochs)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    t0 = time.time()
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'# {time.strftime("%Y-%m-%d %H:%M:%S")} '
                  f'{" ".join(cmd)}\n'); log.flush()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             env=env)
    return rc, round(time.time() - t0, 1)


def run():
    os.makedirs(OUT, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    stage()
    plan = [(s, seed) for s in ('S1', 'S2', 'S3') for seed in SEEDS]
    for size, seed in plan:
        if os.path.exists(f'{OUT}/{size}_s{seed}.json'):
            raise SystemExit(f'refusing: {OUT}/{size}_s{seed}.json exists '
                             '(no silent reruns; delete explicitly)')
    status = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': {}}
    for size, seed in plan:
        tag = f'{size}_s{seed}'
        print(f'[{time.strftime("%H:%M:%S")}] START {tag}', flush=True)
        rc, secs = train_one(size, seed, f'{OUT}/{tag}',
                             f'{LOGS}/v6_probe_{tag}.log')
        status['runs'][tag] = {'rc': rc, 'seconds': secs}
        json.dump(status, open(f'{OUT}/driver_status.json', 'w'), indent=1)
        print(f'[{time.strftime("%H:%M:%S")}] DONE  {tag} rc={rc} '
              f'{secs}s', flush=True)
        if rc != 0:
            raise SystemExit(f'HALT: {tag} exited {rc} — see log')
    status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(status, open(f'{OUT}/driver_status.json', 'w'), indent=1)
    print('ALL SIX TRAININGS DONE — run v6_probe_eval.py next', flush=True)


def preflight():
    d = f'{STAGE}/preflight'
    os.makedirs(d, exist_ok=True)
    for base in ('gen_0_t0.bin', 'gen_0_t1.bin'):
        dst = os.path.join(d, base)
        if not os.path.exists(dst):
            os.link(f'{BANK}/{base}', dst)
    os.makedirs(f'{OUT}/preflight', exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    # 2 files, 2 epochs: exercises snapshot (.ep2) + json + logging
    cmd_size = 'PRE'
    globals()['data_glob'] = (lambda s, _g=data_glob:
                              f'{d}/gen_*_t*.bin' if s == cmd_size else _g(s))
    rc, secs = train_one(cmd_size, SEEDS[0], f'{OUT}/preflight/PRE_s1',
                         f'{LOGS}/v6_probe_preflight.log', epochs=2)
    print(f'preflight training rc={rc} {secs}s')
    if rc != 0:
        raise SystemExit('preflight training FAILED')
    assert os.path.exists(f'{OUT}/preflight/PRE_s1.ep2.pth')
    assert os.path.exists(f'{OUT}/preflight/PRE_s1.json')
    print('preflight training OK')


def status():
    for p in sorted(glob.glob(f'{OUT}/*.json')):
        print(p)
    if os.path.exists(f'{OUT}/driver_status.json'):
        print(open(f'{OUT}/driver_status.json').read())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--stage', action='store_true')
    g.add_argument('--preflight', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--status', action='store_true')
    a = ap.parse_args()
    if a.stage: stage()
    elif a.preflight: preflight()
    elif a.run: run()
    else: status()
