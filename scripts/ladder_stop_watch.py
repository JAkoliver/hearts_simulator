"""Enforce the PPO three-strikes stop rule on a run_loop ladder.

run_loop.py loops forever by design; the house rule (registered in
docs/exploiter_league_prereg.md "STOP RULE (binding)", applied to v6
Stage 4 by the clarification in docs/v6_prereg.md) is that THREE
CONSECUTIVE FAILED TRIALS halt the round. This watcher counts verdicts
in the ladder log and stops the loop when the third consecutive failure
lands, so the machine does not spend another night proving the same
point.

A promotion resets the counter to zero - the rule is about consecutive
failures, not cumulative ones.

Usage:
    python scripts/ladder_stop_watch.py logs/v6_stage4_ladder.log [--max 3]
"""
import argparse
import os
import re
import subprocess
import sys
import time

FAIL = re.compile(r'Experiment FAILED')
PASS = re.compile(r'Experiment SUCCESS')


def verdicts(path):
    """Consecutive-failure count and total verdicts seen so far."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        return 0, 0
    seq = [('F' if FAIL.search(ln) else 'P')
           for ln in text.splitlines()
           if FAIL.search(ln) or PASS.search(ln)]
    run = 0
    for v in seq:
        run = run + 1 if v == 'F' else 0
    return run, len(seq)


def stop_ladder():
    """Kill run_loop and anything it spawned. Windows process names are
    python3.13.exe, NOT python (launcher-discipline rule 10)."""
    killed = []
    try:
        import psutil
    except ImportError:
        subprocess.run(['taskkill', '/F', '/IM', 'python3.13.exe'])
        return ['taskkill python3.13.exe']
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cl = ' '.join(p.info['cmdline'] or [])
            if any(x in cl for x in ('run_loop.py', 'orchestrator.py',
                                     'train.py')):
                p.kill()
                killed.append(p.pid)
        except Exception:
            pass
    return killed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--max', type=int, default=3)
    ap.add_argument('--poll', type=int, default=120)
    args = ap.parse_args()

    print(f'watching {args.log}: halt at {args.max} consecutive failures')
    while True:
        run, total = verdicts(args.log)
        if run >= args.max:
            print(f'THREE-STRIKES: {run} consecutive failed trials '
                  f'({total} verdicts total) - halting the ladder')
            print('killed:', stop_ladder())
            return 0
        time.sleep(args.poll)


if __name__ == '__main__':
    sys.exit(main())
