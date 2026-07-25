"""Freeze (or unfreeze) every Hearts pipeline process instantly.

Suspends the whole process tree - training, orchestrator, gates,
generation, C++ engines - losslessly: VRAM and state stay resident, and
resume continues exactly where it stopped. Use when you need the machine
NOW (video playback, games); resume when done.

Usage:  python pause_ai.py         (or double-click PAUSE_AI.cmd)
        python pause_ai.py resume  (or double-click RESUME_AI.cmd)

Note: trial watchdogs measure wall-clock, so a long pause can trip a
"trial overdue" alert - expected, ignore it if you paused deliberately.
"""
import sys

import psutil

CMD_PATTERNS = ('run_loop.py', 'orchestrator.py', 'train.py', 'distill.py',
                'match_eval', 'neutral_raw_eval', 'promote_raw_line',
                'expert_iter', 'regate_ppo', 'gen_equity_data', 'train_equity')
EXE_PATTERNS = ('SelfPlayGen', 'SearchEval')


def targets():
    """Matched parents AND their whole process trees: multiprocessing spawn
    workers carry no script name in their command line, so tree-walking is
    the only way to catch e.g. the equity generator's 10 workers."""
    me = psutil.Process().pid
    seen = set()
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.pid == me:
                continue
            name = p.info['name'] or ''
            cmd = ' '.join(p.info['cmdline'] or [])
            if 'pause_ai' in cmd:
                continue
            if (name.startswith(EXE_PATTERNS)
                    or any(pat in cmd for pat in CMD_PATTERNS)):
                for proc in [p] + p.children(recursive=True):
                    if proc.pid not in seen and proc.pid != me:
                        seen.add(proc.pid)
                        yield proc
        except psutil.Error:
            continue


def main():
    resume = len(sys.argv) > 1 and sys.argv[1].lower() == 'resume'
    verb = 'resumed' if resume else 'PAUSED'
    count = 0
    for p in targets():
        try:
            name = p.name()
            (p.resume if resume else p.suspend)()
            print(f"  {verb}: pid {p.pid} {name}")
            count += 1
        except psutil.Error as e:
            print(f"  failed on pid {p.pid}: {e}")
    print(f"{count} process(es) {verb}."
          + ("" if resume else "  Run 'python pause_ai.py resume' "
             "(or RESUME_AI.cmd) to continue."))


if __name__ == '__main__':
    main()
