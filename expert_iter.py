"""Expert-iteration loop: the search teacher trains its own successor.

Each iteration:
  1. GENERATE  N parallel SelfPlayGen workers play search-vs-search self-play
               (including passing-phase search) with the CURRENT search trace.
  2. DISTILL   Warm-start from the current baseline; train policy to imitate
               the teacher, value on outcomes, belief on true hands.
  3. GATE      Paired-deal evaluation (raw candidate vs raw baseline, argmax)
               via orchestrator.evaluate_candidate - same statistics as the
               PPO-era promotion gate.
  4. PROMOTE   On success: candidate becomes baseline, Hall of Fame milestone
               saved, export.py re-traces BOTH .pt files - so the next
               iteration's search teacher is built on the improved network.
               That re-export is what closes the flywheel.

Usage:
  python expert_iter.py --iterations 5 --deals 20000 --workers 12 --k 12
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import orchestrator

GEN_EXE = os.path.join('build', 'Release', 'SelfPlayGen.exe')
SEARCH_TRACE = 'hearts_ai_search.pt'
BASELINE = 'hearts_model_final.pth'
CANDIDATE = 'hearts_model_candidate.pth'

def worker_cmd(iter_dir, w, per_worker, k, pass_k, seed, cuda, rollout_tricks=-1):
    out = os.path.join(iter_dir, f'selfplay_{w}.bin')
    cmd = [GEN_EXE, '--model', SEARCH_TRACE, '--deals', str(per_worker),
           '--k', str(k), '--pass-k', str(pass_k),
           '--rollout-tricks', str(rollout_tricks),
           '--seed', str(seed), '--out', out]
    if cuda:
        cmd.append('--cuda')
    return cmd

def generate(iter_dir, deals, workers, k, pass_k, seed0, cuda, rollout_tricks=-1):
    os.makedirs(iter_dir, exist_ok=True)
    t0 = time.time()

    if cuda:
        # One process, N deal-playing threads, one coalescing inference server
        # on the GPU. Separate processes would serialize their small kernel
        # launches at the GPU and cap aggregate throughput.
        cmd = [GEN_EXE, '--model', SEARCH_TRACE, '--deals', str(deals),
               '--k', str(k), '--pass-k', str(pass_k),
               '--rollout-tricks', str(rollout_tricks),
               '--threads', str(workers), '--cuda',
               '--seed', str(seed0),
               '--out', os.path.join(iter_dir, 'selfplay.bin')]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"  SelfPlayGen exited with code {res.returncode}; "
                  f"retrying once with a fresh seed")
            for f in glob.glob(os.path.join(iter_dir, '*.bin')):
                os.remove(f)  # partial shards from the failed run
            cmd[cmd.index('--seed') + 1] = str(seed0 + 500000)
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            if res.returncode != 0:
                raise SystemExit(f"SelfPlayGen failed twice (exit {res.returncode}); aborting")
        print(f"  generated {deals} deals in {time.time() - t0:.0f}s")
        return

    per_worker = max(1, deals // workers)
    procs = []
    for w in range(workers):
        cmd = worker_cmd(iter_dir, w, per_worker, k, pass_k, seed0 + w, cuda)
        # Worker 0's progress streams through as a proxy for the whole fleet
        # (all workers advance at roughly the same rate)
        quiet = subprocess.DEVNULL if w > 0 else None
        procs.append(subprocess.Popen(cmd, stderr=quiet, stdout=subprocess.DEVNULL))
    failed = [w for w, p in enumerate(procs) if p.wait() != 0]
    for w in failed:
        # A crashed worker leaves a truncated .bin; redo its whole shard once
        # with a fresh seed rather than aborting hours of healthy work.
        rc = procs[w].returncode
        print(f"  worker {w} exited with code {rc}; retrying its shard with a fresh seed")
        cmd = worker_cmd(iter_dir, w, per_worker, k, pass_k, seed0 + w + 500000, cuda)
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if res.returncode != 0:
            raise SystemExit(f"SelfPlayGen worker {w} failed twice "
                             f"(exit {res.returncode}); aborting")
    print(f"  generated {per_worker * workers} deals in {time.time() - t0:.0f}s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--deals', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--pass-k', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--eval-deals', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=None,
                    help='base seed (default: derived from time)')
    ap.add_argument('--cuda', action='store_true',
                    help='run search inference on the GPU (workers + distill)')
    ap.add_argument('--rollout-tricks', type=int, default=-1,
                    help='truncate search rollouts after N tricks and score the '
                         'leaf with the value head (-1 = roll to round end)')
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else int(time.time()) % 1000000
    stamp = time.strftime('%m%d_%H%M')

    for it in range(args.iterations):
        print(f"\n================ Expert iteration {it + 1}/{args.iterations} ================")
        iter_dir = os.path.join('selfplay_data', f'{stamp}_iter{it}')

        print("[1/3] Generating self-play data (search teacher)...")
        generate(iter_dir, args.deals, args.workers, args.k, args.pass_k,
                 seed0=base_seed + it * args.workers, cuda=args.cuda,
                 rollout_tricks=args.rollout_tricks)

        print("[2/3] Distilling...")
        res = subprocess.run([sys.executable, 'distill.py',
                              '--data', os.path.join(iter_dir, '*.bin'),
                              '--init', BASELINE, '--out', CANDIDATE,
                              '--epochs', str(args.epochs),
                              '--device', 'cuda' if args.cuda else 'cpu'])
        if res.returncode != 0:
            raise SystemExit("Distillation failed")

        print("[3/3] Gate: paired evaluation vs baseline...")
        success, cand_mean = orchestrator.evaluate_candidate(
            CANDIDATE, BASELINE, num_deals=args.eval_deals)

        if success:
            shutil.copy(CANDIDATE, BASELINE)
            os.makedirs('Hall_of_Fame', exist_ok=True)
            milestone = f"Hall_of_Fame/hearts_model_milestone_{int(time.time())}.pth"
            shutil.copy(BASELINE, milestone)
            print(f"*** PROMOTED (mean={cand_mean:.3f}). Milestone: {milestone} ***")
            exp = subprocess.run([sys.executable, 'export.py'])
            if exp.returncode != 0:
                raise SystemExit("export.py failed - search teacher not updated")
            print("Search teacher re-exported: next iteration trains on a stronger teacher.")
        else:
            print("Iteration FAILED the gate; baseline unchanged "
                  "(next iteration re-generates with fresh seeds).")

if __name__ == '__main__':
    main()
