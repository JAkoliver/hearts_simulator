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
import json
import os
import shutil
import subprocess
import sys
import time

import orchestrator

GEN_EXE = os.path.join('build', 'Release', 'SelfPlayGen.exe')
# Overridable via --teacher: hearts_ai_search_v4m10.pt runs ~4x faster than
# the v5 trace at (currently) tied searched strength - the efficient teacher
# until a v5-lineage search beats it on neutral ground.
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
        cmd.extend(['--cuda', '--bf16'])
    return cmd

def generate(iter_dir, deals, workers, k, pass_k, seed0, cuda, rollout_tricks=-1):
    os.makedirs(iter_dir, exist_ok=True)
    t0 = time.time()

    if cuda:
        # One process, N deal-playing threads, one coalescing inference server
        # on the GPU - run in CHUNKS of fresh processes purely for RETRY
        # GRANULARITY: a chunk failure redoes only that chunk instead of
        # deleting the whole iteration's data (a whole-run retry once
        # destroyed 9 hours of deals). Model-load overhead is ~10s/chunk.
        #
        # There is NO process-age decay (July 16 instrumented runs: per-launch
        # forward time at fixed batch shapes flat over 70k+ launches / 400
        # deals). The historical "2 -> 15 s/deal decay" was the unbucketed
        # server's shape-cache/VRAM leak (fixed by BucketRows), and the
        # residual "decay despite bucketing" was a measurement artifact:
        # short byte-rate windows read synchronized early-deal completions as
        # ~3 s/deal, then desynced steady state as ~9. Measured steady rate
        # with the v5 teacher at K=64 / 14 threads: 6.32 s/deal (400-deal
        # run, 2026-07-17, SDPA traces; see docs/speed_ledger.md).
        chunk = 250
        done = 0
        c = 0
        while done < deals:
            n = min(chunk, deals - done)
            out = os.path.join(iter_dir, f'selfplay_c{c}.bin')
            cmd = [GEN_EXE, '--model', SEARCH_TRACE, '--deals', str(n),
                   '--k', str(k), '--pass-k', str(pass_k),
                   '--rollout-tricks', str(rollout_tricks),
                   '--threads', str(workers), '--cuda', '--bf16',
                   '--seed', str(seed0 + c * 1000),
                   '--out', out]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            if res.returncode != 0:
                print(f"  chunk {c} exited with code {res.returncode}; "
                      f"retrying this chunk with a fresh seed")
                stale = glob.glob(os.path.join(iter_dir, f'selfplay_c{c}_*.bin'))
                if os.path.exists(out):
                    stale.append(out)
                for f in stale:
                    os.remove(f)
                cmd[cmd.index('--seed') + 1] = str(seed0 + c * 1000 + 500000)
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
                if res.returncode != 0:
                    raise SystemExit(f"chunk {c} failed twice (exit {res.returncode}); aborting")
            done += n
            c += 1
            print(f"  chunk {c} done: {done}/{deals} deals, {time.time() - t0:.0f}s elapsed")
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
    global SEARCH_TRACE
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--deals', type=int, default=6000)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--pass-k', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--eval-deals', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=None,
                    help='base seed (default: derived from time)')
    ap.add_argument('--cuda', action='store_true',
                    help='run search inference on the GPU (workers + distill)')
    ap.add_argument('--rollout-tricks', type=int, default=-1,
                    help='truncate search rollouts after N tricks and score the '
                         'leaf with the value head (-1 = roll to round end)')
    ap.add_argument('--teacher', default=SEARCH_TRACE,
                    help='search trace used for generation (default: current '
                         'teacher; hearts_ai_search_v4m10.pt is ~4x faster at '
                         'tied strength)')
    args = ap.parse_args()
    SEARCH_TRACE = args.teacher

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
        # The measured recipe (July sweep): sharpen 2 (sharpen 1 is provably
        # too soft to flip decisions), reduced aux losses, no oracle loss
        res = subprocess.run([sys.executable, 'distill.py',
                              '--data', os.path.join(iter_dir, '*.bin'),
                              '--init', BASELINE, '--out', CANDIDATE,
                              '--epochs', str(args.epochs), '--lr', '5e-5',
                              '--sharpen', '2',
                              '--value-coef', '0.25', '--aux-coef', '0.25',
                              '--oracle-coef', '0',
                              '--device', 'cuda' if args.cuda else 'cpu'])
        if res.returncode != 0:
            raise SystemExit("Distillation failed")

        print("[3/3] Gates: raw guard, then searched strength decides...")
        with open('config.json') as f:
            cfg = json.load(f)
        raw_sig, cand_mean, raw_diff = orchestrator.evaluate_candidate(
            CANDIDATE, BASELINE, num_deals=args.eval_deals)
        guard = cfg.get('raw_guard_threshold', 0.3)
        if raw_diff > guard:
            print(f"Raw guard FAILED ({raw_diff:+.3f} > +{guard}); skipping search gate.")
            success = False
        else:
            success, sg_mean, sg_p = orchestrator.evaluate_candidate_search(
                CANDIDATE,
                deals=cfg.get('search_gate_deals', 600),
                k=cfg.get('search_gate_k', 32),
                alpha=cfg.get('search_gate_alpha', 0.05))

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
