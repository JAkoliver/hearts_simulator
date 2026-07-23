"""Raw-line manual promotion driver (2026-07-21).

Runs the full raw-line promotion pipeline for an arbitrary candidate:
  1. neutral raw gate (promoter)          - orchestrator.evaluate_candidate_neutral_raw
  2. search non-regression guard           - orchestrator.evaluate_candidate_search + margin UB
  3. promote: baseline copy, Hall of Fame milestone, export.py re-trace

Usage: python promote_raw_line.py [candidate.pth]
"""
import json
import os
import shutil
import subprocess
import sys
import time

import scipy.stats as stats

import orchestrator


def main():
    candidate = sys.argv[1] if len(sys.argv) > 1 else 'cand_A_trial3repro.pth'
    with open('config.json') as f:
        cfg = json.load(f)

    ok, mean, se, p = orchestrator.evaluate_candidate_neutral_raw(
        candidate, 'hearts_model_final.pth',
        num_deals=cfg.get('raw_gate_deals', 2500),
        workers=cfg.get('raw_gate_workers', 12),
        alpha=cfg.get('raw_gate_alpha', 0.05))
    if not ok:
        print(f"PROMOTION ABORTED: neutral raw gate FAIL ({mean:+.3f}, p={p:.5f})")
        return

    _, sg_mean, sg_p, sg_se = orchestrator.evaluate_candidate_search(
        candidate,
        deals=cfg.get('search_gate_deals', 2400),
        k=cfg.get('search_gate_k', 32),
        alpha=cfg.get('search_gate_alpha', 0.05))
    if sg_mean is None:
        print("PROMOTION ABORTED: search guard unavailable")
        return
    margin = cfg.get('search_guard_margin', 0.3)
    ub = sg_mean + float(stats.t.ppf(0.95, cfg.get('search_gate_deals', 2400) - 1)) * sg_se
    print(f"Search non-regression guard: delta {sg_mean:+.3f} (SE {sg_se:.3f}), "
          f"one-sided 95% UB {ub:+.3f} vs margin +{margin}")
    if ub > margin:
        print("PROMOTION ABORTED: search regression")
        return

    shutil.copy(candidate, 'hearts_model_final.pth')
    milestone = f"Hall_of_Fame/hearts_model_milestone_{int(time.time())}.pth"
    shutil.copy('hearts_model_final.pth', milestone)

    # Optimizer carry-through (2026-07-23): hearts_optimizer.pth must match
    # hearts_model_final.pth. A stale mismatch either starts future PPO from
    # wrong Adam moments (same arch) or crashes train.py (different arch) -
    # all three post-promotion PPO trials of 2026-07-21 ran on stale moments.
    optim_stash = os.path.splitext(candidate)[0] + '.optim.pth'
    if os.path.exists(optim_stash):
        shutil.copy(optim_stash, 'hearts_optimizer.pth')
        print(f"Optimizer state carried through from {optim_stash}.")
    elif os.path.exists('hearts_optimizer.pth'):
        os.remove('hearts_optimizer.pth')
        print("No optimizer stash for this candidate - stale "
              "hearts_optimizer.pth removed (future PPO starts from fresh "
              "Adam moments rather than wrong ones).")

    ledger = orchestrator.get_ledger()
    ledger['baseline_score'] = mean
    orchestrator.write_ledger(ledger)

    print(f"*** PROMOTED (raw-line): neutral raw {mean:+.3f} (SE {se:.3f}, "
          f"p={p:.5f}); search guard UB {ub:+.3f}. Milestone: {milestone} ***")
    res = subprocess.run([sys.executable, 'export.py'])
    if res.returncode != 0:
        raise SystemExit("export.py FAILED - deployed traces are STALE")
    print("Traces re-exported (hearts_ai_grandmaster.pt, hearts_ai_search.pt).")


if __name__ == '__main__':
    main()
