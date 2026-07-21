"""Gates + promotion for the cloud iteration - mirrors expert_iter.py [3/3]
verbatim. All logic under the __main__ guard: Windows multiprocessing
re-imports this file in every worker, and unguarded top-level code turns
the pool into a crash-respawn loop (cost: 2.3h on 2026-07-18)."""
import json
import os
import shutil
import subprocess
import sys
import time


def main():
    os.chdir(r'e:\hearts_simulator')
    sys.path.insert(0, r'e:\hearts_simulator')
    import orchestrator

    CANDIDATE = 'hearts_model_candidate.pth'
    BASELINE = 'hearts_model_final.pth'

    with open('config.json') as f:
        cfg = json.load(f)

    t0 = time.time()
    raw_sig, cand_mean, raw_diff = orchestrator.evaluate_candidate(
        CANDIDATE, BASELINE, num_deals=2500)
    print(f"RAW: diff {raw_diff:+.3f} (guard +{cfg.get('raw_guard_threshold', 0.3)}) "
          f"in {time.time() - t0:.0f}s", flush=True)
    guard = cfg.get('raw_guard_threshold', 0.3)
    if raw_diff > guard:
        print(f"Raw guard FAILED ({raw_diff:+.3f} > +{guard}); skipping search gate.",
              flush=True)
        success = False
    else:
        t1 = time.time()
        # NOTE (2026-07-21): local promotion is now raw-line (neutral raw gate
        # promotes, search gate guards - see orchestrator.main). This cloud
        # driver still uses search-gate promotion; rework it before the next
        # cloud expert iteration.
        success, sg_mean, sg_p, sg_se = orchestrator.evaluate_candidate_search(
            CANDIDATE,
            deals=cfg.get('search_gate_deals', 600),
            k=cfg.get('search_gate_k', 32),
            alpha=cfg.get('search_gate_alpha', 0.05))
        print(f"SEARCH GATE: mean {sg_mean} p {sg_p} in {time.time() - t1:.0f}s "
              f"-> {'PASS' if success else 'FAIL'}", flush=True)

    if success:
        shutil.copy(CANDIDATE, BASELINE)
        os.makedirs('Hall_of_Fame', exist_ok=True)
        milestone = f"Hall_of_Fame/hearts_model_milestone_{int(time.time())}.pth"
        shutil.copy(BASELINE, milestone)
        print(f"*** PROMOTED (mean={cand_mean:.3f}). Milestone: {milestone} ***",
              flush=True)
        exp = subprocess.run([sys.executable, 'export.py'])
        if exp.returncode != 0:
            raise SystemExit("export.py failed - search teacher not updated")
        print("Search teacher re-exported.", flush=True)
    else:
        print("Iteration FAILED the gate; baseline unchanged.", flush=True)


if __name__ == '__main__':
    main()
