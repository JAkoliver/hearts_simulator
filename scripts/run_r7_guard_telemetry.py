"""r7 Amendment-1 TELEMETRY: ensemble-as-rollout guard number, chunked +
RESUMABLE (docs/exploiter_league_r7_prereg.md Amendment 1). Informs, never
gates.

Design (wedge-proof + restart-proof):
  - n = 2,400 paired deals per arm, split into 12 chunks x 200 deals;
  - fixed per-chunk seeds: SEED0 + chunk * 1_000_000 (deterministic
    SearchEval => a chunk is reproducible bit-for-bit);
  - <= 2 concurrent processes, short lifetimes (~20-40 min each);
  - RESUME: a chunk CSV with exactly 200 data rows is DONE and skipped;
    partial CSVs are deleted and re-run. Survives kills and reboots.
  - pairing: same chunk seed for base and cand => per-deal paired by row.
Stats identical to orchestrator.evaluate_candidate_search (paired t).

Usage: python scripts/run_r7_guard_telemetry.py   (re-run to resume)
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED0 = 745260820          # fresh block for this telemetry (audited: > all used)
CHUNK = 200
NCHUNKS = 12
K = 32
OUT = 'equity_data/exploiter_r4/r7_guard_telemetry'


def rows(csv):
    try:
        with open(csv) as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return -1


if __name__ == '__main__':
    import numpy as np
    from scipy import stats
    import orchestrator as orch

    os.makedirs(OUT, exist_ok=True)
    assert orch._guard_match_aware()
    # exports (cheap; deterministic from the frozen checkpoints)
    orch._trace_for_search('Hall_of_Fame/hearts_model_milestone_1785322724.pth',
                           'search_gate_baseline.pt', obs_dim=556)
    orch._trace_for_search('hybrid_champ_arma_moonhead_0p1.pth',
                           'search_gate_candidate.pt', obs_dim=556)

    jobs = []   # (arm, trace, chunk_idx, csv)
    for arm, trace in (('base', 'search_gate_baseline.pt'),
                       ('cand', 'search_gate_candidate.pt')):
        for c in range(NCHUNKS):
            csv = f'{OUT}/{arm}_c{c:02d}.csv'
            n = rows(csv)
            if n == CHUNK:
                continue                     # done - resume skips it
            if n >= 0:
                os.remove(csv)               # partial - redo
            jobs.append((arm, trace, c, csv))
    print(f'telemetry: {len(jobs)} chunk(s) to run '
          f'({2 * NCHUNKS - len(jobs)} already complete)', flush=True)

    # run jobs <=2 concurrent, base chunks first (cheap), then cand
    jobs.sort(key=lambda j: (j[0] != 'base', j[2]))
    i = 0
    active = []   # (proc, csv, arm, c, t0)
    while i < len(jobs) or active:
        while i < len(jobs) and len(active) < 2:
            arm, trace, c, csv = jobs[i]
            seed = SEED0 + c * orch._GATE_SHARD_STRIDE
            p = orch._search_start(trace, orch.NEUTRAL_OPPONENT, CHUNK, K,
                                   seed, csv, equity_pt=orch.EQUITY_TRACE)
            active.append((p, csv, arm, c, time.time()))
            print(f'START {arm} c{c:02d} seed {seed}', flush=True)
            i += 1
        p, csv, arm, c, t0 = active[0]
        try:
            orch._search_finish(p, csv)
            print(f'DONE  {arm} c{c:02d} rows {rows(csv)} '
                  f'[{round((time.time() - t0) / 60, 1)} min]', flush=True)
        except Exception as e:
            print(f'FAIL  {arm} c{c:02d}: {e} - chunk will re-run on resume',
                  flush=True)
        active.pop(0)

    # assemble (only if everything is complete)
    incomplete = [f'{a}_c{c:02d}' for a in ('base', 'cand') for c in range(NCHUNKS)
                  if rows(f'{OUT}/{a}_c{c:02d}.csv') != CHUNK]
    if incomplete:
        print('INCOMPLETE chunks remain:', incomplete, '- re-run to resume')
        sys.exit(0)
    base = np.concatenate([np.genfromtxt(f'{OUT}/base_c{c:02d}.csv', delimiter=',',
                                         names=True)['diff'] for c in range(NCHUNKS)])
    cand = np.concatenate([np.genfromtxt(f'{OUT}/cand_c{c:02d}.csv', delimiter=',',
                                         names=True)['diff'] for c in range(NCHUNKS)])
    delta = cand - base
    t_stat, p_val = stats.ttest_1samp(delta, 0.0, alternative='less')
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    res = {'mean': mean, 'se': se, 'n': int(len(delta)),
           'ub95_one_sided': mean + 1.645 * se, 'p_less': float(p_val),
           'k': K, 'seed0': SEED0, 'role': 'Amendment-1 TELEMETRY (informs, never gates)'}
    print('TELEMETRY RESULT:', json.dumps(res, indent=1))
    json.dump(res, open('equity_data/verdicts/r7_guard_telemetry_n2400.json', 'w'),
              indent=1)
