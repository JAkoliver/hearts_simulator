"""r7 search guard, STAGED scheduling (ops accommodation, 2026-08-20):
identical measurement to orchestrator.evaluate_candidate_search (same
exports, same per-shard seeds, same per-deal pairing, same t-test) but the
BASE shard pair runs to completion before the CANDIDATE pair starts -
4-way concurrency with the 60MB ensemble trace wedged twice (progressive
multi-process stall; ledger 2026-08-20). Sequential arms are the
established search-speed pattern (ops/run_defense_gate.sh).
Windows spawn discipline: everything behind __main__."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    import numpy as np
    from scipy import stats
    import orchestrator as orch

    CAND = 'hybrid_champ_arma_moonhead_0p1.pth'
    BASE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
    DEALS, K, SHARDS = 4800, 32, 2
    assert orch._guard_match_aware(), 'guard must run match-aware'
    orch._trace_for_search(BASE, 'search_gate_baseline.pt', obs_dim=556)
    orch._trace_for_search(CAND, 'search_gate_candidate.pt', obs_dim=556)  # net dictates 882
    seed = int(time.time())
    per = DEALS // SHARDS
    print(f'STAGED guard: {DEALS} paired deals, K={K}, {SHARDS} shard pairs, '
          f'match-aware, base pair first then cand pair; seed {seed}', flush=True)

    def run_arm(trace, tag):
        procs = []
        for i in range(SHARDS):
            s = seed + i * orch._GATE_SHARD_STRIDE
            csv = f'search_eval_gate_{tag}_s{i}.csv'
            procs.append((i, per, orch._search_start(trace, orch.NEUTRAL_OPPONENT,
                          per, K, s, csv, equity_pt=orch.EQUITY_TRACE), csv))
        parts = []
        for i, n, p, csv in procs:
            d = orch._search_finish(p, csv)
            if len(np.atleast_1d(d)) != n:
                raise RuntimeError(f'{tag} shard {i}: {len(np.atleast_1d(d))} vs {n}')
            parts.append(np.atleast_1d(d))
        return np.concatenate(parts)

    t0 = time.time()
    base = run_arm('search_gate_baseline.pt', 'base')
    print(f'base arm done in {round((time.time()-t0)/60,1)} min', flush=True)
    t1 = time.time()
    cand = run_arm('search_gate_candidate.pt', 'cand')
    print(f'cand arm done in {round((time.time()-t1)/60,1)} min', flush=True)
    delta = cand - base
    t_stat, p_val = stats.ttest_1samp(delta, 0.0, alternative='less')
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    ub = mean + 1.645 * se
    res = {'mean': mean, 'se': se, 'n': int(len(delta)), 'p_less': float(p_val),
           'ub95_one_sided': ub, 'bar': 0.3, 'within_guard': bool(ub <= 0.3),
           'seed': seed, 'k': K, 'scheduling': 'staged (base pair then cand pair)',
           'seconds': round(time.time() - t0, 1)}
    print('GUARD:', json.dumps(res, indent=1))
    json.dump(res, open('equity_data/verdicts/r7_guard_n4800.json', 'w'), indent=1)
    print(f"GUARD {'PASS' if res['within_guard'] else 'FAIL'}: delta {mean:+.3f} "
          f"(SE {se:.3f}) UB95 {ub:+.3f} vs +0.3", flush=True)
