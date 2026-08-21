"""r7 search guard (docs/exploiter_league_r7_prereg.md gate 3): ensemble as
the search's raw policy vs the champion's, n=4800 K=32 match-aware.
Windows spawn discipline: everything behind __main__."""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    import orchestrator
    t0 = time.time()
    ok, mean, se, ub = None, None, None, None
    success, mean, se, p = None, None, None, None
    r = orchestrator.evaluate_candidate_search(
        'hybrid_champ_arma_moonhead_0p1.pth', deals=4800, k=32,
        shards=4, baseline_ckpt='Hall_of_Fame/hearts_model_milestone_1785322724.pth')
    print('guard result tuple:', r, '| seconds', round(time.time()-t0,1))
    try:
        json.dump({'result': list(r), 'seconds': round(time.time()-t0,1)},
                  open('equity_data/verdicts/r7_guard_n4800.json','w'), indent=1, default=float)
    except Exception as e:
        print('json save issue:', e)
