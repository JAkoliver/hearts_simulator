#!/bin/bash
# ONE-SHOT gate battery for the expert-iteration candidate (pre-registered):
#   1. match gate n=3200 (mixed v3-m7/v4-m10 anchors, alpha=0.05 placement)
#   2. evolved match-aware search guard n=4800 (one-sided 95% UB <= +0.3)
# Halt-is-default: candidate promotes ONLY if both pass. Headroom 0.25.
cd /e/hearts_simulator || exit 1
export PYTHONUNBUFFERED=1
export HEARTS_HEADROOM=0.25
CAND=cand_expert_iter1_hard.pth

echo "=== MATCH GATE ($(date +%H:%M)) ==="
python -u -c "
import match_eval, json
r = match_eval.run_gate('$CAND', 'hearts_model_final.pth', matches=3200, workers=12)
json.dump({k: (float(v) if hasattr(v, 'item') or isinstance(v, (int, float)) else v)
           for k, v in r.items()} if isinstance(r, dict) else {'result': str(r)},
          open('expert_iter_gate1.json', 'w'))
"
rc=$?
echo "MATCH_GATE_RC=$rc"
if [ $rc -ne 0 ]; then echo "GATES_ABORTED"; exit 1; fi

echo "=== SEARCH GUARD ($(date +%H:%M)) ==="
python -u -c "
import orchestrator, json
ok, mean, p, se = orchestrator.evaluate_candidate_search('$CAND', deals=4800, k=64, shards=4)
ub = (mean + 1.645 * se) if (mean is not None and se) else None
print(f'GUARD ok={ok} mean={mean} se={se} UB={ub}')
json.dump({'mean': mean, 'se': se, 'ub': ub}, open('expert_iter_guard.json', 'w'))
"
echo "GUARD_RC=$?"
echo "GATES_DONE $(date +%H:%M)"
