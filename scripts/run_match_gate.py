"""Match-gate runner (candidate vs baseline, match_eval.run_gate).

Exists because ad-hoc scripts around multiprocessing are a Windows
footgun: on 2026-08-13 an unguarded pre-flight (run_gate called at
module top level) fork-bombed the machine - every spawn worker
re-imported __main__, re-ran the gate, and spawned its own pool until
31 GB of RAM was gone. Windows multiprocessing entry points MUST live
behind `if __name__ == '__main__':` - use this script, never stdin
one-liners.

Usage (from repo root):
    python scripts/run_match_gate.py --cand v6_stage4/trial1.pth \
        [--base Hall_of_Fame/hearts_model_milestone_1785322724.pth] \
        [--matches 3200] [--workers 12] [--json out.json]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--base',
                    default='Hall_of_Fame/hearts_model_milestone_1785322724.pth')
    ap.add_argument('--matches', type=int, default=3200)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    import match_eval
    t0 = time.time()
    r = match_eval.run_gate(args.cand, args.base,
                            matches=args.matches, workers=args.workers)
    r['seconds'] = round(time.time() - t0, 1)
    r['cand'], r['base'] = args.cand, args.base
    print(json.dumps(r, indent=1, default=float))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=1, default=float)


if __name__ == '__main__':
    main()
