"""CLI for the neutral-opponent raw evaluation (diagnostic A, 2026-07-21).

The measurement itself lives in orchestrator.evaluate_candidate_neutral_raw,
where it now serves as the raw-line promotion gate; this wrapper just runs
it on demand against arbitrary checkpoints. Design rationale:
docs/ppo_v5_round2_findings.md.

Usage:
    python neutral_raw_eval.py --cand <ckpt> [--base hearts_model_final.pth]
                               [--deals 2500] [--workers 12]
"""
import argparse

from orchestrator import evaluate_candidate_neutral_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--base', default='hearts_model_final.pth')
    ap.add_argument('--deals', type=int, default=2500)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=None,
                    help='league r11: explicit seed (default: time)')
    ap.add_argument('--shards', type=int, default=None,
                    help='league r11: fixed shard count so the deal set does '
                         'not depend on --workers / headroom')
    ap.add_argument('--diffs-out', default=None, help='save per-deal diffs (.npy)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    success, mean, se, p = evaluate_candidate_neutral_raw(
        args.cand, args.base, num_deals=args.deals,
        workers=args.workers, alpha=args.alpha, seed=args.seed,
        shards=args.shards, diffs_out=args.diffs_out)
    print(f"RESULT: {'PASS' if success else 'FAIL'} at alpha={args.alpha} "
          f"(mean {mean:+.3f}, SE {se:.3f}, p={p:.5f})")
    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump({'cand': args.cand, 'base': args.base, 'deals': args.deals,
                       'seed': args.seed, 'shards': args.shards,
                       'mean': mean, 'se': se, 'p': p}, f, indent=1)


if __name__ == '__main__':
    main()
