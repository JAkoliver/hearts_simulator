"""Parallel-sharded SearchEval runner for the cross-hardware gate (R1).

The gate workload is batch-1 sequential search: one process uses ~50% of an
H100 (measured 2026-07-17). Deals are independent, so N processes with
disjoint seed ranges use the card fully and cut wall-clock ~Nx.

Pairing is preserved by construction: shard i always runs seed
(base_seed + i * 1_000_000) for deals_i deals, on BOTH stacks being
compared. Run this with identical arguments on each side, concatenate in
shard order (this script does it), and per-row pairing across sides holds
exactly as in a single run.

Usage (identical on both sides except --search-model/--exe/--out):
  python cloud/gate_shard.py --exe build/SearchEval \
      --search-model hearts_ai_search.pt \
      --opponent-model legacy_v3_pass238/hearts_ai_grandmaster_v3_milestone7.pt \
      --deals 8000 --shards 8 --base-seed 20260719 --k 32 \
      --out gate_side.csv [--no-cuda]

Then judge with:  python cloud/xhw_gate.py cloud_side.csv local_side.csv
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

SHARD_SEED_STRIDE = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True)
    ap.add_argument('--search-model', required=True)
    ap.add_argument('--opponent-model', required=True)
    ap.add_argument('--deals', type=int, required=True)
    ap.add_argument('--shards', type=int, default=8)
    ap.add_argument('--base-seed', type=int, required=True)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-cuda', action='store_true')
    ap.add_argument('--no-pass-search', action='store_true')
    args = ap.parse_args()

    # Windows CreateProcess rejects forward-slash executable paths
    args.exe = os.path.normpath(os.path.abspath(args.exe))

    per = args.deals // args.shards
    extra = args.deals % args.shards
    tmp = tempfile.mkdtemp(prefix='gate_shard_')
    procs = []
    t0 = time.time()
    for i in range(args.shards):
        n = per + (1 if i < extra else 0)
        if n == 0:
            continue
        out_i = os.path.join(tmp, f'shard_{i}.csv')
        cmd = [args.exe, '--search-model', args.search_model,
               '--opponent-model', args.opponent_model,
               '--deals', str(n), '--k', str(args.k),
               '--seed', str(args.base_seed + i * SHARD_SEED_STRIDE),
               '--out', out_i]
        if not args.no_pass_search:
            cmd.append('--pass-search')
        if not args.no_cuda:
            cmd.extend(['--cuda', '--bf16'])
        procs.append((i, n, out_i,
                      subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)))
    failed = [(i, p.wait()) for i, n, o, p in procs if p.wait() != 0]
    if failed:
        raise SystemExit(f"shards failed (id, exit): {failed}")

    # Concatenate in shard order: one header, then every shard's rows.
    rows = 0
    with open(args.out, 'w') as w:
        for idx, (i, n, out_i, p) in enumerate(procs):
            with open(out_i) as r:
                lines = r.read().splitlines()
            if idx == 0:
                w.write(lines[0] + '\n')
            body = lines[1:]
            if len(body) != n:
                raise SystemExit(f"shard {i}: {len(body)} rows, expected {n}")
            w.write('\n'.join(body) + '\n')
            rows += len(body)
    dt = time.time() - t0
    print(f"{rows} deals in {dt:.0f}s ({dt / max(rows,1):.2f} s/deal wall) "
          f"across {len(procs)} shards -> {args.out}")


if __name__ == '__main__':
    main()
