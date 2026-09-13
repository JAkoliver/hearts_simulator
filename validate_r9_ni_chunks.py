"""League r9 null contract (§3.5): the chunk-resumable NI path produces the
SAME per-match rows as the unchunked path on an explicit seed.

Runs match_eval.run_gate twice at small n (same cand/base/seed/workers):
unchunked (chunk_dir=None) and chunked (chunk_dir=<scratch>), then a
THIRD time on the chunked dir with two job files deleted (simulated
partial run) - the re-run must rebuild only those and still match.
Rows are compared as exact tuples. Windows: __main__ guard required.
Usage: python validate_r9_ni_chunks.py [--matches 160] [--workers 8]
"""
import argparse
import os
import shutil
import sys

CAND = 'hybrid_champ_arma_moonhead_0p1.pth'
BASE = 'Hall_of_Fame/hearts_model_milestone_1785322724.pth'
SEED = 1789299870          # the r8 NI seed
SCRATCH = 'equity_data/exploiter_r9/ni_chunks_aa'


def rows_of(csv_path):
    with open(csv_path) as f:
        return [l for l in f.read().splitlines() if not l.startswith('#')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, default=160)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import match_eval
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    match_eval.run_gate(CAND, BASE, matches=a.matches, workers=a.workers, seed=SEED,
                        csv_out=f'{SCRATCH}/unchunked.csv')
    match_eval.run_gate(CAND, BASE, matches=a.matches, workers=a.workers, seed=SEED,
                        csv_out=f'{SCRATCH}/chunked.csv', chunk_dir=f'{SCRATCH}/chunks')
    r0, r1 = rows_of(f'{SCRATCH}/unchunked.csv'), rows_of(f'{SCRATCH}/chunked.csv')
    if r0 != r1:
        print(f'FAIL: chunked rows differ from unchunked ({sum(x != y for x, y in zip(r0, r1))} rows)')
        sys.exit(1)
    print(f'chunked == unchunked: {len(r0) - 1} rows identical')
    # simulated partial run: drop two job files, re-run -> only those rebuild
    for w in (1, 5):
        p = f'{SCRATCH}/chunks/job_{w}.pkl'
        if os.path.exists(p):
            os.remove(p)
    match_eval.run_gate(CAND, BASE, matches=a.matches, workers=a.workers, seed=SEED,
                        csv_out=f'{SCRATCH}/resumed.csv', chunk_dir=f'{SCRATCH}/chunks')
    r2 = rows_of(f'{SCRATCH}/resumed.csv')
    if r0 != r2:
        print('FAIL: resumed rows differ from unchunked'); sys.exit(1)
    print(f'resumed (2 jobs rebuilt) == unchunked: {len(r2) - 1} rows identical')
    print('R9 NI-CHUNK CONTRACT PASS')


if __name__ == '__main__':
    main()
