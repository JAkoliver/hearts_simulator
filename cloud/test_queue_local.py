"""Local end-to-end test of the shard queue (no cloud, no GPU required).

Exercises, against the real local SelfPlayGen build:
  A. lease expiry -> requeue with retry seed (silent-worker / preemption path)
  B. orchestrator restart -> state resumes, leased chunks recovered
  C. full drain with a real worker -> every shard validates

Usage: python cloud/test_queue_local.py [--exe build/Release/SelfPlayGen.exe]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from shard_check import validate_shard

PORT = 8791
TOKEN = 'test-token-local'


def req(path, data=None, raw=None, method=None):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=body,
                               method=method or ('POST' if body is not None else 'GET'))
    r.add_header('X-Auth-Token', TOKEN)
    return urllib.request.urlopen(r, timeout=60)


def start_orch(out_dir, lease_ttl, extra=()):
    env = dict(os.environ, HEARTS_QUEUE_TOKEN=TOKEN)
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, 'orchestrator.py'),
         '--deals', '8', '--chunk', '2', '--seed', '9000',
         # v4m10 MLP trace: the queue doesn't care which model runs, and the
         # v5 transformer is far too slow on CPU for a quick drain test
         '--model', os.path.join(ROOT, 'hearts_ai_search_v4m10.pt'),
         '--out-dir', out_dir, '--port', str(PORT),
         '--k', '8', '--pass-k', '8', '--threads', '2', '--no-cuda',
         '--lease-ttl', str(lease_ttl), *extra],
        env=env, cwd=ROOT)
    for _ in range(50):
        time.sleep(0.2)
        try:
            req('/status')
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit("orchestrator died at startup")
    raise SystemExit("orchestrator never came up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(ROOT, 'build', 'Release', 'SelfPlayGen.exe'))
    args = ap.parse_args()
    out_dir = tempfile.mkdtemp(prefix='hearts_queue_test_')
    work_dir = tempfile.mkdtemp(prefix='hearts_worker_test_')
    print(f"[test] out={out_dir}")
    orch = None
    worker = None
    try:
        # --- A: lease expiry -> requeue with retry seed ---
        orch = start_orch(out_dir, lease_ttl=3)
        lease = json.loads(req('/lease', data={'worker_id': 'ghost'}).read())
        cid, seed0 = lease['chunk_id'], lease['seed']
        print(f"[test] A: ghost leased chunk {cid} (seed {seed0}); waiting for expiry")
        deadline = time.time() + 40
        while time.time() < deadline:
            time.sleep(2)
            st = json.load(open(os.path.join(out_dir, 'queue_state.json')))
            c = st['chunks'][cid]
            if c['status'] == 'pending' and c['retries'] == 1:
                break
        assert c['status'] == 'pending' and c['retries'] == 1, f"no expiry requeue: {c}"
        assert c['seed'] == seed0 + 500000, f"retry seed wrong: {c['seed']}"
        print("[test] A PASS: expired lease requeued with fresh derived seed")

        # --- B: restart resume (leased chunk at shutdown -> pending on resume) ---
        lease2 = json.loads(req('/lease', data={'worker_id': 'ghost2'}).read())
        orch.terminate(); orch.wait(timeout=30); orch = None
        orch = start_orch(out_dir, lease_ttl=600)
        st = json.load(open(os.path.join(out_dir, 'queue_state.json')))
        assert st['chunks'][lease2['chunk_id']]['status'] == 'pending', \
            "leased chunk not recovered on restart"
        print("[test] B PASS: restart recovered the in-flight lease")

        # --- C: full drain with a real worker ---
        wenv = dict(os.environ,
                    HEARTS_QUEUE_URL=f"http://127.0.0.1:{PORT}",
                    HEARTS_QUEUE_TOKEN=TOKEN,
                    HEARTS_GEN_EXE=args.exe,
                    HEARTS_WORK_DIR=work_dir)
        worker = subprocess.Popen([sys.executable, os.path.join(HERE, 'worker.py')],
                                  env=wenv, cwd=ROOT)
        deadline = time.time() + 900
        while time.time() < deadline:
            time.sleep(5)
            status = json.loads(req('/status').read())
            if status['done'] == status['total']:
                break
        assert status['done'] == status['total'], f"queue not drained: {status}"
        worker.wait(timeout=120)  # worker exits on 204
        shards = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.startswith('chunk_') and f.endswith('.bin')]
        assert len(shards) == status['total'], f"{len(shards)} shards for {status['total']} chunks"
        total_recs = 0
        for s in shards:
            ok, detail = validate_shard(s, expect_deals=2)
            assert ok, f"{s}: {detail}"
            total_recs += int(detail.split()[1])
        print(f"[test] C PASS: drained {status['total']} chunks, "
              f"{total_recs} records, all shards valid")
        print("[test] ALL PASS")
    finally:
        for p in (worker, orch):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[test] shards left for inspection in {out_dir}")


if __name__ == '__main__':
    main()
