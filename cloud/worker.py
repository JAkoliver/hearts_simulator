"""Shard-queue worker (REQUIREMENTS R2/R3): stateless puller.

Runs on a cloud instance (inside the container) or locally for tests.
Loop: lease a chunk -> fetch/verify the model trace (once, sha-checked
against the lease, R6) -> run SelfPlayGen -> validate the shard -> upload.
On any error the worker reports /fail (best-effort) and moves on; a silent
death simply lets the lease expire. Exits cleanly when the queue is drained.

Env:
  HEARTS_QUEUE_URL    e.g. http://localhost:8757
  HEARTS_QUEUE_TOKEN  shared bearer token
  HEARTS_GEN_EXE      path to SelfPlayGen (default: /app/SelfPlayGen)
  HEARTS_WORK_DIR     scratch dir (default: /tmp/hearts_worker)
"""
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_check import validate_shard

QUEUE = os.environ.get('HEARTS_QUEUE_URL', 'http://localhost:8757')
TOKEN = os.environ.get('HEARTS_QUEUE_TOKEN', '')
GEN_EXE = os.environ.get('HEARTS_GEN_EXE', '/app/SelfPlayGen')
WORK = os.environ.get('HEARTS_WORK_DIR', '/tmp/hearts_worker')
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"


def _req(path, data=None, raw_body=None):
    url = QUEUE + path
    body = raw_body if raw_body is not None else (
        json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method='POST' if body is not None or path.startswith('/lease') else 'GET')
    req.add_header('X-Auth-Token', TOKEN)
    if raw_body is not None:
        req.add_header('Content-Type', 'application/octet-stream')
    return urllib.request.urlopen(req, timeout=300)


def fetch_model(expected_sha, dest):
    if os.path.exists(dest):
        with open(dest, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha:
                return dest
    resp = _req('/model')
    data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f"model sha mismatch: {got} != {expected_sha}")
    with open(dest, 'wb') as f:
        f.write(data)
    print(f"[worker] model fetched ({len(data)} bytes, sha ok)")
    return dest


def run_chunk(lease, model_path):
    cid = lease['chunk_id']
    out = os.path.join(WORK, f"gen_{cid}.bin")
    for f in glob.glob(os.path.join(WORK, f"gen_{cid}*.bin")):
        os.remove(f)
    cmd = [GEN_EXE, '--model', model_path,
           '--deals', str(lease['deals']), '--k', str(lease['k']),
           '--pass-k', str(lease['pass_k']),
           '--rollout-tricks', str(lease['rollout_tricks']),
           '--threads', str(lease['threads']),
           '--seed', str(lease['seed']), '--out', out]
    if lease.get('cuda', True):
        cmd.extend(['--cuda', '--bf16'])
    timeout = lease['deals'] * 30 + 300
    print(f"[worker] chunk {cid}: {lease['deals']} deals, seed {lease['seed']}")
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"SelfPlayGen exit {res.returncode}")
    # Concatenate per-thread shard files; record order is irrelevant.
    parts = sorted(glob.glob(os.path.join(WORK, f"gen_{cid}*.bin")))
    if not parts:
        raise RuntimeError("no output files produced")
    shard = os.path.join(WORK, f"chunk_{cid}.bin")
    with open(shard, 'wb') as w:
        for p in parts:
            with open(p, 'rb') as r:
                w.write(r.read())
            os.remove(p)
    ok, detail = validate_shard(shard, expect_deals=lease['deals'])
    if not ok:
        raise RuntimeError(f"local validation failed: {detail}")
    dt = time.time() - t0
    print(f"[worker] chunk {cid} generated in {dt:.0f}s "
          f"({dt / lease['deals']:.2f} s/deal); {detail}")
    return shard


def main():
    if not TOKEN:
        raise SystemExit("set HEARTS_QUEUE_TOKEN")
    os.makedirs(WORK, exist_ok=True)
    model_path = os.path.join(WORK, 'model.pt')
    while True:
        try:
            resp = _req('/lease', data={'worker_id': WORKER_ID})
        except Exception as e:
            print(f"[worker] queue unreachable ({e}); retrying in 30s")
            time.sleep(30)
            continue
        if resp.status == 204:
            print("[worker] queue drained; exiting")
            return
        lease = json.loads(resp.read())
        cid = lease['chunk_id']
        try:
            fetch_model(lease['model_sha256'], model_path)
            shard = run_chunk(lease, model_path)
            with open(shard, 'rb') as f:
                up = _req(f"/complete?chunk={cid}", raw_body=f.read())
            result = json.loads(up.read())
            if not result.get('ok'):
                print(f"[worker] orchestrator rejected chunk {cid}: "
                      f"{result.get('detail')} (it was requeued there)")
            os.remove(shard)
        except Exception as e:
            print(f"[worker] chunk {cid} failed: {e}")
            try:
                _req(f"/fail?chunk={cid}", data={'reason': str(e)[:300]})
            except Exception:
                pass  # silent death is fine - the lease will expire
            time.sleep(5)


if __name__ == '__main__':
    main()
