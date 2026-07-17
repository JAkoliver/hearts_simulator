"""Shard-queue orchestrator (REQUIREMENTS R2/R3): the one stateful component.

Runs locally (or on one small non-spot node). Workers PULL chunk leases over
HTTP, run SelfPlayGen, and upload finished shards. Leases have a TTL; a
worker that goes silent (crash, spot preemption) forfeits its chunk back to
the queue with a fresh derived seed. State is a JSON file flushed on every
transition, so an orchestrator restart resumes exactly where it stopped.

Usage:
  python cloud/orchestrator.py --deals 3500 --chunk 250 --seed 123456 \
      --model hearts_ai_search.pt --out-dir selfplay_data/cloud_iter0 \
      --port 8757 [--k 64] [--pass-k 24] [--threads 26] [--no-cuda]
  HEARTS_QUEUE_TOKEN must be set (shared bearer token; workers send it as
  X-Auth-Token).

Endpoints (all require the token):
  POST /lease     {"worker_id": str}        -> chunk params | 204 when drained
  POST /complete?chunk=ID                    body = shard bytes
  POST /fail?chunk=ID {"reason": str}       -> immediate requeue
  GET  /model                                -> model trace bytes (R6)
  GET  /status                               -> queue state JSON
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_check import validate_shard

RETRY_SEED_OFFSET = 500000  # mirrors expert_iter's per-chunk retry seeds
MAX_RETRIES = 4


class Queue:
    def __init__(self, state_path, out_dir, params, deals, chunk, seed, lease_ttl):
        self.lock = threading.Lock()
        self.state_path = state_path
        self.out_dir = out_dir
        self.params = params
        self.lease_ttl = lease_ttl
        if os.path.exists(state_path):
            with open(state_path) as f:
                self.state = json.load(f)
            # Restart recovery: leases from a dead orchestrator epoch are
            # unverifiable - requeue anything not done (R3: instance death
            # and chunk failure are the same event).
            for c in self.state['chunks'].values():
                if c['status'] == 'leased':
                    c['status'] = 'pending'
                    c['lease_worker'] = None
                    c['lease_until'] = 0
            self._flush()
            print(f"[queue] resumed from {state_path}: "
                  f"{self._count('done')} done / {len(self.state['chunks'])}")
        else:
            chunks = {}
            done_deals = 0
            i = 0
            while done_deals < deals:
                n = min(chunk, deals - done_deals)
                chunks[str(i)] = {
                    'seed': seed + i * 1000, 'deals': n, 'status': 'pending',
                    'lease_worker': None, 'lease_until': 0, 'retries': 0,
                    'shard': None,
                }
                done_deals += n
                i += 1
            self.state = {'params': params, 'chunks': chunks}
            self._flush()
            print(f"[queue] created {len(chunks)} chunks of <= {chunk} deals")

    def _flush(self):
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.state_path)

    def _count(self, status):
        return sum(1 for c in self.state['chunks'].values() if c['status'] == status)

    def lease(self, worker_id):
        with self.lock:
            now = time.time()
            for cid, c in self.state['chunks'].items():
                if c['status'] == 'pending':
                    c['status'] = 'leased'
                    c['lease_worker'] = worker_id
                    c['lease_until'] = now + self.lease_ttl
                    self._flush()
                    return dict(chunk_id=cid, seed=c['seed'], deals=c['deals'],
                                **self.params)
            return None

    def complete(self, cid, body):
        with self.lock:
            c = self.state['chunks'].get(cid)
            if c is None or c['status'] == 'done':
                return False, "unknown or already-done chunk"
            shard = os.path.join(self.out_dir, f"chunk_{cid}.bin")
            tmp = shard + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(body)
            ok, detail = validate_shard(tmp, expect_deals=c['deals'])
            if not ok:
                os.remove(tmp)
                self._requeue(c, f"upload failed validation: {detail}")
                return False, detail
            os.replace(tmp, shard)
            c['status'] = 'done'
            c['shard'] = shard
            self._flush()
            print(f"[queue] chunk {cid} done ({detail}); "
                  f"{self._count('done')}/{len(self.state['chunks'])}")
            return True, detail

    def fail(self, cid, reason):
        with self.lock:
            c = self.state['chunks'].get(cid)
            if c is None or c['status'] != 'leased':
                return
            self._requeue(c, reason)

    def _requeue(self, c, reason):
        # A retried chunk gets a fresh derived seed so a poisoned seed can't
        # wedge the queue (R2); after MAX_RETRIES it is parked for a human.
        c['retries'] += 1
        c['seed'] += RETRY_SEED_OFFSET
        c['lease_worker'] = None
        c['lease_until'] = 0
        c['status'] = 'pending' if c['retries'] <= MAX_RETRIES else 'parked'
        self._flush()
        print(f"[queue] requeue (retry {c['retries']}): {reason}"
              + (" -> PARKED" if c['status'] == 'parked' else ""))

    def sweep_expired(self):
        with self.lock:
            now = time.time()
            for cid, c in self.state['chunks'].items():
                if c['status'] == 'leased' and now > c['lease_until']:
                    self._requeue(c, f"lease expired (worker {c['lease_worker']})")

    def drained(self):
        with self.lock:
            return all(c['status'] in ('done', 'parked')
                       for c in self.state['chunks'].values())

    def status(self):
        with self.lock:
            out = {'done': self._count('done'), 'pending': self._count('pending'),
                   'leased': self._count('leased'), 'parked': self._count('parked'),
                   'total': len(self.state['chunks'])}
            return out


def make_handler(queue, token, model_path, model_sha):
    class Handler(BaseHTTPRequestHandler):
        def _auth(self):
            if self.headers.get('X-Auth-Token') != token:
                self.send_response(401)
                self.end_headers()
                return False
            return True

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._auth():
                return
            if self.path == '/model':
                with open(model_path, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Model-SHA256', model_sha)
                self.end_headers()
                self.wfile.write(data)
            elif self.path == '/status':
                self._json(200, queue.status())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self._auth():
                return
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length else b''
            if self.path == '/lease':
                worker = json.loads(body or b'{}').get('worker_id', 'anon')
                lease = queue.lease(worker)
                if lease is None:
                    self.send_response(204)
                    self.end_headers()
                else:
                    lease['model_sha256'] = model_sha
                    self._json(200, lease)
            elif self.path.startswith('/complete'):
                cid = self.path.split('chunk=')[-1]
                ok, detail = queue.complete(cid, body)
                self._json(200 if ok else 422, {'ok': ok, 'detail': detail})
            elif self.path.startswith('/fail'):
                cid = self.path.split('chunk=')[-1]
                reason = json.loads(body or b'{}').get('reason', 'worker-reported')
                queue.fail(cid, reason)
                self._json(200, {'ok': True})
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            pass  # queue prints its own transitions

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deals', type=int, required=True)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--model', default='hearts_ai_search.pt')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--port', type=int, default=8757)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--pass-k', type=int, default=24)
    ap.add_argument('--threads', type=int, default=26,
                    help='worker deal-threads (cloud boxes have more cores)')
    ap.add_argument('--rollout-tricks', type=int, default=-1)
    ap.add_argument('--no-cuda', action='store_true')
    ap.add_argument('--lease-ttl', type=int, default=None,
                    help='seconds before a silent worker forfeits its chunk '
                         '(default: 20s/deal + 600)')
    ap.add_argument('--exit-when-drained', action='store_true')
    args = ap.parse_args()

    token = os.environ.get('HEARTS_QUEUE_TOKEN')
    if not token:
        raise SystemExit("set HEARTS_QUEUE_TOKEN")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.model, 'rb') as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    params = {'k': args.k, 'pass_k': args.pass_k, 'threads': args.threads,
              'rollout_tricks': args.rollout_tricks, 'cuda': not args.no_cuda}
    ttl = args.lease_ttl or (20 * args.chunk + 600)
    queue = Queue(os.path.join(args.out_dir, 'queue_state.json'),
                  args.out_dir, params, args.deals, args.chunk, args.seed, ttl)

    server = ThreadingHTTPServer(('0.0.0.0', args.port),
                                 make_handler(queue, token, args.model, model_sha))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[queue] serving on :{args.port}  model sha256 {model_sha[:12]}...  "
          f"lease ttl {ttl}s")
    try:
        while True:
            time.sleep(15)
            queue.sweep_expired()
            if args.exit_when_drained and queue.drained():
                print(f"[queue] drained: {queue.status()}")
                break
    except KeyboardInterrupt:
        print(f"[queue] interrupted: {queue.status()} (state persisted; rerun to resume)")
    server.shutdown()


if __name__ == '__main__':
    main()
