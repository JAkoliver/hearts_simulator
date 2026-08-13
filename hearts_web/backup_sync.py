"""Periodic off-host backup of the site's append-only data files to an
S3-compatible bucket (Cloudflare R2, AWS S3, MinIO...).

Why: the container/VPS disk is not the durable home of telemetry - the
match logs are training data. This module uploads each configured file
whenever it has grown/changed, on a background thread started by
server.py at boot (guarded by config; absent config = disabled, which
keeps local development exactly as before).

Stdlib only (hand-rolled AWS SigV4 for PUT/HEAD): no new dependencies.

Config (site_config.py):
    BACKUP_S3 = {
        'endpoint': 'https://<account_id>.r2.cloudflarestorage.com',
        'bucket': 'perilune-data',
        'access_key': '...',
        'secret_key': '...',
        'prefix': 'hearts_web/',       # optional key prefix
        'interval_s': 3600,            # optional, default hourly
        'region': 'auto',              # R2 uses 'auto'; AWS: real region
    }

Files uploaded: the append-only .jsonl set + supporters.json (see
DATA_FILES). Keys are '<prefix><filename>'; each upload replaces the
object, so the bucket always holds the latest full copy (append-only
files make "latest full copy" the correct durable form).

One-shot connectivity test (run after creating the bucket):
    python -m hearts_web.backup_sync --test
"""
import datetime
import hashlib
import hmac
import os
import threading
import time
import urllib.parse
import urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))

# Introspection for the admin page: current backup state, updated by the
# thread. Read-only elsewhere.
STATUS = {'enabled': False, 'last_cycle': None, 'uploads': {}, 'errors': {}}

DATA_FILES = [
    '.share_secret',        # share-link HMAC key: losing it silently
                            # invalidates every link players have posted
    'match_logs.jsonl',
    'daily_attempts.jsonl',
    'player_names.jsonl',
    'player_keys.jsonl',
    'progress_stats.jsonl',
    'search_shares.jsonl',
    'supporters.json',
]


def _sign(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _sigv4_headers(method, url, region, access_key, secret_key, payload):
    """Minimal AWS Signature V4 for a single S3 request."""
    u = urllib.parse.urlparse(url)
    host = u.netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime('%Y%m%dT%H%M%SZ')
    datestamp = now.strftime('%Y%m%d')
    payload_hash = hashlib.sha256(payload).hexdigest()
    canonical_headers = (f'host:{host}\n'
                         f'x-amz-content-sha256:{payload_hash}\n'
                         f'x-amz-date:{amz_date}\n')
    signed_headers = 'host;x-amz-content-sha256;x-amz-date'
    canonical_request = '\n'.join([
        method, urllib.parse.quote(u.path, safe='/-_.~'), '',
        canonical_headers, signed_headers, payload_hash])
    scope = f'{datestamp}/{region}/s3/aws4_request'
    string_to_sign = '\n'.join([
        'AWS4-HMAC-SHA256', amz_date, scope,
        hashlib.sha256(canonical_request.encode()).hexdigest()])
    k = _sign(_sign(_sign(_sign(('AWS4' + secret_key).encode(), datestamp),
                          region), 's3'), 'aws4_request')
    signature = hmac.new(k, string_to_sign.encode(),
                         hashlib.sha256).hexdigest()
    return {
        'x-amz-date': amz_date,
        'x-amz-content-sha256': payload_hash,
        'Authorization': (
            f'AWS4-HMAC-SHA256 Credential={access_key}/{scope}, '
            f'SignedHeaders={signed_headers}, Signature={signature}'),
    }


def _request(method, url, region, ak, sk, payload=b''):
    headers = _sigv4_headers(method, url, region, ak, sk, payload)
    req = urllib.request.Request(url, data=payload if method == 'PUT'
                                 else None, method=method, headers=headers)
    return urllib.request.urlopen(req, timeout=60)


def _put(cfg, key, payload):
    url = f"{cfg['endpoint'].rstrip('/')}/{cfg['bucket']}/{key}"
    with _request('PUT', url, cfg.get('region', 'auto'),
                  cfg['access_key'], cfg['secret_key'], payload) as r:
        return r.status


class BackupThread(threading.Thread):
    """Uploads changed data files every interval_s; never raises out."""

    def __init__(self, cfg):
        super().__init__(daemon=True, name='backup-sync')
        self.cfg = cfg
        self._seen = {}          # path -> (size, mtime) of last upload

    def _cycle(self):
        prefix = self.cfg.get('prefix', '')
        for name in DATA_FILES:
            path = os.path.join(_DIR, name)
            if not os.path.exists(path):
                continue
            st = os.stat(path)
            sig = (st.st_size, st.st_mtime)
            if self._seen.get(path) == sig:
                continue
            try:
                with open(path, 'rb') as f:
                    payload = f.read()
                status = _put(self.cfg, prefix + name, payload)
                if 200 <= status < 300:
                    self._seen[path] = sig
                    STATUS['uploads'][name] = {'ts': time.time(),
                                               'bytes': st.st_size}
                    STATUS['errors'].pop(name, None)
                    print(f'[backup] {name}: {st.st_size} bytes uploaded')
                else:
                    STATUS['errors'][name] = f'HTTP {status}'
                    print(f'[backup] {name}: HTTP {status}')
            except Exception as e:                      # log, never crash
                STATUS['errors'][name] = str(e)[:200]
                print(f'[backup] {name} failed: {e}')
        STATUS['last_cycle'] = time.time()

    def run(self):
        interval = self.cfg.get('interval_s', 3600)
        # first cycle shortly after boot so a fresh deploy uploads early
        time.sleep(60)
        while True:
            self._cycle()
            time.sleep(interval)


def start_from_config(cfg_module):
    """Called by server.py at boot. Silently a no-op without config."""
    cfg = getattr(cfg_module, 'BACKUP_S3', None)
    if not cfg:
        return None
    STATUS['enabled'] = True
    t = BackupThread(cfg)
    t.start()
    print(f"[backup] enabled: {cfg['endpoint']}/{cfg['bucket']} "
          f"every {cfg.get('interval_s', 3600)}s")
    return t


if __name__ == '__main__':
    import sys
    if '--test' not in sys.argv:
        raise SystemExit(__doc__)
    sys.path.insert(0, _DIR)
    import site_config as cfg          # the live (gitignored) config
    c = cfg.BACKUP_S3
    probe = f'backup-sync probe {datetime.datetime.now(datetime.timezone.utc).isoformat()}'.encode()
    key = c.get('prefix', '') + 'probe.txt'
    status = _put(c, key, probe)
    print(f'PUT {key}: HTTP {status} - ' +
          ('OK, bucket reachable and writable' if 200 <= status < 300
           else 'FAILED'))
