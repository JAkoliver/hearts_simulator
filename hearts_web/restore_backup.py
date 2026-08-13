"""Restore/verify the off-host backup (the R2 drill, repeatable).

Downloads every backed-up data file from the S3-compatible bucket in
site_config.BACKUP_S3 into a directory, printing sizes and sha256
prefixes. Run from any machine holding the config - the point of the
drill is proving recovery WITHOUT the origin host.

Usage:
    python hearts_web/restore_backup.py <dest_dir>

First run 2026-08-12: all files byte-identical to the origin's live
copies. A backup that has never been restored is a hope, not a backup.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backup_sync as bs
import site_config as cfg


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dest = sys.argv[1]
    os.makedirs(dest, exist_ok=True)
    c = cfg.BACKUP_S3
    if not c:
        raise SystemExit('BACKUP_S3 is not configured in site_config.py')
    print(f"restoring from {c['endpoint']}/{c['bucket']}")
    failures = 0
    for name in bs.DATA_FILES:
        url = (f"{c['endpoint'].rstrip('/')}/{c['bucket']}/"
               f"{c.get('prefix', '')}{name}")
        try:
            with bs._request('GET', url, c.get('region', 'auto'),
                             c['access_key'], c['secret_key']) as r:
                data = r.read()
        except Exception as e:
            missing = getattr(e, 'code', None) == 404
            print(f"  {name}: {'absent in bucket (never existed at origin?)' if missing else f'FAILED {e}'}")
            failures += 0 if missing else 1
            continue
        with open(os.path.join(dest, name), 'wb') as f:
            f.write(data)
        print(f'  {name}: {len(data)} bytes  '
              f'sha256 {hashlib.sha256(data).hexdigest()[:16]}')
    print('RESTORE ' + ('OK' if failures == 0 else f'{failures} FAILURES'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
