"""Cloud training job launcher (local side of cloud/train_job.py).

Assembles a self-contained job directory (inputs copied, hashes recorded
in manifest.json), then either:
  --local     executes the runner right here (dry-run / audit path), or
  --pod HOST  prints the exact rsync/ssh commands for a rented pod
              (execution against a pod stays manual until a rental is
              approved - experiment_rules.md #13).

Usage (example, Phase 2 distill):
  python cloud/launch_train_job.py --job-id phase2-distill-001 \
      --input hearts_model_final.pth --input distill.py --input hearts_net.py \
      --input headroom.py --input "targets/*.bin" \
      --output hearts_model_candidate.pth \
      --cmd python distill.py --data "targets/*.bin" --init hearts_model_final.pth \
            --out hearts_model_candidate.pth --epochs 3 \
      [--local]

Standing rule: the returned checkpoint is EVALUATED LOCALLY against a
locally-run baseline before any promotion decision.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cloud.train_job import sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--input', action='append', default=[],
                    help='file or glob, relative to repo root; copied into the job dir')
    ap.add_argument('--output', action='append', default=[])
    ap.add_argument('--cmd', nargs=argparse.REMAINDER, required=True)
    ap.add_argument('--jobs-root', default='cloud_jobs')
    ap.add_argument('--local', action='store_true',
                    help='execute the runner locally (dry-run / audit)')
    ap.add_argument('--pod', default=None,
                    help='user@host of an approved pod; prints sync+run commands')
    args = ap.parse_args()

    job_dir = os.path.join(args.jobs_root, args.job_id)
    if os.path.exists(job_dir):
        raise SystemExit(f"job dir {job_dir} already exists - job ids are single-use")
    os.makedirs(job_dir)

    inputs = {}
    for pattern in args.input:
        files = glob.glob(pattern) or [pattern]
        for f in files:
            if not os.path.isfile(f):
                raise SystemExit(f"input not found: {f}")
            dest = os.path.join(job_dir, f)
            os.makedirs(os.path.dirname(dest) or job_dir, exist_ok=True)
            shutil.copy2(f, dest)
            inputs[f.replace('\\', '/')] = sha256(f)

    manifest = {'job_id': args.job_id, 'cmd': args.cmd,
                'inputs': inputs, 'outputs': args.output}
    with open(os.path.join(job_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    shutil.copy2(os.path.join('cloud', 'train_job.py'),
                 os.path.join(job_dir, 'train_job.py'))
    print(f"job assembled: {job_dir} ({len(inputs)} inputs)")

    if args.local:
        rc = subprocess.call([sys.executable, 'train_job.py'], cwd=job_dir)
        print(f"local run rc={rc}; results in {job_dir}/result.json")
        raise SystemExit(rc)
    if args.pod:
        print("\nRun these against the approved pod:")
        print(f"  rsync -az {job_dir}/ {args.pod}:/workspace/{args.job_id}/")
        print(f"  ssh {args.pod} 'cd /workspace/{args.job_id} && python train_job.py'")
        print(f"  rsync -az {args.pod}:/workspace/{args.job_id}/"
              f"{{result.json,run.log,{','.join(args.output)}}} {job_dir}/")
        print("Then: verify result.json hashes match the fetched files, and "
              "evaluate the checkpoint LOCALLY before any promotion decision.")


if __name__ == '__main__':
    main()
