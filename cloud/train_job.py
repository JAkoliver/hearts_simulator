"""Cloud training job runner (TRAIN IN CLOUD, EVALUATE LOCALLY -
experiment_rules.md #14; built 2026-07-24 targeting Phase 2 as first use).

Runs INSIDE a job directory containing manifest.json:
  {
    "job_id":  "phase2-distill-001",
    "cmd":     ["python", "distill.py", "--data", "targets/*.bin", ...],
    "inputs":  {"relative/path": "sha256", ...},   # verified before run
    "outputs": ["hearts_model_candidate.pth", ...] # hashed after run
  }

Protocol: verify every input hash -> run cmd (cwd = job dir, streamed to
run.log) -> hash outputs -> write result.json {job_id, returncode,
duration_s, output_hashes}. The launcher syncs the job dir up, invokes
this runner, and syncs outputs + result.json home. A checkpoint produced
here decides NOTHING until evaluated locally against a locally-run
baseline.

All logic under the __main__ guard per experiment_rules.md #9.
"""
import hashlib
import json
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open('manifest.json') as f:
        man = json.load(f)

    for rel, want in man.get('inputs', {}).items():
        got = sha256(rel)
        if got != want:
            print(f"INPUT HASH MISMATCH: {rel} want {want[:12]} got {got[:12]}")
            raise SystemExit(3)
    print(f"[{man['job_id']}] {len(man.get('inputs', {}))} inputs verified")

    t0 = time.time()
    with open('run.log', 'w') as log:
        rc = subprocess.call(man['cmd'], stdout=log, stderr=subprocess.STDOUT)
    duration = time.time() - t0

    result = {'job_id': man['job_id'], 'returncode': rc,
              'duration_s': round(duration, 1), 'output_hashes': {}}
    if rc == 0:
        for out in man.get('outputs', []):
            result['output_hashes'][out] = sha256(out)
    with open('result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[{man['job_id']}] rc={rc} in {duration:.0f}s; "
          f"{len(result['output_hashes'])} outputs hashed")
    raise SystemExit(0 if rc == 0 else 4)


if __name__ == '__main__':
    main()
