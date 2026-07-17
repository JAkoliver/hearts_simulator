# H100 validation session runbook (RunPod, ~4 h, abort-early)

Pre-registered before the session; seeds fixed here so nothing is chosen
after results are visible. Local baseline: steady 6.32 s/deal
(docs/speed_ledger.md). Stop after step 4 + cost table - nothing else.

## User action (the only manual part)

RunPod -> Deploy Pod -> **H100 80GB, On-Demand (Secure Cloud)**:
- Image: `nvidia/cuda:12.6.2-base-ubuntu22.04`
- Disk: 40 GB volume mounted at /workspace
- Enable SSH (public key or RunPod's generated key)
- Paste the SSH connection line back to the session driver. Note the $/hr.

## Files pushed to /workspace after SSH is up (driver does this)

- `hearts_src.tar` - `git archive HEAD` of the validated commit
- `hearts_ai_search.pt` (v5 SDPA production trace)
- `hearts_ai_grandmaster_v3_milestone7.pt` (neutral anchor, from legacy_v3_pass238/)

## Steps (each gated; abort and stop the pod on failure)

1. **Setup + smoke** (`bash cloud/h100_session/instance_setup.sh`, then a
   4-deal `--cuda --bf16` generation): binary runs, GPU visible, shard
   passes `shard_check.py`. Budget ~30 min incl. downloads.
2. **Throughput**: 50-deal A/B at seed 4242 for threads {14, 26, 52} (pick
   winner), then one **300-deal steady run at seed 777** with the winning
   thread count - directly comparable to the local ledger rows. Record
   s/deal + $/1k deals. Budget ~1 h.
3. **R1 equivalence gate**: SearchEval on-instance vs the same run locally,
   **3,000 paired deals, seed 20260718** (fresh, pre-registered here),
   K=32, --pass-search, neutral anchor, both sides concurrent (instance +
   home 4090). Judge ONLY with `cloud/xhw_gate.py` (mean < +0.20 AND
   one-sided 95% UB < +0.30). FAIL = cloud data feeds nothing; stop.
   Budget ~1 h.
4. **Queue over tunnel**: from local, `ssh -R 8757:localhost:8757 <pod>`;
   local orchestrator serves one 250-deal chunk; on-instance `worker.py`
   leases, generates, uploads; shard validates locally. Budget ~30 min.

## Deliverable

docs/speed_ledger.md gains the H100 row(s): measured s/deal, $/1k deals,
$/3,500-deal iteration, vs local 6.32 - plus go/no-go on spot fleets.
Then the pod is STOPPED and the session ends.
