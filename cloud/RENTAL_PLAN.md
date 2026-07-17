# H100 validation rental plan — awaiting explicit go-ahead (NO SPEND YET)

Everything below this line costs money only after approval of a specific
rental. All software is committed and locally verified (ce0af4a), including
GPU generation inside the Linux container on the 4090 via WSL2.

## Instance

- **1× H100 80GB (PCIe or SXM), on-demand — NOT spot — for this validation
  session** (spot preemption noise would pollute first measurements; spot
  is for production fleets after validation).
- Provider: RunPod Secure Cloud / Lambda / equivalent. On-demand H100
  pricing at last check: ~$2.20–2.80/hr (verify at rental time).
- Needs: docker with NVIDIA runtime, 30+ GB disk for the 9.6 GB image,
  outbound SSH (for the reverse tunnel to the local orchestrator).
- Image delivery: `docker save hearts-worker | gzip` (~4–5 GB) pushed via
  scp, or rebuilt on-instance from the repo (build is ~15 min; the
  Dockerfile pins everything — R6 parity holds either way).

## Estimated hours and cost

| Step | Instance time |
|---|---|
| 1. Boot, image load, GPU-visible smoke (4-deal generation) | ~0.5 h |
| 2. Throughput: 50-deal A/B + 300-deal steady at threads {14, 26, 52} | ~1.0 h |
| 3. Cross-hardware equivalence gate (xhw_gate.py, 3000 paired deals; H100 side on-instance, local side runs concurrently at home) | ~1.0 h |
| 4. Queue integration: worker pulls one 250-deal chunk from the local orchestrator over `ssh -R 8757:localhost:8757`, shard lands + validates | ~0.5 h |
| Buffer | ~1.0 h |
| **Total** | **~4 h → ~$9–12** |

## What gets validated, in order (abort early if any step fails)

1. **It runs**: container + driver + trace sha-verified from orchestrator.
2. **Measured H100 s/deal** at production settings across thread counts —
   the honest speed number, recorded next to the local baseline of
   **6.32 s/deal** (docs/speed_ledger.md; steady convention only).
3. **R1 equivalence gate PASSES** (criterion pre-committed in xhw_gate.py:
   pooled mean < +0.20, one-sided 95% UB < +0.30). FAIL = cloud data does
   not feed any real iteration; investigate before any further spend.
4. **The queue works over the tunnel** — lease, generate, upload, validate.

## Deliverable after the session

A cost/speed table in docs/speed_ledger.md (R5): H100 s/deal, $/1k deals,
$/3,500-deal iteration vs local 6.32 s/deal — and a go/no-go on production
fleets (N× spot instances with the lease queue absorbing preemptions).

## Explicitly deferred (post-validation, separate approvals)

- Spot-fleet production runs; AOTInductor-compiled Linux inference (R6:
  optimize AFTER the plain-trace gate passes, then re-gate the optimized
  stack); expert_iter --remote wiring (C3).
