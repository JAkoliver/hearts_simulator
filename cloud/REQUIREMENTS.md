# C1 requirements: cloud generation pipeline (H100 shard-runner)

Scope: only the GENERATE step of expert_iter moves to the cloud. Distill,
gates, promotion, and export stay local and unchanged.

## R1. Cross-hardware equivalence gate (pre-registered, statistical)

Cross-hardware runs are NOT bitwise-identical (kernel selection, accumulation
order differ per GPU/compiler). Equivalence is therefore statistical, judged
by the same style of gate as the July 2026 SDPA acceptance test:

- Before the first cloud-generated data is used for a real iteration, run a
  pre-registered paired gate: search-vs-anchor on N paired deals with the
  cloud-built inference stack vs the same deals with the local stack
  (identical seeds, K=32, neutral v3-m7 anchor).
- PASS iff (a) pooled mean delta < +0.20 pts/deal AND (b) mean + 1.645*SE
  < +0.30 (the pipeline's raw_guard_threshold - its standing definition of
  "not meaningfully worse"). One-sided, alpha = 0.05. Criterion is written
  down BEFORE the runs execute; no post hoc judging, no re-rolls.
- Additionally, record-level sanity on every pulled shard: correct record
  size/count, all recorded actions legal under the recorded mask, per-deal
  rewards zero-mean, seat counts balanced (reuse the selfplay_gen
  validation checks).
- A distill trained on cloud data must pass the normal local promotion gate
  before promotion - which it does anyway; this is the final backstop.

## R2. Pull-based dynamic work queue

Purpose: fault tolerance and preemption recovery, NOT load balancing.

- The work unit is a chunk (deal count + base seed + output shard name),
  same granularity as the local 250-deal chunks.
- Workers PULL the next chunk from a queue; a chunk is only marked done
  when its shard passes the record-level sanity check and is durably
  uploaded. Lease-based: a chunk leased to a worker that goes silent past
  its lease TTL returns to the queue.
- A retried chunk runs with a fresh derived seed (seed + retry offset,
  mirroring the local per-chunk retry) so a poisoned seed cannot wedge the
  queue, and the failed attempt's partial shard is deleted before retry.

## R3. Spot/preemptible instance tolerance

- Preemption must lose at most one chunk of work per worker (the leased
  chunk), never the iteration. Completed shards live in durable storage
  (object store or the orchestrator's disk), not on the instance.
- Resume = start a fresh instance pointing at the same queue; no state
  handoff. The orchestrator treats instance death and chunk failure
  identically (lease expiry).
- The orchestrator itself runs locally (or on one small non-spot node) and
  is the only stateful component.

## R4. Shard transfer must not become the bottleneck (verified: it cannot)

Measured record economics: ~62 records/deal x 818 B = ~51 KB/deal.
- 3,500-deal iteration: ~180 MB total transfer.
- 20,000-deal iteration: ~1 GB total.
At even 100 Mbit/s home downlink, 1 GB pulls in ~90 s - two orders of
magnitude below generation time. Shards stream back as chunks complete
(overlapped with generation), so transfer adds ~zero wall-clock. No
compression or delta scheme needed; gzip (~2-3x on these records) is a
nice-to-have only.

## R5. Cost-per-iteration reporting

Every cloud run reports, alongside wall-clock speedup vs local:
- $/iteration (instance-hours x rate, incl. orchestrator overhead time)
- $/1k deals, and deals/$ vs the local 4090's electricity-only baseline
- these numbers land in the experiment ledger next to the run they paid for,
  so the speed-vs-cost trade is always visible when choosing where to run.

## R6. Build/runtime parity

- Linux container: same libtorch major.minor (2.12 cu126), same engine
  commit, same trace files (bit-identical .pt shipped into the image or
  pulled from the orchestrator at start).
- AOTInductor-compiled inference is a Linux-side optimization candidate
  AFTER the plain-trace cloud gate passes (R1 first, then optimize and
  re-gate the optimized stack the same way).
