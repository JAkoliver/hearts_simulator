# Generation speed ledger — measured figures only

Settings for every entry: v5 teacher trace, K=64, pass-k 24, 14 threads,
CUDA bf16, single process. Two measurement conventions, never mixed:
- **A/B**: 50 deals, seed 4242 (startup-inflated; good for controlled diffs)
- **Steady**: long-run average (seed 777 for 400-deal runs; the honest
  planning number)

| Config (chronological) | A/B 50-deal | Steady (long-run avg) |
|---|---|---|
| Power-of-2 buckets, per-launch autocast (commit 3422383) | 693 s = 13.86 s/deal | 12.20 s/deal (200-deal, seed 4242) |
| + finer buckets + persistent autocast (d790ef8) | 489 s = 9.78 s/deal | 9.56 s/deal (400-deal, seed 777) |
| **+ SDPA fused attention (24bb45d) — current production** | **328 s = 6.56 s/deal** | **6.32 s/deal (400-deal, seed 777, 2026-07-17)** |

Current-production steady detail: 2527 s / 400 deals; per-100-deal bins
7.03 / 6.45 / 6.02 / 5.77 (usual startup skew, no decay); 76,122 launches;
record count 24,424 — identical to the pre-SDPA seed-777 run, i.e. the
teacher's play is unchanged at this seed.

**Measured steady speedups (same-convention ratios):**
- vs previous steady 9.56: **1.51×**
- vs original-config steady 12.20: **1.93×**
- The often-quoted 2.11× (693→328 s) is the A/B convention vs the 13.86
  s/deal mark; dividing steady 6.32 into A/B 13.86 would mix conventions
  and overstate the gain — do not do it.

**Projection retired:** the ~6.4 s/deal projection (9.56 × 328/489) measured
in at 6.32 — confirmed within 1.5%. From here on, only these measured
figures are quoted.

**Planning numbers (steady 6.32 s/deal):**
- 3,500-deal generation: ~6.1 h — 2,500 deals: ~4.4 h
- v4m10 teacher for reference: 0.34 s/deal (100-deal run, 2026-07-16)

This ledger is the local baseline for all H100 cost/speed comparisons
(cloud/REQUIREMENTS.md R5).
