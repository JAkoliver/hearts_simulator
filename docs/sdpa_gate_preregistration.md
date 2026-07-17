# SDPA promotion gate — pre-registration

Registered 2026-07-17 ~00:40 local, while gate run 2 was in progress and
unobserved. Committed to git before the gate completes so the criterion and
the response to each outcome are in durable history ahead of the result.

## Context

`hearts_net.py` V5Block currently carries an UNCOMMITTED change replacing the
explicit softmax(qk/sqrt(dk))v attention with `F.scaled_dot_product_attention`
(same math, one fused kernel). Measured on 50 deals at production settings:
1.49x end-to-end generation speedup (489 s -> 328 s, seed 4242), forward
45.5 -> 30.7 ms per 2048-row launch. fp32 parity to ~1e-5; bf16 argmax
deviates from fp32 ground truth LESS than the current production path does
(0.745% vs 1.038%, all disagreements near-ties, median fp32 logit gap 0.004).
Production traces are NOT yet re-exported; the benchmark used a scratch trace.

## Criterion (fixed before run 2 reports; no peeking, no re-rolls)

Data: pool all 3,600 per-deal paired deltas from both gate runs.
- Run 1 (observed before registration): 600 deals, seed 20260716 —
  mean +0.470, SE 0.3125. Counts against SDPA in the pool.
- Run 2 (unobserved at registration): 3,000 deals, seed 555001
  (parity2_sdpa.csv / parity2_old.csv in the session scratchpad).

Delta per deal = SDPA_diff − old_diff (positive = SDPA worse). Both runs:
paired same-seed deals, K=32, --pass-search, neutral v3-m7 anchor, CUDA bf16.

PASS iff BOTH:
1. pooled mean < +0.20 pts/deal, AND
2. pooled mean + 1.645 x SE(pooled) < +0.30
   (one-sided alpha=0.05 against the pipeline's raw_guard_threshold margin —
   the codebase's standing definition of "not meaningfully worse").

FAIL otherwise.

## Pre-decided actions — execute the matching branch mechanically

### On PASS (promote SDPA)

1. Append the observed run-2 and pooled numbers + verdict PASS to this file.
2. Run `python export.py` (re-traces BOTH production .pt files from
   `hearts_model_final.pth`; weights unchanged, attention now SDPA).
   The v4m10 legacy traces are untouched.
3. Post-export sanity: load `hearts_ai_search.pt`, compare fp32 outputs vs
   eager `net_from_checkpoint` on random batches — max |diff| must be < 1e-4
   and argmax agreement 100%.
4. Smoke: 10-deal SelfPlayGen run with the new trace at production settings
   (exit 0, plausible record count, no server errors).
5. Commit `hearts_net.py` + this file's appended verdict in one commit.
6. Update auto-memory (true v5 teacher rate becomes the SDPA-config number;
   note steady-state ~6.4 s/deal is a projection until a long run measures it).
7. Follow-up (not part of this gate): a 400-deal steady-state run to replace
   the 6.4 s/deal projection with a measurement.

### On FAIL (reject SDPA)

1. Append the observed run-2 and pooled numbers + verdict FAIL to this file.
2. `git checkout -- hearts_net.py` (the SDPA change is uncommitted; revert
   restores explicit attention exactly). Do NOT run export.py — production
   traces never embodied SDPA and need no restore.
3. Commit this file's appended verdict (the only repo change).
4. Investigation order, first checks (before any new gate attempt, which
   would require a fresh pre-registration):
   a. Force SDPA's math backend (torch.nn.attention.sdpa_kernel MATH) and
      re-check bf16 argmax deviation — isolates flash-kernel numerics from
      the op swap itself.
   b. Compare run-2 delta distribution vs run-1: consistent location shift
      (real effect) vs heavy-tail deal noise (gate underpowered).
   c. Check whether disagreeing decisions concentrate in the passing phase
      (pass-search evaluates C(13,3) combos where near-tie flips could bias
      pass selection systematically rather than symmetrically).

## RESULT (appended 2026-07-17 after run 2 completed)

- Run 1 (n=600, seed 20260716): delta mean +0.4700
- Run 2 (n=3000, seed 555001): delta mean -0.0123
  (sdpa vs anchor -0.0447, old vs anchor -0.0323)
- POOLED n=3600: mean +0.0681, SE 0.1326, one-sided 95% UB +0.2862
- Criterion 1 (mean < +0.20): MET. Criterion 2 (UB < +0.30): MET.

**VERDICT: PASS** — executing the PASS branch. Deviations from plan (full
detail in docs/gate_result_report.md): run 2's two sides ran sequentially,
not concurrently (launcher shell bug killed the reference side at start;
outcomes are seed-deterministic so this is timing-only), and diagnosing that
failure exposed the SDPA side's marginal mean before the reference side ran
(no remaining degrees of freedom: criterion was committed and the reference
data was seed-fixed).
