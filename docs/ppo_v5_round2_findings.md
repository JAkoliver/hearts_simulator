# PPO-on-v5 round 2: findings and next steps (2026-07-20)

Campaign: 4 run_loop trials on the v5-M PPO-promoted baseline
(hearts_model_final.pth, 2026-07-16), every verdict from the re-powered
search gate (n=2400, K=32, alpha=0.05, neutral v3-m7 opponents, sharded
4-way). Raw numbers live in docs/speed_ledger.md ("PPO-on-v5 round 2").

## Result

All four trials FAILED the search gate. Pooled search delta across
9,600 paired deals: **-0.078, SE 0.081, 95% CI [-0.24, +0.08]** —
indistinguishable from zero, and the CI excludes every effect size this
project has ever considered promotion-worthy.

Meanwhile three of four trials produced **large, highly significant raw
gains** (-0.626 to -0.775, |t| > 4.5) against the baseline in
head-to-head raw play.

## What this establishes

1. **PPO reliably improves raw head-to-head play from this baseline**
   (~0.6-0.8 pts/deal per trial) across distinct hyperparameter
   mutations. That signal is real and repeatable.
2. **None of it survives search.** The searched player (the deployed,
   gate-relevant configuration) is flat: four independent trials, four
   nulls, tight pooled CI. Raw-gain size does not even correlate with
   search delta (trial 3: strongest raw at -0.775, weakest search lean
   at -0.062).
3. This is the search-extraction-ceiling pattern already documented from
   the amplification experiments (K-escalation, ISMCTS): full-rollout
   search already computes at play-time much of what PPO teaches the
   raw policy. The gate is not "unlucky" — it is correctly refusing
   noise-sized searched effects.

## Two confounds worth naming honestly

- **The raw guard is not a neutral measurement.** It plays candidate vs
  the exact baseline the candidate trained against (opponent pool
  includes baseline snapshots). Some or all of the -0.6..-0.8 raw gain
  may be opponent exploitation rather than general strength. Untested
  either way; cheap to test (see A below). The same trap once flipped a
  search comparison by +1.4 pts (neutral-opponent rule, 2026-07-15).
- **The one historical PPO promotion (2026-07-15, -0.712 at n=600,
  p=0.016) predates the powered gate.** Under the old gate's ~25% power
  and -0.51 bar, a true effect far smaller than -0.712 could have
  produced that verdict. It may have been real, it may have been the
  same flat effect we now measure four times in a row. Retestable for
  free (see B below).

## Where this leaves the program

Both cheap strength levers are now measured flat from the current
baseline:
- Same-teacher expert iteration (3.5k-deal recipe): +0.45..+0.68,
  actively worse (cloud-iter-0, re-gated at n=2400).
- PPO mutation search: pooled -0.078 +/- 0.081, null (this campaign).

The network remains the binding constraint, and the recipe that produced
the largest verified jump was an architecture change (v5), not an
optimization loop on a fixed architecture.

## Next steps, ranked

**A. Neutral-opponent raw evaluation of a strong-raw PPO candidate
(~1 h local, free).** Retrain with trial 3's config (~45 min; its
candidate was overwritten by trial 4's in hearts_model_last_rejected.pth),
then evaluate raw candidate-vs-baseline seats against neutral v3-m7
tables. Decides whether PPO's raw gains are genuine strength or
opponent exploitation.
- Real -> a **raw-line promotion track** becomes attractive: the stated
  goal is a raw net at best-human level, and we are currently discarding
  ~0.7 pts/trial of exactly that currency because the promoter only
  rewards searched strength. Design: promote on a powered neutral raw
  gate, keep the search gate as a non-regression guard.
- Exploitation -> PPO-on-v5 is closed in its current form; strike the
  raw-line idea.

**B. Re-gate the 2026-07-15 PPO promotion at n=2400 (~82 min local,
free).** Hall_of_Fame/hearts_model_milestone_1784156801.pth (post-PPO)
vs milestone_1784120250.pth (pre-PPO parent) through the powered gate.
Tells us whether PPO ever produced a real searched gain on v5. If it
did not, "PPO-on-v5 works (searched)" gets reclassified as a
weak-gate artifact and the closed-directions list gets simpler.

**C. Architecture/scale: v5-L distillation (overnight local, free).**
Distill a larger card-token transformer (d=384-448, L=8) from the
2.93M banked records — the exact recipe that produced the v5
breakthrough. Architecture, not size, was the lever last time; there is
no evidence the ceiling has been reached on that axis.

**D. Expert iteration done properly (cloud, needs budget approval).**
>= 12k deals per generation, cumulative pooled data across generations,
possibly tree-search visit-count targets instead of flat-search actions.
~$22-25 per generation on one H100, less wall time on four. Only worth
proposing after A/B/C inform the picture.

Recommended order: **B tonight (fully unattended, uses existing
artifacts), A next session, then decide between C and the raw-line based
on what A and B say.** Nothing here spends cloud money.

---

# Diagnostics A and B: results (2026-07-21)

**B - re-gate of the 2026-07-15 PPO promotion at n=2400
(regate_ppo_promotion.py, pre-PPO milestone 1784120250 as candidate vs
the post-PPO baseline trace, neutral opponents):**

    pre-PPO delta +1.025 (SE 0.180, n=2400, t=5.68)

The promotion was REAL and the weak gate UNDERestimated it (-0.712
claimed at n=600; true searched gap ~1.0). PPO-on-v5 demonstrably can
move searched strength; round 2's four nulls mean the searched gains
available from the current baseline were already harvested, not that PPO
never worked.

**A - neutral-opponent raw evaluation (neutral_raw_eval.py: candidate
and baseline each seated vs 3x v3-m7 anchors on identical deals,
n=2500 paired):**

| Candidate | Head-to-head raw (vs own baseline) | Neutral raw delta |
|---|---|---|
| trial-3 config repro (cand_A_trial3repro.pth) | -0.775 (orig trial) | **-0.654 (SE 0.143, t=-4.57, p<1e-5)** |
| trial-4 rejected (hearts_model_last_rejected.pth) | -0.187 (ns) | **-0.636 (SE 0.144, t=-4.43, p<1e-5)** |

The raw gains are GENUINE strength, not opponent exploitation - they
fully survive against neutral opponents the candidates never trained
against. (Note trial 4: head-to-head raw understated its true neutral
gain by 3x - the head-to-head raw guard is a noisy, biased measure in
both directions.)

## Combined interpretation

PPO keeps producing genuinely stronger RAW nets (~-0.65/deal per trial,
robust across mutations); full-rollout search then flattens the
difference because it already computes at play time most of what PPO
internalizes. The project goal is a raw net at best-human level - the
current promoter measures the one axis (searched strength) that is
saturated and discards the axis (raw strength) that is both improving
and goal-relevant.

## Raw-line promotion: RECOMMENDED, with guardrails

1. **Promoter**: neutral raw gate (neutral_raw_eval measurement),
   n=2500, alpha=0.05 -> promotion bar ~-0.24; measured effects ~-0.65
   give >99% power. Runs in ~2 min (vs 82 for the search gate).
2. **Search non-regression guard**: keep the n=2400 search gate but as a
   REJECT-IF-WORSE check (fail only if the one-sided 95% bound shows
   searched strength degraded beyond +0.3). The deployed search player
   must never get worse.
3. **Anchor-overfit watch**: promoting repeatedly against a fixed v3-m7
   anchor invites anchor exploitation over generations. Mitigate by
   adding a second anchor family (v4-m10) to the neutral table mix
   and/or periodic user calibration matches.
4. **Compounding check**: the point of the raw line is STACKING gains.
   After each promotion, the next trial's neutral delta is measured
   against the new baseline; if raw gains stop compounding after 1-2
   promotions, the raw axis has its own ceiling and we reassess.
