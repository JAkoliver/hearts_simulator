> *Published as registered (Study 2 of the Perilune v5 vs. xinxin report, docs/xinxin_eval_report.md). The approval signature sentence and the author email address have been replaced by a registration date, and the in-text references to signing now read "registered"; the registered text is otherwise unchanged (md5 of the original file: a4e7c507b59e47ce07a942d5e0ec0c68). The title carries the working name "DRAFT" the document had at registration; "the user" throughout is the author, who directed and approved the study the AI drafted.*

# PRE-REGISTRATION 2 (DRAFT v3) — Study 2: the deployed raw network
# vs. xinxin across simulation budgets

Status: DRAFT v3 (pivoted 2026-08-15 for proper powering: the study is
now RAW-ONLY on the Perilune side — one properly powered confirmatory
cell plus a xinxin-budget frontier. The v2 design's K-ladder,
EQUAL-TIME rung, and P0 rung are DROPPED and return to Future Work;
rationale in §1. Energy measurement DROPPED entirely, same date: cost
claims are wall-clock-only on this machine; §4.) Blanks marked `[CAL]` are filled from the cost
calibration, which gates no design choice. Then the design is registered and
becomes binding. House rules apply (halt-default, one
pre-unblinding amendment, registered analysis frozen before data;
crash-forced repairs follow the Study-1 precedent).

Relationship to Study 1 (xinxin_eval_prereg_study1.md, registered 2026-08-14): a
SEPARATE study motivated by Study 1's results — its raw-vs-10k cell is
the weakest important cell (n = 8, least Perilune-favorable point
estimates: placement +0.19, wins −0.75) while carrying the most
product-relevant question. Stated openly here and in the combined
report. Nothing from Study 1 is pooled into Study 2 estimates;
cross-study comparisons are labeled exploratory.

**STOP CLAUSE (binding at registration):** the combined report (Study 1 +
Study 2) is published after Study 2's registered analysis completes,
REGARDLESS of outcome — explicitly including a decisive result AGAINST
the raw network. All remaining open questions (the Perilune-side
cost–strength ladder and equal-time cell, M-C′, M-B′ vs. his standalone
heuristic players, moon microscopy tiers, opponent-model sensitivity,
other opponents) stay in Future Work. No "one more experiment".

## 1. Question, claims, and why the design is raw-only

**Q1 (confirmatory, the study's only claim-carrying cell):** is the
DEPLOYED PRODUCT — the raw policy net, no search, ~2 ms/decision (as of
2026-08-15, champion lineage 8a89da90) — not meaningfully worse than,
or stronger than, xinxin at his shipped default (sims = 10,000)?

**Q2 (estimation):** the frontier of that fixed product against
xinxin's simulation budget x ∈ {300, 1k, 3k, 10k, 30k, 93k} — pricing
the raw net in sims, with the crossing of D(x) = 0 re-estimated at
adequate n (Study 1's version ran at n = 8/level and was right-censored
with a 45%/55% bootstrap split).

**Why raw-only (the pivot):** powering Q1's non-inferiority conclusion
properly at the SD's upper confidence limit requires n ≈ 438 pairs —
affordable ONLY because the raw cell is the cheapest configuration in
the design space (its match cost is almost entirely xinxin's).
Restricting Study 2 to the raw side buys full statistical rigor on the
product question for less GPU time than the v2 design spent on
machinery (equal-time selection, bundle calibration) that served an
engine-design question, not the product question. The dropped questions
are not disavowed; they are Future Work under the stop clause.

Engines: the SAME champion pair as Study 1 (hearts_ai_search_match.pt
md5 3a2abd36…, hearts_equity.pt md5 efdfee07…, lineage 8a89da90) — md5s
re-verified at registration AND immediately before counted runs; if a new
champion is promoted meanwhile, this study still runs on THIS pair.
Perilune configuration: `--peri-raw` with `--device cpu` ONLY (raw
sequential play AND raw sequential passing; no search anywhere; CPU
inference). **Device note:** the live site serves the raw net on CPU,
so Study 2 runs it on CPU for deployment fidelity — verified clean
(smoke 2026-08-15: 0 FATALs, verifier 0 errors, ~4–7 ms/play decision).
This also makes the confirmatory cost comparison SAME-SILICON (both
engines on the same CPU), so the Study-1 §3.7 hardware-class
exchange-rate caveat does not apply to this cell. Study 1's raw arms
(M-B, sweep) ran the raw net on GPU; strength is taken as
device-independent (same weights, argmax decision rule; CPU/GPU float
rounding can flip rare near-ties) and this difference is disclosed
wherever cross-study raw cells are compared. During matches the raw
net's CPU inference shares cores with xinxin's threaded search;
strength is unaffected (his playout count is fixed by sims, our argmax
by weights), and all cost figures come from the isolated calibration
runs, not from contended match time. Opponent: xinxin, git
6ba11548af49d3953f989f50b318ee4f4c1b91f7 with the two recorded local
patches (Study 1 §1), rules mask 0xda1, at sims = x, worlds = 30,
C = 0.4, opponent-model level 2, ε = 0.1, threaded (his shipped
primary-player configuration with only the sims knob varied — x =
10,000 IS that configuration verbatim, and 5,000–50,000 is his
published experimental range, ICGA 2008).

## 2. Design

**Cells.** All cells: 2v2 stratum (driver seating rows 4,5,6,7,8,9 —
the six two-vs-two arrangements of the driver's seating table),
complement pairs, matches to 100 points, Perilune raw vs. xinxin@x.
- CONFIRMATORY: x = 10,000, n_conf = 438 pairs (876 matches).
- ESTIMATION: x ∈ {300, 1,000, 3,000, 30,000, 93,000}, n_est = 48
  pairs each (96 matches each; double Study 1's sweep resolution).

**Seeds.** ONE block, base 45260814 + 7919·i per pair; per-deal seed =
pair seed + 257·deal (Study-1 amendment-3 dealer). Estimation cells use
the block's first 48 pairs; the confirmatory cell uses pairs 0–437 —
nested prefixes, so cross-level contrasts on common pairs are paired by
design (CRN), including against the confirmatory cell. Disjoint from
Study 1's blocks (20260814, 30260814): no integer solutions to the
collision equations within the ±257·20 deal windows (checked). Machine
otherwise idle; one worker per directory; every run keeps its
games.jsonl; verifier must report 0 hard errors per cell or the cell is
void (fix, re-run fresh).

## 3. Registered metrics and inference

Computed by the FROZEN analyze.py + supplement_stats.py per cell; the
frontier script (frozen, §5) adds non-inferiority readouts, the family
correction, the dominance table, and the crossing estimate.

- Per cell: paired placement difference (tie-averaged) and paired
  match-win difference (tie-split), exact Wilcoxon + sign test,
  secondaries with 95% CIs, as in Study 1. Sign convention unchanged
  (negative placement favours Perilune; positive win difference
  favours Perilune).
- **Confirmatory inference (the ONLY claim-carrying tests): the two
  primaries of the x = 10,000 cell, Holm over exactly m = 2.** The
  within-cell Holm printed by the frozen supplement for other cells is
  descriptive; estimation cells carry CIs only, no significance claims.
- **Q1 registered language (both axes, all three outcomes decidable):**
  - Placement: margin δ_p = 0.35 places. "Not meaningfully worse" if
    the 95% CI upper bound < +0.35; "stronger" if the CI excludes 0 in
    Perilune's favour with Holm q < 0.05; "meaningfully worse" if the
    CI lower bound > 0 and the point estimate ≥ δ_p is contained —
    reported exactly as the CI shows, whichever obtains.
  - Wins: margin δ_w = 0.5 wins/pair, same tripartite structure.
    (Study 1's n = 8 point estimate here was −0.75: a decisive
    "meaningfully worse on wins" is a live outcome and will be
    published as such under the stop clause.)
- **Dominance (registered definition):** the raw net DOMINATES
  xinxin@10k if (a) placement non-inferiority holds and (b) its
  measured median wall-clock per play decision is lower (§4). Energy is
  NOT measured in this study (dropped by design decision, 2026-08-15;
  §4). Because both engines run on the same CPU (§1 device note), this
  is a same-silicon time comparison — the hardware-class exchange-rate
  caveat does not apply to it; machine-specificity (this CPU) still
  does and is stated. Claim upgrade if dominance holds: "the deployed
  product matches or beats xinxin's shipped default at a fraction of
  the per-decision time, on the same processor."
- **Crossing (Q2, estimation):** weighted isotonic fit of the paired
  placement difference in log x across all six levels (weights = pair
  counts), crossing of 0 by interpolation, bootstrap over pairs within
  level (10,000 resamples, censoring fractions reported) — the Study-1
  method, now at adequate n. The SAME estimate is computed for the win
  axis (Study 1 never fit it; its point estimates flip sign around
  10k–30k, so the win crossing is the more informative curve).
- **Playout accounting (registered, unit defined):** per-cell realized
  playout counters, Study-1 §3.8 corrected convention: Perilune raw = 0
  by construction; xinxin's realized playouts/decision reported with
  his mixed play+pass denominator disclosed. The frontier is plotted
  against sims, realized playouts, and wall-clock.
- MDE clause: any null reported as its CI plus the realized-power MDE
  (2.80 × realized SE); realized, not projected, values in the writeup.

## 4. Sample sizes, powering, and cost measurement
##    (mechanical; fixed BEFORE any counted run; nothing gated by [CAL])

**n rule — powered for the registered conclusion at the SD's upper
confidence limit (the disciplined version, now affordable):**
- SD_ref = the 80% upper confidence limit of Study 1's paired placement
  SD at this cell: 1.10 from n = 8 → UCL = 1.10 · sqrt(7 / χ²₀.₂₀,₇) =
  1.10 · 1.353 = **1.49**.
- Planning alternative θ_plan = +0.15 (conservative, between 0 and
  Study 1's noisy +0.19; powering at θ = 0 would give ~25% power under
  the study's own best estimate — the v1 error, corrected).
- n_conf = ceil((2.80 · SD_ref / (δ_p − θ_plan))²)
         = ceil((2.80 · 1.49 / 0.20)²) = 436 → multiple of 6 =
  **438 pairs**. Realized power: 80% at the SD UCL by construction,
  ≈ 97% at the SD point estimate. Win axis: SD_w(UCL from 1.49 at
  n = 8… Study 1's win-cell SD) — the same n gives SE_w ≈ 0.07–0.10,
  decisive against the ±0.5 margin either way. No cap; every input is
  fixed at registration.
- n_est = **48 pairs** per estimation level (multiple of 6; doubles
  Study 1's sweep resolution; CI half-widths ≈ 0.17–0.31 places at the
  per-level SDs Study 1 measured).
- No optional stopping; all counts final at registration.

**Cost measurement (wall-clock only; reporting only — gates nothing).**
Energy measurement is DROPPED (design decision, 2026-08-15): WSL2
cannot measure CPU energy (no RAPL/hwmon — checked), and at this cell's
~20× wall-clock gap an energy exchange rate would not change any
conclusion; the study registers no energy claim of any kind.

Wall-clock protocol (CORRECTED pre-registration for implementability,
2026-08-15 — the earlier hold-deal text could not be realized: the
binary emits xinxin's timing only as a per-run aggregate, and
single-deal matches always begin at the Left-pass deal, so hold deals
cannot be isolated per invocation without modifying the frozen binary):
**per-deal invocation design** — each datum is ONE single-deal match on
a uniform table (`--types`), one process invocation per deal, so each
invocation's BUDGET aggregate IS that deal's cost. Every deal is a
Left-pass deal with an IDENTICAL decision structure for both engines
(52 plays + 12 pass picks), so the primary cost metric —
**total decision wall-clock PER DEAL** — has the same denominator on
both sides by construction, which answers the Study-1 §3.8 denominator
critique more directly than the hold-deal design would have.
Secondary figures: Perilune per-play-decision (exact, from his
play/pass BUDGET split) and xinxin per-decision over all decisions
(mixed denominator, disclosed). The DOMINANCE comparison (§3) uses the
per-deal metric. ≥ 32 deals per configuration, first invocation
discarded as warmup, distributions (median, IQR), idle machine.
- Cost table [CAL FILLED 2026-08-15, idle machine, 32 deals/config,
  warmup excluded; per-deal totals are uniform-table sums over all four
  seats' decisions, identical structure both engines]:
  | config | per-deal median [IQR] ms | secondary (per-decision) |
  |---|---|---|
  | raw (CPU) | **452** [444, 463] | 4.4 ms/play [4.3, 4.6] |
  | xin 300 | 314 [307, 326] | 4.9 ms (mixed denom) |
  | xin 1,000 | 496 [486, 518] | 7.8 ms |
  | xin 3,000 | 1,002 [979, 1,043] | 15.6 ms |
  | xin 10,000 | **2,685** [2,643, 2,720] | 42.0 ms |
  | xin 30,000 | 7,341 [7,258, 7,443] | 114.7 ms |
  | xin 93,000 | 21,798 [21,690, 22,054] | 340.6 ms |
  Dominance cost condition (raw < xin@10k per deal): 452 < 2,685 —
  SATISFIED, ratio ≈ 5.9× on the same processor. (Realized
  playouts/decision are reported from the counted cells' BUDGET
  counters at analysis.)

**Time budget (measured bases):** calibration ≈ 1.5 h; confirmatory
cell 876 matches × ~16 s ≈ 4 h; estimation cells ≈ 0.2 + 0.2 + 0.3 +
0.9 + 2.4 h ≈ 4 h; total ≈ **9–10 h GPU** (one night).

## 5. Instruments (frozen at registration)

- perilune_tourney binary: 5f5a905ad947667137213171ae7a287f (unchanged
  from Study 1 amendment 3; `--peri-raw` and `--xin-sims` are existing
  flags — no code changes).
- analyze.py / verify_games.py / supplement_stats.py: unchanged (md5s
  per Study 1 §5 + Addendum A).
- NEW — built, battery-tested, frozen before registration:
  - cost measurement scripts (per-deal invocation design, wall-clock
    only): cost_calibrate2.sh md5 6b977434004e310b0eb7548a55e426a6,
    cost_summary2.py md5 4650bf568ff42ab6520c59f74c969a34
  - frontier analysis script (non-inferiority CIs, Holm m = 2,
    dominance table, weighted isotonic crossing on both axes with
    bootstrap; self-tested in the battery): frontier2.py
    md5 9f0099c29ba2b28dfb62e461d05c36dc
- Pre-flight battery before counted runs: one miniature run per cell
  configuration on throwaway seeds — including the `--device cpu` raw
  configuration (first smoke already green, 2026-08-15) —
  verifier-gated (armcheck pattern), plus the information-isolation
  probe re-run on this binary (Study 1: 12/12 identical under
  hidden-hand permutation; reported in the Study-1 report §2).
- Verifier gates: 0 hard errors per cell or the cell is void; any
  bridge FATAL or gamelog WARNING voids the affected run.
- Champion trace md5s re-verified at registration and immediately before the
  first counted run.

## 6. What results are allowed to mean

- Q1 outcomes, all publishable under the stop clause:
  - RAW dominates (non-inferior + lower measured wall-clock per play
    decision): the combined report's practical-budgets claim upgrades
    to "the deployed product matches or beats xinxin's shipped default
    at a fraction of the per-decision time, on the same processor"
    (energy unmeasured by design; same-silicon comparison per §1, so no
    hardware-class exchange rate is involved; machine-specific to this
    CPU).
  - RAW non-inferior on placement but meaningfully worse on wins (the
    axis Study 1's point estimate leans toward): reported exactly so —
    the product concedes matches to his default while holding
    placement; the win-axis crossing (Q2) then locates where that
    begins.
  - RAW meaningfully worse on placement: the product claim FAILS; the
    combined report's practical-budgets claim rests on the teacher
    configuration (Study 1 M-A) alone and says so.
- Q2 is estimation; crossing estimates carry CIs and censoring
  fractions, no significance claims; the win-axis crossing is expected
  to be the informative one.
- **Coverage shape (stated before a reviewer does):** Study 1 traced
  xinxin's budget axis against fixed raw at n = 8 and Perilune's
  configurations at fixed xinxin@10k only at K = 64; Study 2 re-traces
  the xinxin-budget axis at proper n against the fixed product. The
  two-dimensional budget plane's interior — both budgets varying, and
  every Perilune-search-vs-strong-xinxin cell other than M-A/M-C —
  remains unmeasured; no claim reaches it. The Perilune-side ladder is
  Future Work.
- Study-1 comparisons (including to its n = 8 sweep) are exploratory,
  labeled.
- Nulls are reported as CI + realized MDE, never "no difference"; the
  stop clause applies regardless of outcome.

## 7. Threats to validity (carried into the combined report)

All Study-1 threats apply (single opponent/implementation; single
machine — Ryzen 7 7800X3D 8C/16T + RTX 4090; hardware-class exchange
rate; objective mismatch; opponent-model mismatch both directions —
note his OM-2 models xinxin-like opponents while facing a raw policy
net in every cell of this study; statistical-not-bit-exact
reproduction). Study-2-specific: all cost figures are machine
properties, and the cost comparison is wall-clock only — energy is
unmeasured by design (§4), so cost claims are time claims on this
machine, nothing more; n_conf's planning values (θ_plan = +0.15, SD UCL
1.49) are stated in §4 — powering at the UCL means the study is, if
anything, overpowered at the point estimate, which is the correct
direction of error for a confirmatory product claim; the confirmatory
cell is the one where Perilune plays its cheapest configuration — the
matched-cost engine-design questions this does not answer are Future
Work by design, not omission.

## 8. Execution sequence (halt-default at every gate)

0. Re-verify champion trace md5s. Build + freeze the two new
   instruments; run the pre-flight battery.
1. Cost calibration (idle, hold-deal design) → fill [CAL] (reporting
   only; gates nothing).
2. REGISTRATION. After registration: no changes to frozen instruments
   without a documented amendment (budget: one, plus crash-forced
   repairs per Study-1 precedent).
3. Counted runs, sequential, machine idle, order: estimation cells
   cheap → expensive (300 → 93k, skipping 10k), then the confirmatory
   x = 10,000 cell last (so the longest run benefits from the most
   settled machine). Verifier after every cell.
4. Frozen analysis (analyze.py + supplement_stats.py + frontier
   script). Unblinding only after every cell verifies clean.
5. Roll results into the combined report (Study-1 record frozen as of
   2026-08-15; Study-2 additions layered, provenance separate; the
   Study-1 report's §8 item 1 is superseded by Study 2 and its
   remaining Perilune-side half moves to Future Work). Publish per
   EXPERIMENT_PLAN.md §11 and the STOP CLAUSE — regardless of outcome.

Registered 2026-08-15 on the v3 raw-only design with
energy dropped and Perilune on CPU. Recorded with one pre-freeze
implementability correction disclosed to the user before any counted
run: the §4 cost-measurement protocol changed from the unimplementable
hold-deal design to the per-deal invocation design (same-denominator
per-deal metric; gates nothing — the [CAL] table is reporting only and
no design choice depends on it). Champion traces and all frozen
instruments md5-re-verified at registration (2026-08-15): traces 3a2abd36…/
efdfee07… unchanged; binary 5f5a905a…; analyze.py 200ad385…;
verify_games.py 41a6440f…; supplement_stats.py 00f0f084…. Binding from
this point: halt-default, stop clause in force, no changes to frozen
instruments without a documented amendment.
