# JOURNEY — the project in eras

> STATUS: DRAFT - not yet released; project ongoing.

This is the derived narrative of the Hearts AI project, July 2026 onward:
what was attempted, what was measured, what was closed permanently, and
what carried forward. Primary sources win on any conflict
(docs/release/INDEX.md): the append-only measurement ledger
(docs/speed_ledger.md), the rules file (docs/experiment_rules.md), the
roadmap (docs/ROADMAP.md), the pre-registrations, the verdict JSONs
(equity_data/verdicts/), and the git history with its retrospective notes
on the earliest twelve commits. Citations below are (ledger DATE),
(commit hash), (verdict file), or (doc path).

A note on tone: roughly half of what follows is failure. That is
deliberate — the negative results are load-bearing. Each closed
direction narrowed the search space that eventually produced the
headline result (the N=8000 match-aware search validation, era 6).

---

## Era 1 — Genesis (2026-07-04 .. 07-09)

Toolchain: the Antigravity IDE with Gemini 3.1 Pro as coding assistant
(git note on d205c3e; docs/RELEASE_PLAN.md sec. 1).

What was built, in five days:
- The C++ engine skeleton (d205c3e, 07-04), correct moon-shooting
  scoring the next day (dadfc6d), and a first playable game with a
  neural net seated by 07-06 (23d4f1e — HeartsEnv.hpp at 336 lines
  plus pybind bindings, per the git note).
- The automation loop (3d2e83c, 07-07): orchestrator.py + run_loop.py —
  propose a config, train, gate, promote-or-rollback — with rollback via
  config files after a git-branch-based scheme proved unworkable. This
  loop, re-gated and re-powered many times in later eras, carried every
  promotion the project ever made (git note on 3d2e83c).
- The ledger habit (cc2322f, 07-07): experiment_ledger.json begins
  accumulating machine-readable trial records and never stops.
- The first measurement tool: analyzer.py born at 268 lines (18f3892,
  07-09), tripling to 775 lines the same day (038746c) with the
  original preserved as analyzer_legacy.py — the first instance of a
  recurring pattern: measurement tooling grows faster than the thing it
  measures, and old instruments are kept, not deleted.
- Card passing enters the engine, bindings, and model (8320fb4, 07-09).

What did NOT exist yet: measurement discipline. "improved ai a lot"
(84b7a1e, 07-05) was by-eye judgment; the paired-deal, SE-reported
regime of later eras had no counterpart here (git note on 84b7a1e).

**Closed forever:** nothing yet — there was no apparatus to close
anything with.

**Carried forward:** the engine, the promote-or-rollback loop, the
ledger habit, the analyzer.

### Interlude — the toolchain switch and the quiet gap (07-09 .. 07-13)

On 2026-07-09 the project moved from Antigravity + Gemini 3.1 Pro to
Claude Code with the Fable 5 model doing ops and implementation under
human direction (docs/RELEASE_PLAN.md sec. 1; commit 0810764: last
old-toolchain commit 83f7437 at 07-09 23:41, first structured-style
commit f4963a2 at 07-13 08:39). Git is quiet for three days, but the
automated loop was not: analyzer_history.csv carries comparison rows
dated 07-11 and 07-12 for v3- and v4-generation milestones
(analyzer_history.csv rows 2026-07-11/12) — the v3/v4 MLP lineages,
including the v4-m10 net that later served as anchor, teacher, and
calibration opponent, were produced by the loop during this gap.

---

## Era 2 — Search and expert-iteration infrastructure; the v5 breakthrough (07-13 .. 07-15)

Three dense days. The infrastructure first:
- Decision-time search: belief-weighted determinized rollouts
  (f4963a2) — sample hidden hands weighted by the net's belief head,
  roll each candidate action out to the end of the deal with the raw
  net playing all seats, pick the best mean outcome.
- Expert iteration: SelfPlayGen (the C++ self-play data generator) +
  a distillation trainer + pass search (7d5d78d), moved onto the RTX
  4090 (2fb1ce5), then the batched InferenceServer that coalesces all
  self-play inference into one GPU funnel (973dc31).

Then a burst of negative results, all measured, all closed:
- **Value-bootstrapped search** (truncated rollouts + learned leaf
  values): negative — truncated search collapsed by +6 pts/deal with a
  visible-info evaluator (f3d280d; hearts_net.py oracle-head comment).
- **Oracle-leaf evaluation** (leaf values conditioned on the true
  determinized hands): NEGATIVE and closed (7a3abc3). Root cause
  measured on 7,200 leaf states from 150 fresh deals: oracle head true
  EV 0.106, RMSE 6.8 pts, indistinguishable from the visible-info head
  (0.118); the impressive 0.99 train EVs were memorization hidden by
  same-deal sibling leakage in a per-record holdout split. With ~7 pts
  of evaluator error against action-value gaps of 0.5–1.5 pts, no
  learned evaluator of that quality can replace rollouts.
- **ISMCTS tree search** (PUCT priors, Max^n backup): converges to the
  flat search's level and never exceeds it — at 1600 iterations it ties
  flat K=64 while costing 55x the wall clock (b835897). Closed.
- The **opponent-modeling confound** was identified in dual-model
  search (0a91f38) — the seed of the later neutral-opponent rule.

The synthesis (b835897): flat K-curve, tree search, and both leaf
evaluators plateau at the SAME level. The search extracts everything
the network knows; **the network is the binding constraint.**

That conclusion produced HeartsNetV5 (d0d6d6a): a card-token
transformer that re-encodes the SAME 550-dim observation as 52 card
tokens + 1 global token (see docs/release/ARCHITECTURE.md). Verdict
(a4136b5), both arms distilled from scratch on 2.93M banked K=64
teacher records with an honest per-file-tail holdout:
- **v5-M (7.6M params, d=320 L=6): gated -1.233 pts/deal vs the v4-m10
  baseline (t=-9.27, p<1e-6)** — more than 3x the best MLP candidate's
  -0.388 at HALF that candidate's parameter count. Belief BCE 0.270,
  the best belief head trained to that point (MLPs ~0.32).
- v5-S (1.9M, same size class as v4-m10): +0.058, a tie with the fully
  PPO-trained baseline, from distillation alone.

**Architecture, not size, was the lever** (a4136b5). v5-M was promoted
(293c0fd) with searched play a statistical tie with the v4 teacher
(-0.099 ± 0.526, n=1000) — the raw net leapt, search had already been
extracting most of it. The same 07-13..15 window also fixed the holdout
discipline itself: random per-record splits leak same-deal siblings;
distill.py moved to per-file-tail splits (d0d6d6a).

Immediately after promotion, the first PPO fine-tune on v5 passed the
then-current search gate at **-0.712 pts/deal (n=600, p=0.016)**
(docs/ppo_v5_round2_findings.md; milestone 1784156801) — a number the
weak gate would later turn out to have UNDERestimated (era 4).

**Closed forever:** learned leaf evaluators (both variants), ISMCTS-style
tree search for strength, search amplification past the net's ceiling
(docs/experiment_rules.md, closed directions).

**Carried forward:** the v5 architecture, SelfPlayGen + distill,
InferenceServer, the "network is the constraint" diagnosis,
TreeSearchPlayer (kept as a possible target generator, unused).

---

## Era 3 — Speed and cloud (07-16 .. 07-19)

The generation pipeline was the bottleneck for everything downstream,
so this era measured it honestly and made it fast. The speed ledger's
two-convention rule dates from here: A/B 50-deal runs for controlled
diffs, steady long-run averages for planning, never mixed
(docs/speed_ledger.md header).

Local throughput truths (all ledger, header table):
- Baseline v5 generation: 13.86 s/deal A/B, 12.20 steady.
- Finer batch buckets + persistent autocast: 9.78 / 9.56 s/deal
  (d790ef8) — and an instrumented proof that a suspected "process-age
  decay" did not exist (d790ef8 commit title).
- **SDPA fused attention: 6.56 / 6.32 s/deal steady** — promoted
  through a pre-registered gate with both outcome branches written
  down before the result was seen (docs/sdpa_gate_preregistration.md):
  **PASS, pooled +0.068 ± 0.133, n=3,600** (24bb45d). The prior ~6.4
  projection measured in at 6.32, within 1.5%, and projections were
  retired from the ledger thereafter (ledger, "Projection retired").
- CUDA Graph replay: measured a wash, env-gated off (fb4ef3b). Server
  staging and a two-slot pipeline: neutral and negative respectively;
  the 19–35 ms queue waits are intrinsic queueing on a saturated GPU
  (ledger 2026-07-18, queue-wait investigation).

Cloud, done the hard way:
- **H100 session 1 (07-17, $2.99/h):** steady 3.24 s/deal = 1.95x
  local; $2.69 per 1,000 deals; but the pre-registered cross-hardware
  equivalence gate **FAILED on power, not effect** — n=3,000 paired
  deals gave mean -0.010 (dead parity) with SE 0.207, one-sided 95% UB
  +0.331 vs the +0.30 margin. Cross-hardware bf16 flips decorrelate the
  pairing (per-deal delta std 11.35 vs the ~7.6 the criterion assumed)
  (ledger, H100 validation session). Lesson banked, spend $7.48.
- **H100 + AOTInductor session 2 (07-18):** AOTI forward 10.79 ms per
  2048 rows (vs 15.6 JIT on H100, 45.5 local); **steady 2.22 s/deal =
  2.85x local, $1.84 per 1,000 deals**; the card runs power-limited at
  667 W of its 700 W cap. The re-attempted equivalence gate **PASSED at
  n=8,000: mean -0.139, SE 0.123, UB +0.064 < +0.30** (ledger, H100
  AOTI session). The AOTI stack was certified for real generation.
- **First cloud-generated expert iteration (07-18/19):** operationally
  clean end to end (3,500 deals at 2.30 s/deal, $7.02, every shard
  validated twice) — and a strength FAIL: search gate +0.450 at n=600
  (ledger, first cloud iteration). Also this era's ops scar: a
  scratchpad gate driver without the `__main__` guard crash-respawned
  pool workers for 2.3 h (rule #9, docs/experiment_rules.md).

**Closed forever:** nothing scientific — but the idea that cross-hardware
numerical equivalence comes free was gone, and the eventual rule #14
("train in cloud, evaluate locally", era 6) grew from this failure.

**Carried forward:** the 6.32 s/deal local baseline, the certified
H100+AOTI pipeline, the Docker/queue/worker harness, the
pre-registration habit (SDPA was the first formal one).

---

## Era 4 — Gate re-powering and the PPO campaigns (07-19 .. 07-23)

The cloud-iter candidate's n=600 FAIL demanded a question the gate could
not answer: worse, or unlucky? The gate itself was the problem — n=600
had ~25% power against a true -0.3 (docs/experiment_rules.md rule #2).

- **Re-power: n=600 -> 2,400** (7f484f9), sharded 4-way, SE and n in
  every verdict. First full-power verdict: the cloud candidate re-gated
  at **+0.679 (SE 0.169, t=4.0) — definitively worse, not noise**
  (ledger 2026-07-19). Notably its raw play was dead-even (+0.003):
  the "distillation erodes search-carrying properties" pattern, visible
  inside a single same-teacher iteration.
- **PPO-on-v5 round 2 (4 trials, powered gate):** every trial produced
  large significant raw gains (-0.63 to -0.78) and a null searched
  delta. **Pooled: -0.078, SE 0.081 over 9,600 paired deals; 95% CI
  [-0.24, +0.08]** — excludes every historically promotion-worthy
  effect (ledger 2026-07-19/20; docs/ppo_v5_round2_findings.md).
- **Diagnostics A and B (07-21)** resolved the two live confounds:
  (B) re-gating the 07-15 PPO promotion at n=2,400 showed the pre-PPO
  net **+1.025 worse (SE 0.180, t=5.68)** — the promotion was real and
  the weak gate had underestimated it; (A) neutral-opponent raw
  evaluation showed the PPO raw gains were **genuine strength, not
  opponent exploitation** (-0.654 and -0.636, SE ~0.14, vs anchors the
  candidates never trained against). One candidate's head-to-head raw
  guard had understated its true neutral gain 3x — the head-to-head
  guard is biased in both directions (ppo_v5_round2_findings.md).
- **The raw-line promotion design** followed: promote on a powered
  neutral raw gate (n=2,500), demote the search gate to a
  non-regression guard (one-sided 95% UB vs +0.3). First promotion:
  **-0.619 (SE 0.141, n=2,500, p=1e-5), guard UB +0.187** (fff75a2;
  ledger 2026-07-21).
- And then the ceiling: round 1 against the NEW baseline pooled
  **-0.093 ± 0.074 (7,500 paired deals)** — the one-step PPO raw gain
  is substantially **one-shot** (ledger, raw-line round 1).

In parallel, the distillation program ran itself into a wall, carefully:
- v5-L (14–20M) distills from the existing 2.1M-record bank: fail by
  ~4 pts (ledger 2026-07-22). Hypothesis: stale bank. So a **fresh
  12,500-deal / 766,400-record bank** was generated locally over four
  installments, $0 (ledger 2026-07-22/23).
- Step 1, warm-start distill on the fresh bank: **+0.479, significantly
  worse.** Step 2: sharpening the teacher targets to 2.0 halved the
  damage (+0.248) but no variant reached parity; v5-L failed on every
  data mix (+3.3..+5.3). Step 3: sharpen saturates at 2.0 (4.0 is worse
  without entropy collapse), and 255k further PPO games on the best
  candidate moved nothing (ledger, steps 1–3). **The same-lineage
  distill-refresh recipe was closed** (docs/experiment_rules.md).
- A measurement bug worth its permanent rule: in-loop stage gates
  compared checkpoints against a file that WAS the working candidate —
  +0.000 (SE 0.000) self-comparisons. "SE of exactly 0.000 = you gated
  a net against itself" (rule #3).

The era closed with a calibration point: the deployed search player
measured **-1.016/deal stronger (SE 0.234, n=1,200)** than the 07-14
deployed snapshot — matching the predicted chain of promotions
(ledger 2026-07-23).

**Closed forever:** PPO fine-tune at the ceiling (one-shot); same-lineage
distillation of own-search targets; reaching a larger net by imitation
of existing teachers (docs/experiment_rules.md, closed directions).

**Carried forward:** the powered-gate discipline, neutral_raw_eval and
eval_search_pair instruments, the fresh bank as evaluation stock — and
a program with no remaining cheap lever on the per-deal axis.

---

## Era 5 — The match pivot (07-23 .. 07-24)

With both per-deal levers measured flat, the strategy was rewritten
rather than pushed harder. docs/ROADMAP.md (adopted 07-23, commit
5c9d8b8) reframed the goal: Hearts is played in MATCHES to 100, and the
real objective is match placement, not deal points. Phase 1: give the
system the match objective. Phase 2: prove a self-improvement loop at
7.6M params. Phase 3: scale only inside a working loop. The rules file
(docs/experiment_rules.md) was written the same day.

Phase 1 infrastructure (21cbd01, 77d38b6; ledger 2026-07-23):
- A match-to-100 env with score carry and tie-aware placements; a
  6-dim match context APPENDED to the 550-dim observation with a
  **zero-initialized projection** — the extended net is bit-identical
  to the baseline until training moves those weights.
- A paired match gate: null calibration (baseline vs itself, 24
  matches) gave ALL paired deltas exactly 0 — the pairing is airtight.
- The headroom teaser, worth the whole pivot: the score-blind baseline,
  ~2 pts/deal above the anchors, won only ~30–37% of 4-seat matches
  (chance 25%). Per-deal strength translates weakly to match wins.

Then the fastest-paying night of the project (ledger 2026-07-24):
- **Trial 1, FIRST match-aware promotion:** placement **-0.085
  (SE 0.041, n=800, p=0.018)**; neutral raw improved too (-0.233,
  p=0.043) — score-conditioning added strength rather than trading it.
- **Trial 2:** -0.133 (SE 0.040, p=0.00044), win 36.0% vs 31.5%
  (p=0.011). Unlike the per-deal axis, **the match axis compounds.**
- **Trial 4:** -0.087 (SE 0.036, p=0.008), win 39.8% vs 35.9%
  (p=0.025). Three promotions in four trials; cumulative win rate vs
  the fixed anchor field 26.8% -> 39.8%, mean placement 2.44 -> 2.05.
- The web app shipped the same day (hearts_web/: human seat vs three AI
  seats over HTTP, full match telemetry) — the user-calibration surface
  (68f92ea, d6f4ca2).

Then the instructive failure:
- **Trial 6: first-ever search-guard veto.** Match gate STRONG PASS
  (-0.121, p=0.0006) but searched per-deal +0.477, UB +0.739 vs +0.3.
  Match training had begun trading away search-substrate quality —
  foreshadowed by the narrow guard margins on promotions 1–3
  (ledger, trials 5–6).
- Was the guard protecting the right thing? The **bridge measurement**
  answered with data (8c934be; ledger 2026-07-24): the match-BLIND
  search player vs the match-aware raw net, both at match play, n=200
  paired matches: **search still wins matches — 50.5% vs 38.0%
  (discordant 51:26, McNemar p=0.006)**, placement diff -0.175
  (p=0.068). The p-value split is real, not contradictory: the effect
  concentrates near the win boundary (P2 conversion), which the win
  indicator captures at full power while mean placement dilutes it.
  The guard's protection was justified; **match-aware search became the
  queued ceiling configuration**, with the success gate pre-committed
  to "significantly beats match-blind search", nothing more.

**Closed forever:** nothing — but "per-deal strength is the objective"
was retired as a premise.

**Carried forward:** match env + gates, the match-aware raw lineage
(3 promotions), the guard-veto pattern, the web app, and the bridge
result that justified building match-aware search.

---

## Era 6 — Match-aware search and the N=8000 validation (07-24 .. 07-27)

The most heavily engineered measurement of the project. The design doc
(docs/match_aware_search_design.md) went through nine numbered reviews
in roughly two days (commit titles 0f1fc7e..edafaee), ending with a
simple spine governed by **halt-is-default**: flip/SNR gate ->
behavioral diagnostics -> N=8000 validation, each emitting a
machine-readable verdict JSON.

The components:
- **Equity model:** a small net mapping (rotated totals/100, deals/20,
  leader distance, pass direction) -> P(place 1..4) for the acting
  seat. Trained on 30k seeded coverage-mixture matches; calibrated on a
  dedicated 5k-match natural holdout. Result: ECE below its clustered
  noise floor everywhere (aggregate 0.0037 vs floor 0.0575); Brier
  0.614 aggregate / 0.336 near-terminal; SELECTED over both frozen
  lookup baselines (0.645) in aggregate and per-stratum
  (verdicts/diagnostics.json, verdicts/selection.json; ledger
  2026-07-25).
- **The spine gate HALTED.** Probe of 200 matches / 11,245 logged
  decisions: raw tension-band flip rate cleared its 5% floor hugely,
  but SNR missed the 1.0 threshold — and the deal-point REFERENCE
  scoring also lived below 1.0, so the threshold itself was mis-set;
  the built-in reference comparison did its job. Decisive diagnostic:
  **CONFIDENT flips (equity gap > 2xSE at K=64) were only ~1.6% of
  tension decisions — flips are ~97% noise at K=64**
  (verdicts/flip_snr.json: tension flip 38.9%, SNR 0.62 vs deal-point
  reference 0.86; the ledger entry quotes 36.8% / 0.41 / 0.57 from its
  analysis pass — both conclusions identical: HALT). The halt-default
  held exactly as designed; options went to the user, no unilateral
  proceed (ledger 2026-07-25).
- The user chose the adaptive-K probe: **K=256 in endgame states
  (max >= 85) raised the confident-flip rate 2.6x at +37% match cost**,
  adopted as the standing search schedule, rules #15 (003a846;
  docs/experiment_rules.md #15).
- Full C++ integration followed (48c1394): equity leaf scoring with
  exact placements at terminal states, and match context written for
  the ACTING seat at every observation site — the per-seat rotation
  rule the design review had flagged as the top silent-null risk.
  Behavioral diagnostics ran via a constructed-score-state mode
  (--behave, ea2d33e; behave_*.csv). [NEEDS CITATION: the behavioral
  suite's pass/fail outcome was not found in ledger or verdict JSONs —
  only the tooling and its output CSVs.]

The ops saga, kept in full because it recurs in era 8:
- The first N=8000 local launch (8 shards, K=256) **wedged the NVIDIA
  driver** — unkillable processes, hung nvidia-smi, reboot required.
  Root cause, diagnosed then confirmed by fix-and-retest: DirectBackend
  lacked BOTH server-path cures (it cleared the bf16 cast-cache every
  forward and never bucketed batch shapes); 16 CUDA contexts of
  allocation storm livelocked WDDM. Fixed (b929c3d), re-test stable
  (ledger 2026-07-25/26).
- The measured concurrency curve then killed the local plan: optimum 2
  shards at ~56 pairs/h, true pair cost ~2 min — **N=8000 local ~ 6
  days**. The design doc's "~5.8 h at 8 shards" had rested on a ~21
  s/pair figure that was search-vs-RAW and never applied
  (ledger 2026-07-25/26). So the validation went to a cloud fleet —
  inverting the doc's own "search validation stays local" policy, with
  the pairing kept intact by never splitting a pair across nodes
  (e2280a3).
- Fleet ops (ledger 2026-07-26): a $0.55 pilot pod proved the harness —
  A/A ref-vs-ref, 20/20 pairs BIT-IDENTICAL across arms. Then 8
  community RTX 3090s at $0.22/h, bootstrapped via a private release
  bundle and a scoped 7-day read-only token (the broad token never left
  the local machine).

The result (ledger 2026-07-27; single pre-registered analysis, no
interim looks):

> **Match-aware search: 48.91% match wins vs the frozen match-blind
> reference's 44.47% = +4.44 win-points (SE 0.68). McNemar one-sided:
> discordant 1668 vs 1313 (q=0.373), p ~ 5e-11. All 8 shards positive
> (+2.4 .. +6.6). Realized paired placement SD 1.351 vs the bridge
> run's 1.35 — dead on.**

The placement structure is the interesting part: the match-aware arm
converts P2 into BOTH P1 and P4 (P1 3913 v 3558, P2 1028 v 1838,
P4 1780 v 969) and its MEAN placement is worse (2.098 v 1.980). It
trades expected placement for win probability — exactly the win-equity
objective, not generic strength. Exploratory tercile split by match
length: +13.3 win-pts in short matches, +0.9 mid, -5.9 long (deal count
is endogenous; labeled exploratory).

Two things owned in the record: the fleet binary logged only final
outcomes, so the pre-registered S1/S2 strata and dose-response
secondaries were NOT computable (the instrumentation gap was closed the
next day, rules #16); and the run cost $62.72 against ~$46–48 projected
(realized host pace varied 28–44 pairs/h; lesson: re-forecast mid-run).

**Closed forever:** nothing closed — one thing opened: match-aware
search became the validated ceiling configuration (rules #16).

**Carried forward:** the equity model, the K=64/256 schedule, the frozen
reference artifact (see docs/release/model_cards/), the fleet harness,
and the validated teacher for Phase 2.

---

## Era 7 — The evolved regime (07-28 .. 07-29)

Post-validation lock-in, all local, $0 (ledger 2026-07-28):
- **Rules #16**: match-aware search is the ceiling config; the search
  guard now runs BOTH arms match-aware (556-dim traces + equity
  leaves), and promotion re-exports the match trace so the guard
  baseline tracks the champion. Null calibration: exactly 0.000.
- **Anchor diversification**: the match gate alternates v3-m7 and
  v4-m10 anchor fields — retiring the anchor-overfit watchpoint that
  had stood since the raw-line era.
- Match CSVs now carry the stratum columns the N=8000 analysis had
  lacked.

Then both gates were re-powered, and both re-powers were vindicated
within a day (ledger 2026-07-28/29):
- **Match gate n=800 -> 3,200.** Trials 1–4 under the evolved guard
  pooled to -0.027 ± 0.017 (p ~ .06) — a real sub-bar effect the n=800
  gate was coin-flipping (43% power vs a true -0.05; 90% at n=3,200).
  Same lesson as the 07-19 search-gate re-power: don't half-power.
- **4th match-era promotion** (first at n=3,200): -0.029 (SE 0.017,
  p=0.0456) — n=800 would have coin-flipped it. The evolved guard
  passed by 0.008, the second consecutive knife-edge, and the false-veto
  analysis said a dead-neutral candidate passed only ~61% of the time.
- **Search guard n=2,400 -> 4,800** (neutral pass rate ~61% -> ~86%;
  margin unchanged — the noise was the problem, not the tolerance).
- **5th match-era promotion, re-power directly vindicated:** placement
  -0.031 (p=.0247), win 53.3% v 50.5% (discordant 499:407, p=.0012),
  score -0.98 (p=.0078); guard at n=4,800 UB +0.258 — at the old
  n=2,400 the UB would have been ~+0.335, a FALSE VETO. Since the
  re-powers: 3 gate-passes in 3 trials, 2 promotions + 1 correct
  substrate veto.

The analyzer's before/after bookend (analyzer_history.csv, rows
2026-07-23 and 2026-07-28, both vs the same v4-m10 milestone): the
promoted baseline moved from 6.18 avg pts / 34.9% deal wins to 6.11 /
36.1% while v4-m10 held ~7.2–7.3 / ~28%; solo-difficulty delta -2.134.
And the known weakness stayed known: moon defense 51.5% vs v4-m10's
61.1%, moons conceded 272 vs 151 over 2,000 deals — telemetry informs,
never gates (rule #5), and this hole is on the roadmap as a curriculum
problem, not an architecture one (docs/ROADMAP.md, v6 section).

**Carried forward:** an evolved, correctly-powered gate/guard regime and
two more promotions (milestones 1785273667 / cbfde942 and 1785322724 /
8a89da90).

---

## Era 8 — Expert iteration against the match-aware teacher (07-29 .. ongoing)

The roadmap's pre-committed sequence: on PPO plateau, ONE gated
match-aware expert-iteration experiment (docs/ROADMAP.md). The bet was
explicit: this is NOT the closed same-lineage recipe, because the
teacher now demonstrably makes different, better decisions in score
context (+4.44 win-pts of validated equity signal).

Tooling and generation first (ledger 2026-07-29..31):
- SelfPlayGen --match: four match-aware search seats, score carry,
  per-deal match context for every seat, 824-byte records with
  tie-aware placement rewards, and seeded tension starts
  (--start-totals). Built, reviewed, smoke-verified down to
  zero-sum reward checks.
- **The wedge recurred** — one process this time. The first diagnosis
  ("the equity model bypasses the b929c3d hardening") was WRONG and is
  marked wrong in the ledger: the equity module runs entirely on CPU
  and cannot touch VRAM. The actual, code-confirmed cause: **unbounded
  batch coalescing in InferenceServer** — the server forwarded its
  whole queue as one batch; K=256 endgames x 14 threads produced
  single forwards of tens of thousands of rows, the allocator retained
  the peak blocks, VRAM ratcheted 12.8 -> 23.9 GB until WDDM wedged.
  Fix: **cap coalesced rows per forward** (default 8,192; 45821a6).
  Verified across a 4.5 h full-throttle run: peak 12.9 GB, no wedge
  (ledger 2026-07-30/31). A surprise came free: the old unbounded
  batches were themselves hurting throughput (44 s/match at 5 paced
  threads vs 37 at 14 unpaced pre-fix).
- Bank complete: **333,415 records** — 72.8% natural, 27.2% seeded
  tension families (~4.5x enriched over natural play's tension rate)
  (ledger 2026-07-31).

Then the experiment, one shot, and its anatomy (ledger 2026-07-31):
- Three distill variants were measured on holdout BEFORE the gate.
  Soft targets (sharpen 2 and 8): the equity-scored teacher's policies
  are **near-uniform** — P(win) gaps of a few percent — so
  power-sharpening is a no-op; both variants UN-sharpened the champion
  (entropy 0.32 -> 1.04–1.08) and dropped teacher-match below the
  baseline. Hard argmax targets fixed the imitation metrics (60.1%
  holdout, +10.2 on tension decisions) —
- — and **the gate failed catastrophically anyway: win 39.9% v 50.3%
  (discordant 377:710), placement +0.292 (SE 0.019, ~17 SE worse).**
  Per the pre-registration, the experiment is CLOSED; the baseline was
  untouched (hash 8a89da90 verified).
- The post-mortem is the era's real product: in the ~73% of states
  where equity is flat, **the teacher's argmax IS noise** — a coin flip
  between statistically tied actions. Hard training copies those coin
  flips and overwrites the baseline's genuine per-deal knowledge. The
  +4.44-win-pt edge exists at decision time but is NOT extractable by
  whole-distribution imitation. New closed direction: distilling a
  search teacher requires targets that ENCODE PREFERENCE STRENGTH.
- **v2 — filtered targets — ran 08-01..08-05 and closed the direction
  decisively** (docs/expert_iter_v2_prereg.md, user-signed; results in
  docs/expert_iter_v2_results.md and equity_data/verdicts/). The
  recipe: train only on flip-confident decisions (top-2 equity gap >
  2x its SE), anchor the flat-state policy to the baseline via a KL
  term; record format v2 carries per-decision search statistics so
  filtering happens at train time. Generation ran demand-aware to
  per-family reserves (natural >=50k confident records plus six seeded
  tension families >=8,400 each, ~1.16M records total) with
  user-controlled pacing profiles and PID-file-only kills.
- The recipe freeze produced its own finding first: BOTH pre-registered
  anchor coefficients {0.25, 1.0} violated the entropy diagnostic
  (ratios 3.03/2.35 vs the <=2.0 bar). A holdout-only exploration —
  run before any gate data existed, recorded in the freeze report —
  found a **monotone dose-response: a stronger anchor improved every
  axis simultaneously** (teacher-match 0.594 -> 0.688, non-confident KL
  1.62 -> 0.229, entropy ratio 3.03 -> 1.76 across coefs 0.25 -> 4.0;
  epochs were not the lever). A registered amendment froze lambda=4.0
  (docs/expert_iter_v2_freeze_report.md). A second pre-data amendment
  added three exploratory continuous-certainty arms (weight w=erf(z/2),
  loss w*CE + lambda(1-w)*KL) to test whether BINARY filtering was the
  problem.
- The comparative stage (16 trainings, 32 evals, zero failures):
  **all five binary candidate mixes significantly WORSE than baseline**
  (+0.10..+0.16 placement, one-sided max-T p_adj = 1.0 on 6,400
  (block, match) pairs per arm) — and **all three continuous-certainty
  arms statistically identical to baseline** (|delta| <= 0.009; the
  registered enrichment and size contrasts both null). The confirmation
  battery was skipped: nothing to confirm.
- The mechanism verdict is the era's product: v1's noise hypothesis was
  **refuted**. The harm lives in the confident teacher signal itself —
  neutralizing the noise converges to no-change, so the equity-scored
  teacher's ordering signal is INERT for match play. The closure
  (docs/experiment_rules.md) covers ANY equity-scored-target recipe,
  binary or continuous. Outside it: teachers with a DIFFERENT signal
  source — visit counts, or demonstrations from games where something
  is actually at stake. Era 9 went to the second.
- One transferable side finding: the KL anchor behaved as a pure
  regularizer with clean monotone dose-response — worth knowing for any
  distill-onto-RL-sharpened-net setting.

---

## Era 9 — The exploiter league (08-05 .. ongoing)

If self-improvement recipes are dead from this baseline (eras 7-8), the
remaining lever is targeted: attack the measured weakness, then teach
the defense. The weakness had been on the books for weeks — the
analyzer's moon-defense number (51.5% vs v4-m10's 61.1%, concessions
~2x) — but what promoted it from telemetry to agenda was a human:
**the user shot four moons in eight fully-logged deals against the raw
net** on the project's web app (match VFFCIjaDZn188tAJ, 2026-08-02;
hearts_web/, see ARCHITECTURE sec. 8). Replay analysis showed free
tempo blocks declined and two moons with no in-suit block available at
all — a passing-layer failure, a curriculum problem, not an
architecture one. Human play's role in the loop is exactly this and
deliberately no more: exploit discovery and calibration, never direct
policy training (the volume math forbids it).

The exploiter league attacks the hole with the project's full
measurement discipline — frozen instruments, certified attackers,
pre-registered gates, halt-default (docs/exploiter_league_prereg.md,
user-approved).

**Phase A — build and validate the attacker instrument (08-05/06).**
A search-shooter mode inside SearchPlayer: moon-probability scoring
over the shared determinizations, a moon-line rollout continuation, and
pass-phase shooting via a rewound pass search. Two modes: AGG (always
shoots — dense threat) and SEL (commits only when moon equity beats
normal play — realistic threat). The instrument spec froze at K=64 flat
after a hard lesson: K=256-endgame under six shooter processes wedged
WDDM and froze the machine (moon matches cross totals>=85 within ~3
deals, so the endgame boost fired on most decisions). Base rates over
402 matches per combo (docs/exploiter_league_phaseA.md, all checks
PASS): AGG completes **0.515 moons/deal** against baseline defenders —
77x the background rate; SEL 0.367 at a 71.2% attempt rate; and the
ordering check that validates the whole premise: **v4-m10 defenders
hold the same SEL attacker to 0.237** — the older net really does
defend better, inside the instrument that will judge the fix.

**Phase B — certify cheap attacker clones (08-06/07).** Generation at
search speed is unaffordable, so the shooters were distilled into small
nets with a registered retention bar: >=50% of the teacher's moon rate.
SEL clone passed first try (0.224 moons/deal [0.196, 0.252]). The AGG
clone HALTED at d192 (retention 46% — confirmed at n=500, a real miss,
not noise) and passed as a d256 retrain on its single registered shot
(**shooter_agg_v1b, 0.291 moons/deal [0.276, 0.306]**,
docs/exploiter_league_phaseB.md). Clones are identified by md5 —
weights stay out of the repo.

**Phase C / round 1 — PPO exposure (08-07/08).** Match-mode PPO with
0.15/0.15 clone shares in the opponent pool. The registered defense
gate (64 CRN-paired matches vs the frozen SEL probe) went monotonic
across trials: -0.031 -> -0.047 -> **-0.250 moons/match (p=0.029) —
the first defense-gate pass the project has recorded.** Defense is
teachable by exposure. But the same candidate failed both protection
gates (verdicts/exploiter_r1_gates23_r1t3.json): search guard +0.267
(UB +0.453 vs +0.3), match non-inferiority UB +0.034 vs +0.030 on a
~zero point estimate — a precision failure that motivated re-powering
the bound to n=6,400 for round 2. Round 1's verdict, recorded in the
round-2 pre-registration: **the gain is real, the vehicle pays for it
wrongly — PPO's drift is unaimed.**

**The teacher check that set round 2 (08-08).** Before designing
imitation, measure whether the behavior to imitate exists: CRN-paired
on the gate seed block, **search defenders concede 1.208 moons/match
where raw defenders concede 2.417** — a 50% reduction (t=-8.3,
p<1e-5), plus 13 counter-moons in 440 deals. The search player already
knows how to defend; the raw net does not.

**Round 2 — anchored defense distillation (08-08 .. in flight).**
The pre-registration (docs/exploiter_league_r2_prereg.md, approved
08-08) inverts round 1's failure: imitate the search defenders
SUPERVISED (hard-CE on moon-alive defender decisions and passes), and
control drift BY CONSTRUCTION — ordinary decisions self-distill to the
baseline's own argmax, an offline drift screen (>=97% argmax agreement
on 20k held-out ordinary positions) runs before any gate, and the gate
family is unchanged so results stay comparable. Phase A2 generates the
corpus: certified clones attack, three search defenders play and have
their decisions recorded (seat-tagged v2 records; generation is
losslessly pausable by prereg amendment — kill-anytime by PID file,
resume trims the at-most-one partial match). Status at time of
writing: AGG half complete and in-band, the >=30,000 moon-alive volume
condition already met (validate_r2_corpus.py, first live read
all-clean), SEL half finishing; one watch item — SEL completion sits
just above its 0.05/deal halt floor, and the complete-corpus number
decides.

---

## Meta — how the project was run

**Era 10 (08-09..08-11): the last teacher signal closes.** League
rounds 2 and 3 ended the same week: supervised imitation of search
defenders was dead-null at the defense gate even after a KL-anchored
containment redesign (and a registered diagnostic showed the drift
screen excluded the very pass-region where defense lives), and
anchored PPO could not be dosed into its target drift band in two
calibration shots. Phase 2 then asked the roadmap's own make-or-break
question with tree-search visit counts as distillation targets: the
pipeline succeeded at every stage (validity probe adjudicated, strength
screen passed, 188k records generated, teacher-KL cut 5.6x) and the
faithful student was WORSE, with a clean dose-response (the more
distilled pick failed harder: match +0.083 vs +0.042 placement, guard
UB +0.427 vs +0.318). The mechanism reads plainly: a search's visit
distribution encodes where it had to LOOK, not what it endorsed;
imitating attention softens the policy onto explored-but-rejected
moves. With era 8's equity-ordering targets and era 10's visit counts
both measured non-improving, every available encoding of the searched
teacher is closed at 7.6M — the strongest evidence yet that the
network, not the signal, is the constraint
(docs/phase2_visitcount_results.md).

**Era 11 (08-11..): capacity and structure, together.** The v6
preregistration (docs/v6_prereg.md) treats the moon-defense hole and
the capacity question as one design: an observation extension whose
capture channels make trick winners and point flow OBSERVABLE instead
of a 13-step recursion (and which exposed a long-standing wart — the
old context blocks were absolute-seat while the net never knew its own
seat); a seat-token architecture where moon threat and match targeting
live on entities that exist; auxiliary heads that force a threat
representation the way the belief head forced hidden-hand inference;
a fresh search-teacher bank generated with certified shooter clones at
one table in eight (defenders recorded, attacker never); and control
arms that finally isolate scale from structure from data. Promotion
runs the same gates as every era, with moon defense as a registered
secondary outcome that reports but never gates.

**The AI-assisted workflow, and its evolution.** The project had two
toolchains: Antigravity IDE + Gemini 3.1 Pro built the engine, the
first nets, and the automation loop (07-04..07-09); Claude Code + the
Fable 5 model carried everything from era 2 on, doing implementation
and ops under human direction — with all promotion decisions, spend
approvals, and design sign-offs remaining explicitly human
(docs/RELEASE_PLAN.md sec. 1; rules #13 and the "options surfaced to
user, no unilateral proceed" pattern in the ledger, e.g. 2026-07-25).
Both toolchains are credited; the switch date (2026-07-09) is
corroborated by the commit record (0810764).

**Pre-registration and the append-only ledger.** From the SDPA gate
(07-17) onward, consequential measurements were registered before
results existed — criterion, and the action on EACH branch (e48d3ec;
docs/sdpa_gate_preregistration.md). The N=8000 validation went through
nine design reviews and ran as a single pre-registered analysis. Gates
default to HALT, and halts held when triggered (flip/SNR, 07-25). The
ledger (docs/speed_ledger.md) is append-only: corrections are appended,
never rewritten.

**Wrong-then-corrected diagnoses, kept deliberately.** The record
preserves its own mistakes because they are evidence about method: the
mis-set SNR threshold caught by its own reference comparison (07-25);
the wedge-recurrence first diagnosis marked WRONG in place, with the
lesson "verify device placement before blaming a code path" (07-30);
the retired throughput projection (07-17); the gate-a-net-against-itself
bug with its diagnostic signature (07-23); and the launcher-discipline
rule violated, re-learned, and re-documented three times
(docs/experiment_rules.md #8; ledger 2026-07-30). Every rule in
docs/experiment_rules.md carries the datestamp of the incident that
taught it. That file, not any single result, is the project's most
transferable artifact.

---

## Coda (2026-08-21): the sixth promotion is an ensemble

The v6 campaign's larger network — which tied its small sibling on
imitation and was shelved without a promotion — turned out to be the
best moon-defense specialist the project ever measured, once it only
had to play the ~10% of decisions where its own threat head fires. Two
rounds of preregistered training on the champion (reward shaping,
threat-information adapters) had already measured that the champion
itself could not be moved in-band; composition moved it. The gated
ensemble (champion + arm a + moon-head router, one 882-input module)
passed non-inferiority (−0.011 ± 0.007), cut concessions to the search
attacker 29% (−0.742 ± 0.071, n=256 fresh seeds), and was promoted
raw-only — searched play keeps the champion's traces. Round-7 record:
docs/exploiter_league_r7_results.md; the program's design space and
next steps: docs/gated_ensemble_program.md. The era also bought its
measurement lessons the usual way: one retracted gate (an 882 defender
path that silently zero-filled — hence the standing gate-fires check)
and a wedge-and-recovery week that ended in chunked resumable
evaluation drivers.

Cross-references: system details in docs/release/ARCHITECTURE.md;
measurement practice in docs/release/METHODOLOGY.md; every number above
in table form in docs/release/RESULTS.md; builds and reruns in
docs/release/REPRODUCING.md.
