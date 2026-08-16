# v6 network — post-mortem and the v6(2) design

Written 2026-08-16, after the v6 program concluded (Stage 4 halted
2026-08-13; data-scaling probe + Addendum A 2026-08-15/16 → Stage-2b not
proposed, v6 shelved). Sources: docs/v6_prereg.md, docs/v6_next_plan.md,
docs/v6_data_scaling_prereg.md, docs/speed_ledger.md (08-12 → 08-16),
equity_data/verdicts/, hearts_net.py, HeartsEnv.hpp. Every number below
is quoted from those records; nothing here is a new measurement.

---

## 0. One-paragraph verdict

v6 asked whether a bigger network with more information and seat-level
structure, distilled from a fresh, defense-pressured bank of the searched
champion's decisions and then sharpened by match-PPO, could beat the
7.6M v5 champion (8a89da90). It could not, in this form. The information
and structure were real improvements to **imitation** (arm b vs c: CE
0.838 vs 0.868, teacher-match 64.2% vs 63.0%); the 2.55× scale bought
**nothing** at this bank size (arm a vs b tied); every from-scratch
distill landed **0.6–0.8 points/deal behind** the champion; match-PPO in
the champion's config regime **damaged** the distill (+0.296 → +0.167 →
+0.080 placement vs its own lineage, never crossing); the bank was
data-bound in imitation but **strength saturated between 1.0M and 1.5M
records** (S3−S2 = +0.018 ± 0.074/deal), so doubling it was declined on
measurement. The champion is unchanged. The most durable lessons are
about *method*: bundle nothing, never start from scratch against an
RL-mature champion, and never compare distills on one training seed.

## 1. What v6 was for — the hypothesis as registered

The prereg's converging evidence (docs/v6_prereg.md §"Why this"):

1. Every search amplifier (K-scaling, ISMCTS, learned leaf evals)
   plateaus at the same ceiling → the network is the binding constraint.
2. Both encodings of the searched teacher's knowledge are measured
   non-improving at 7.6M (equity-ordering targets, visit-count targets).
3. League r1 showed defense is teachable but not containable in the
   mature 7.6M net (guard +0.453) — read as capacity contention.
4. The moon hole is partly an **information** problem: the observation
   never says who CAPTURED cards or who won tricks (winner identity
   needs a 13-step recursion the obs does not seed), and seats have no
   representation in v5's token set.
5. v5-L closed "scale from the stale bank", not "scale inside a fresh
   loop".

Claim under test: *structure (capture information + seat entities +
threat-shaping aux heads) is the lever, scale is the multiplier, and
together they reopen a compounding distill→PPO loop that 7.6M could not
host.*

## 2. What v5 tracks, and what it does not

The v5 observation (HeartsEnv.hpp `ObserveFor`, 550 dims + 6 match ctx):

| block | dims | content | frame |
|---|---|---|---|
| hand | 0–51 | my cards | — |
| current trick | 52–103 | cards on the table | — |
| seen | 104–155 | cards already played (not in any hand / on table / in a pass pick) | — |
| deal scores | 156–159 | round_scores/26 | **ABSOLUTE seats** |
| trick position | 160–163 | one-hot #cards already in trick | — |
| hearts broken | 164 | flag | — |
| void tracker | 165–180 | 4 seats × 4 suits | **ABSOLUTE seats** |
| pass direction | 181–184 | one-hot L/R/A/hold | — |
| in-passing | 185 | flag | — |
| passed away | 186–237 | the 3 cards I passed | — |
| received | 238–289 | the 3 cards I received | — |
| played-by | 290–497 | 4×52 who played what this deal | **RELATIVE** (me, left, across, right) |
| timing | 498–549 | (trick_index+1)/13 per played card | — |
| match ctx | 550–555 | 4 match scores, deals played, leader distance | RELATIVE |

HeartsNetV5 re-encodes this as **52 card tokens + 1 global token**: each
card token = identity embedding + a projection of that card's 10 per-
card channels (hand, table, seen, passed, received, played-by×4,
timing); the global token projects the 30 ctx dims (156–185) plus the
zero-init match projection. Policy = one logit per card token; belief =
3 logits per card token; value on the global token. 7.6M params
(d320 L6 h10).

**What v5 cannot know from its inputs** (established during Stage 0):
- **Who took which trick, therefore who holds which points.** The
  played-by planes + timing say who played what and *in which trick*,
  but the trick winner requires replaying lead order and rank
  comparisons through the whole deal; nothing seeds that recursion.
- **Which opponent's deal score is which.** Deal scores and voids are
  in ABSOLUTE seat order and the net is never told its own seat index
  → per-opponent deal points and voids are structurally
  unattributable to the played-by planes. (This wart was discovered
  building obs v2 and is real. It did not obviously cost strength — see
  §5 F8.)
- Within-trick order (who led, who followed in what order) except by
  inference from the table cards; whether a seat is still moon-alive;
  Q♠ status; hearts remaining.
- Anything across deals in the match except the match totals — no
  memory of an opponent's earlier behavior.

## 3. What v6 added

**Observation v2 (Stage 0):** +326 dims appended past 556, all public,
all RELATIVE:
- per-card within-trick position (556–607) and led-the-trick flag
  (608–659);
- **taken-by**, 4×52 planes (660–867) — the capture attribution v5
  lacked;
- tricks won per seat (868–871); **moon-alive per seat** (872–875);
  hearts unseen (876); Q♠ status one-hot (877–881).

Verified before anything trained: A/A determinism, validator invariants
(taken-by sums reproduce round_scores; winner-recursion cross-check;
moon-alive consistency), and the extended-v5 identity (a v5 checkpoint
with zero-init projections over the new dims is bit-identical on
550/556/882 inputs).

**HeartsNetV6 (Stage 1):** 57 tokens = 1 global + **4 seat tokens** + 52
card tokens. Card channels 10 → 16 (position, led, taken-by×4 added).
Seat tokens (relative) = identity embedding + 5 features [match score,
tricks won, moon-alive, deal points taken (derived from taken-by ×
penalty), leads-current-trick (derived)]. Global token = ctx 30 + match
ctx 6 + obs-v2 tail. Heads: policy per card, value on global, belief per
card (as v5), plus **training-only aux heads on seat tokens** — moon
(did this seat moon the deal) and per-seat final deal points. The v5
oracle head (value given true hands, measured uninformative) deleted.
Size d448 L8 h8 = **19.37M** (2.55× v5-M).

**Record format v3 + Stage-2 bank:** obs[882] + mask + chosen action +
seat + match id + per-deal outcome labels (final round_scores,
mooned-by) + belief labels; teacher = deployed flat searcher K=64/256
match-aware, every decision recorded; **1/8 of matches seat a certified
shooter clone** (defenders recorded, attacker excluded). 1,524,821
records / 2,304 matches; $96.31 cloud + local tail; all v3 checks pass.

## 4. What we did and what happened

| stage | what | result |
|---|---|---|
| 0–1 (08-11) | obs v2 + HeartsNetV6 built, null contracts | all verification PASS; the absolute-seat wart found |
| 2 (08-12) | bank: 24 chunks × 96 matches, cloud+local | 1.52M records, all checks pass, $96.31 |
| 3 (08-13) | from-scratch distills, 3 arms × 2 lr × {2,3,4} ep, batch 512 | freeze: a = lr1e-4 ep3 (CE 0.8413, 64.2%, entropy 0.832); b = lr3e-4 ep4 (0.8381, 64.2%, 0.854); c = lr3e-4 ep2 (0.8681, 63.0%, entropy 0.891 out of band — registered amendment). **a-vs-b tied; b>c.** |
| 3 screens | neutral raw n=2500 vs champion (single seed each) | a +0.828 (SE 0.155), b +0.628 (0.153), c +0.594 (0.143). Band (UB ≤ +1.5) met → Stage 4 |
| 4 (08-13) | run_loop match-PPO from arm a, candidate_lineage, n=3200 lineage gates | +0.296 → +0.167 → +0.080 placement vs own baseline (SE ~0.021), win% 35/46 → 44/48; moon-shot suppression every trial; **HALT** (three-strikes clarification, amendment budget untouched). Trials 1–2 of the earlier ledger entries had run train.py on a fixed config (owned deviation) and gated against the v5 champion (+0.137, +0.169). |
| side | C++ obs-v2 search path | built + verified (882 in ProbeObsDim last, FillObsRow assembles from env; two latent memory bugs fixed; v5 byte-identical regression) — Stage-5 tooling never blocked |
| probe (08-15) | frozen arm-a recipe on nested 0.46M / 0.97M / 1.44M subsets, 2 seeds, fixed full-bank holdout, per-match paired | control S1→S2: CE −0.070, match +2.75 pp; decision S2→S3: CE −0.022, match +0.89 pp; all q<1e-4 → **DATA-BOUND** (slope diminishing: +2.5 → +1.6 pp/doubling). Screens: +1.888 / +0.439 / +0.766 (unpaired). |
| Addendum A (08-16) | paired strength, both arms probe nets, n=5000 × 4 | S2−S1 −1.20 (SE 0.08); **S3−S2 +0.018 (SE 0.074), UB95 +0.163 → Stage-2b NOT PROPOSED; v6 SHELVED.** Same-size seeds differ 0.30–0.36/deal. |

Cost of the arc: ~$96 cloud, roughly six machine-days across generation,
training, gates and probes; zero promotions; champion md5-verified intact
after every hard kill.

## 5. Findings (measured)

- **F1 Structure helps imitation at fixed scale.** obs-v2 channels + aux
  heads on the v5-M architecture: CE 0.868 → 0.838, teacher-match 63.0 →
  64.2% (arm c → b). Bundled — the contribution of obs v2 vs aux heads
  is not separated.
- **F2 Scale alone bought nothing pre-PPO.** Arm a (19.37M) tied arm b
  (~7.4M, same structure) on imitation, and its single-seed screen was
  slightly *worse* (+0.83 vs +0.63, within seed noise — see F6).
- **F3 Match-PPO from a fresh distill, in the champion's config
  regime, damages the net.** Three run_loop proposals in the
  config_chain_base neighborhood (lr ~1e-5, eps 0.199, entropy 0.002 —
  evolved for gently refining an RL-mature net at its ceiling) each made
  the candidate worse than its own lineage baseline; monotone mitigation
  never reached zero; the candidate shot fewer moons than its baseline in
  every trial (teacher aggression eroded before reward replaced it).
  This is the same drift class league r2/r3 measured.
- **F4 → measured.** Imitation is data-bound (every step significant,
  slope diminishing) but **strength saturated between 1.0M and 1.5M
  records**. Imitation gains and strength gains decoupled at the top of
  the range: +0.89 pp teacher-match bought +0.02 ± 0.07/deal.
- **F5 Fresh data alone did not reproduce the lineage.** Arm c (v5-M
  architecture, obs v1, no aux) — the "v5 scratch distill from a fresh
  bank" — landed +0.59/deal behind the champion. v5-M's own scratch
  distill (July) landed AHEAD of its era's baseline; the difference is
  the baseline: 8a89da90 is five match-era promotions of RL on top of a
  distill. From-scratch imitation of search argmax caps out well short
  of an RL-mature net.
- **F6 Training-seed variance in strength is large.** Two seeds of the
  same recipe and data differ by 0.30–0.36/deal (Addendum A), ~2 SE at
  n=5000, and comparable to the Stage-3 arm gaps. Imitation metrics are
  far less seed-sensitive (0.3–0.6 pp). Any single-seed strength ordering
  of distills — including Stage 3's a/b/c screens — is inside seed noise.
- **F7 The champion's own play is not very imitable by argmax.**
  Teacher-match ceilings ~64% on the general bank (and ~54% for the
  champion vs its own search defender on moon-alive states, r2
  diagnostic) reflect determinization noise in a K=64 teacher as much as
  student capacity; teacher-match is a coarse currency (league r2 called
  it dead for defense).
- **F8 Fixing the seat-attribution wart did not visibly buy strength.**
  Obs v2 made per-opponent points, captures and moon-alive explicit; the
  imitation gain (F1) was real but small, and no strength gain
  attributable to it survived the from-scratch start. Either the raw net
  had already learned workarounds, or the information matters mostly in
  states the strength instruments weight lightly, or the from-scratch
  confound hid it — the design cannot tell, which is itself the lesson.

## 6. What went wrong — honestly

1. **Everything was bundled.** Scale + structure + information + fresh
   data + defense pressure + aux heads went into one from-scratch
   distill. Arms a/b/c gave two contrasts (scale; structure+obs+aux vs
   none) and no way to attribute the structure gain, or to see whether
   the information alone would have moved the *champion*. The roadmap's
   queued v6 entry (2026-07-29) had said the opposite — pure scale first,
   warm-started, "no bundled architecture deltas — the run answers exactly
   one question." The prereg overrode that on the strength of the
   converging-evidence argument.
2. **From scratch against an RL-mature champion.** The v5-L closure was
   "reaching a LARGER net by imitation of existing teachers, any data
   mix — closed; untested variable: visit-count targets." v6 reopened it
   with *fresh data + structure* as the argument, not the recorded
   untested variable, and reproduced the closure: a from-scratch student
   lands 0.6–0.8/deal behind and PPO does not carry it across. The
   student had to re-derive five promotions' worth of RL from imitation
   of a noisy argmax teacher, and then be refined by a PPO regime tuned
   for a different starting point (F3).
3. **The bank was sized by records/param without a measurement.** F4's
   "records/param fell 4.9×" was reasoning, and the probe showed the
   truth was subtler: imitation keeps improving but strength doesn't.
   Records/param is the wrong unit; paired strength per doubling is the
   right one, and it was measurable for $0 before any of Stage 2 was
   spent (subsets of the existing v5 bank could have been probed).
4. **Single seeds.** Every Stage-3 arm and screen was one training seed.
   F6 says the screen ordering of a/b/c is unreadable. The register now
   carries the rule.
5. **The PPO ladder started with the wrong instrument** — train.py on a
   fixed config, gated against the v5 champion — for two trials before
   run_loop with candidate_lineage; and Stage 5 was tooling-blocked
   (no 882 in the C++ path) until mid-Stage-4. Both were caught and
   owned, but both were foreseeable at signing.
6. **A 6-hour fork bomb** (unguarded stdin pre-flight around match_eval)
   — launcher-discipline rules 4 and 10 recurred; run_match_gate.py is
   now the only sanctioned entry.

**What went right, and should be kept:** every stage halted or passed
by a rule written before the data; two amendments only, both
pre-unblinding; every instrument (obs v2, net, recorder, C++ path,
distill, probe driver) verified with A/A + identity + reproduction
tests — the S3 seed-20260812 probe run reproduced the frozen Stage-3
arm-a bit-for-bit; the champion was never at risk; the whole arc from
"maybe capacity" to "shelved on measurement" took five days and ~$96,
and the negative is clean enough to build on.

## 7. v6(2) — how I would design it if we came back

### 7.1 Principles (each from a measured failure above)

- **Extend, don't rebuild.** Start from the champion, add new inputs
  through **zero-initialized projections** (the verified match_proj /
  extended-v5 pattern — bit-identical at init), and let training move
  them. The RL maturity is kept; the from-scratch trap (F3, F5) is
  avoided; and the question "does this information help THIS net" is
  answered directly. Vehicle: **anchored PPO** on the champion (the
  league's KL-to-baseline machinery, dose from the measured curve) —
  not same-lineage distillation on own-search targets, which is closed.
- **One variable per A/B, paired, ≥2 seeds.** The Addendum-A instrument
  (neutral_raw with candidate and baseline both being the arms under
  test) gives paired per-deal strength at SE ~0.10 per 5,000 deals in
  ~11 min; the probe evaluator gives per-match paired imitation in
  seconds. Imitation is a *filter*, never a verdict (F4/F7).
- **Do not scale.** 2.55× bought nothing at this bank; strength saturated
  at 1.0–1.5M records. Any scale claim needs a new signal source, not
  more of the same teacher.
- **Measure information value before paying for it.** Two zero-GPU
  diagnostics (7.3 T0) rank candidates before any training.

### 7.2 What information I would try to add (ranked)

1. **Relative-frame deal scores and voids (fix the wart in place).**
   Rotate obs 156–159 and 165–180 to the acting seat, or append rotated
   copies via zero-init adapters. Cheapest possible change; tests
   whether per-opponent attribution alone moves anything on the
   champion (F8 left this open).
2. **Cross-deal opponent memory within the match.** Per relative seat:
   moons shot so far this match, moon attempts (deals in which that seat
   held all points taken through trick ≥6), points taken per deal
   history, deals led first. Rationale: the live exploit is a *repeated*
   attacker; the shooter clones in league training attack repeatedly;
   the champion has zero memory past match totals. New env bookkeeping
   in MatchEnv, ~12–16 dims, relative, appended. This is the one
   candidate that is genuinely absent from every prior net.
3. **Capture attribution + moon-alive + Q♠ status** (obs v2's core,
   re-tested additively on the champion, not from scratch).
4. **Trick-sequence tokens.** The last k tricks as (relative seat, card,
   position) tokens rather than per-card flags — gives attention the
   order information for signalling/inference. Structural, so it goes to
   the small-distill A/B ladder first (7.3 T3).
5. Derivable-but-explicit scalars (current trick winner, points on the
   table, tricks left, points left) — low expected value; include only
   if the T0 probes say the champion cannot decode them.

### 7.3 Test design — how to know which changes are useful

- **T0 — information audit (zero GPU, on the frozen champion).**
  (a) *Decodability probes:* linear/small-MLP probes on the champion's
  hidden states (global and per-card tokens) for each candidate feature
  — trick winner, per-seat points, moon-alive, opponent moon count.
  Features the champion already decodes are unlikely to pay when made
  explicit; undecodable ones are candidates. (b) *Disagreement corpus:*
  the states where the searched teacher's argmax differs from the
  champion's with a large equity gap (the search-vs-raw gap IS the
  headroom any information change could reclaim without lookahead —
  search halves moon concessions and adds +4.4 win-pts); bin them by
  feature (moon-alive, capture ambiguity, endgame, pass) — the
  roadmap's registered-but-never-run diagnostic. Where the champion's
  losses concentrate is where information can help.
- **T1 — the information ceiling (one distill).** A perfect-information
  policy net (true hands as extra inputs) on the same bank vs the raw
  net: the gap bounds what ANY input information can buy for the raw
  policy; if it is small, information is not the lever and v6(2)
  should not run.
- **T2 — additive A/Bs on the champion.** For each candidate block: zero-
  init adapter → anchored PPO from 8a89da90 (λ from the measured curve,
  drift band 5–15%), **2 seeds**, paired neutral-raw n=5000 vs the
  champion (Addendum-A instrument) + moon telemetry; standard NI + guard
  battery only on winners. Rank blocks by paired effect; combine only
  the winners; re-test the combination.
- **T3 — structural A/Bs by small-distill ladder.** Seat tokens vs FiLM
  vs none; distributional value head (placement/points distribution) vs
  scalar; categorical belief vs 3×BCE; trick-sequence tokens vs flags —
  each on the fixed bank, v5-M size, frozen recipe, ≥2 seeds, per-match
  paired imitation as filter, paired strength as verdict. Winners graft
  onto the champion via T2 where the surface allows, otherwise they wait
  for the next from-scratch generation *and* a new signal source.
- **Bands and halts written before data**, one pre-unblinding
  amendment, gates unchanged, instruments md5-frozen — the discipline
  that made this arc's negative cheap.

### 7.4 Costs (measured rows)
Distill 5.0 min/epoch on the full bank; paired strength n=5000 ≈ 11 min;
anchored-PPO trial ≈ 2.2 h at 25% headroom; match gate n=3200 32–34 min
(v5 vs v5); guard n=4800 ≈ 3.6 h. A T2 block = 2 seeds × 2.2 h + 2 × 11
min ≈ 5 h; T0/T1 ≈ a session + one distill.

### 7.5 What would justify starting v6(2)
Either T0/T1 showing concentrated, decodable-but-absent information at
the disagreement states with a non-trivial PI ceiling, or a new teacher
signal source (exploiter-league games qualify — the only source outside
every closure). Absent both, the champion's next strength comes from the
league (docs/exploiter_league_r4_prereg.md), not from a new network.
