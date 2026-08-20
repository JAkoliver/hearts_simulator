# The gated-ensemble program — design space, plan and discipline (2026-08-18)

Status: PROGRAM DOCUMENT (routing + design record), not a prereg. Every
experiment below runs under its own signed pre-registration
(docs/experiment_rules.md). Written after the hybrid probe, the gate
ladder and the r6 signature; read this before proposing any ensemble
work. Companion records: docs/hybrid_specialist_probe.md (the probe, the
NI pass, the ladder), docs/exploiter_league_r6_prereg.md (the first
ensemble battery), docs/exploiter_league_r4/r5 results (why the champion
itself could not be moved), docs/v6_postmortem.md §7 (why "extend, don't
rebuild" and why single seeds mislead).

---

## 1. What it is, and what to call it

A **gated ensemble of raw policies**: a DEFAULT net plays every decision
except those a ROUTER hands to a SPECIALIST. Two frozen nets and a gate,
exported as **one TorchScript module** with 882-dim (obs v2) inputs — so
to the engine, the guard, the gates and the site it is a single raw net
(one or two forward passes per decision, millisecond latency, no search).
"Hybrid" is the shorthand in filenames; "gated ensemble" in prose. Every
gate uses PUBLIC information only (obs v2 is all public — v6 Stage 0).

Why this exists: two rounds measured that the champion 8a89da90 itself
cannot be moved on moon defense in-band (r4: a stable ~−0.056 moons/match
across six seeds regardless of reward shaping, threat density or dose)
and cannot be given new inputs additively (r5: zero-init adapters never
become a pathway; warm-started adapters disturb the trunk at any dose).
Meanwhile a from-scratch obs-v2 student (arm b) defends better than the
champion (T1, −0.117) but is 0.6/deal weaker overall. The ensemble takes
each net where it is good.

## 2. Instance on the table (round 6 candidate)

- default = champion 8a89da90 (556 prefix of obs v2);
- specialist = v6 Stage-3 arm b (v5-M architecture on obs v2 + aux heads,
  from-scratch distill of the searched champion, single seed);
- router = the specialist's OWN aux moon head: switch when
  max_opponent sigmoid(moon logit) > 0.1 and that opponent is moon-alive
  (~9.6% of play decisions).
Measured (docs/hybrid_specialist_probe.md): defense −0.247 ± 0.032
moons/match vs the champion's defenders (clone attacker) — twice arm b
alone, ≈ the r1-t3 pass — at +0.014 ± 0.031/deal (no measurable strength
cost); the cruder ≥1-point-rule variant passed match NI (UB95 +0.017).
Battery in progress (r6): NI → SEL defense gate n=320 → search guard.

## 3. Evaluation currency (settled; do not relitigate per experiment)

- Cheap loop (estimation, hours → minutes): the validated **fast defense
  probe** (n=1,000 CRN vs the champion's defenders, clone SEL attacker;
  SE ≈ 0.03–0.04) + **paired per-deal strength** (neutral raw, n=5,000,
  SE ≈ 0.03) on the SAME candidate. Both informs-only.
- Claim instruments (the standing battery, conjunctive, halt-default):
  **NI** match placement vs mixed anchors (UB ≤ +0.030); **SEL defense
  gate** n=320 vs the search shooter (fewer moons, α=.05); **search
  guard** n=4,800 (UB ≤ +0.3) with the ensemble as the search's raw
  policy. Deal points are reported, never gated — a defender pays points
  to remove 26-point tails; the champion's own promotions traded points
  for placement.
- **GATE-FIRES CHECK (rule from the retracted r6 gate, 2026-08-19):** the
  r6 defense-gate "null" was an artifact — the engine's 882-defender path
  zero-filled the obs-v2 extension, so the router never fired and the
  gate replayed the champion (byte-identical candidate CSVs across
  different candidates). Engine fixed (MatchRawPolicy assembles 882).
  RULE: before any C++-side measurement of an ensemble, verify its
  output CSV DIFFERS from the default net's on the same seeds; clone
  probes remain estimation-only and the search gate remains the claim.
  The attacker-transfer question (clone vs search shooter) is OPEN again
  — the AGG diagnostic showed all candidates defend both clones easily,
  and no valid search-side ensemble measurement exists yet.
- Registered secondary for ensembles: placement in a shooter-mixed field
  (SEL clone in 1/8 of matches) — where insurance is supposed to pay.
- Seed rule: ≥ 2 independent training runs for any specialist strength
  comparison (register 2026-08-16); an ensemble of frozen nets has no
  seed of its own, but its specialist is one draw of a recipe — say so.
- Selection provenance: routers/specialists chosen on the fast probe use
  seed block 740M; batteries use other blocks (720M/724–731M for the
  gate; NI/guard their own), so ladder selection cannot leak into gates.

## 4. The design space

### 4.1 Routers (state → which net plays)
1. **Hand rules** on public state (threat = opponent moon-alive & ≥ N
   points; any-alive; trick index; Q♠ status). Measured: the ≥1-point
   rule is the WORST gate (fires in low-value states, +0.10/deal); ≥3/6/10
   defend more for ~0 cost. Ceiling = what a rule can express.
2. **The specialist's own detector** — arm b's aux moon head thresholded.
   Measured best (τ=0.1: −0.247 at ~0 cost; τ=0.3: −0.182 at −0.03).
   A stronger/different specialist needs τ re-tuned.
3. **Cheap refinements** (each ~20 min on the ladder): soft mixture of the
   two probability vectors near the threshold; hysteresis / commit for
   the trick or until the threat dies (avoid incoherent mid-trick plan
   switches); a match-state veto (near-100 seats); champion-uncertainty
   conditioning (measured DEAD as a stand-alone gate: +0.026 defense,
   +0.049 cost).
4. **A dedicated router trained on search-judged labels**: for a state,
   the searcher (probe log) rates every legal action; label = which net's
   action it rates higher; train a small classifier on the champion's
   global token + the specialist's aux outputs. This uses search as a
   JUDGE of a choice between fixed policies — a different mechanism from
   the closed distillation directions (imitation targets for a policy).
   ~3 h of search to label ~20k states, minutes to fit; validate on the
   cheap loop before any battery. The right tool for "is the switch worth
   it for ME, here" (the 4-player public-good subtlety, §5).
5. **k-way routing** once several specialists exist (MoE); each new
   domain first needs a T0/T1-style audit: is there a specialist that
   beats the default there, and is the domain recognizable from public
   information.
6. RL-trained router (2-way action, sparse reward) — held in reserve;
   the labeling route is cheaper and less noisy.

### 4.2 Specialists (the expert that plays gated states)
1. **Specialist ladder with the router fixed** (~20 min per candidate; no
   training). The router and the specialist are DECOUPLED: arm b's moon
   head can route any net. Existing candidates: r1-t3 (the strongest
   defender the league produced, −0.235; failed guard/NI as a WHOLE net,
   but a specialist on ~10% of decisions confines its drift to the
   states where it is good), arm a (19M obs-v2 student, never probed for
   defense), r4/r5 candidates, other arm-b epochs.
2. **Train the specialist for its role** — PPO on the ensemble with the
   DEFAULT FROZEN: gradients reach only gated decisions; no anchor needed
   on the specialist (its drift is confined by the router); shooter
   clones in the pool; placement reward. Sidesteps the drift/guard
   problem structurally because the champion never changes; the
   specialist can be trained unanchored, r1-style. This is the first
   TRAINING step of the program (round 7), whichever way r6 lands on the
   defense axis. Readouts: cheap loop; battery on the winner. Also
   candidates for a specialist-side aux moon head so router and
   specialist can be re-tuned together.
3. **Purpose-built specialists**: distill the SEARCH DEFENDER's
   threat-state decisions into a small net (r2's corpus and recipe
   exist; r2 failed as a whole-net candidate under drift constraints,
   which a specialist does not have), or a from-scratch obs-v2 net
   trained only on threat states. Larger build; only if 1–2 plateau.

### 4.3 More specialists / more domains
Passing, endgame, under-attack, match-endgame (near 100). Each domain:
audit first (T0-style probes on the champion; disagreement corpus vs the
searcher; T1-style "does a student beat the champion there"), then a
specialist ladder, then a router. Composition is not free: every
addition changes the state distribution the others see — re-measure the
ensemble as a whole after each change; the battery always runs on the
ensemble as ONE module.

## 5. The four-player subtlety (recorded reasoning)
Blocking a moon is a public good among three defenders: the blocker pays
points, the other two benefit; the right blocker is state-dependent
(someone may already be forced to take the trick; someone near 100 must
not). We do NOT hand-design this. It lives in the SPECIALIST's policy
(which sees voids, plays, the partial trick, deal points and match
totals) and is learned in self-play where all defender seats run the
same ensemble; the search-judged router (4.1.4) captures the individual
side ("worth it for me here"); and outcomes are judged COLLECTIVELY
(the defense gate measures the table of three defenders) plus per-
defender placement.

## 6. Serving and product implications (recorded, not research)
A promoted ensemble is served as one 882-input traced module; the raw
play path must feed obs v2 (the site's engine already assembles 882 for
search); the router runs first so only the selected net is evaluated on
gated decisions. Search mode may keep the champion as its rollout
policy if the guard prefers it (search sees the threat itself) — a
deployment decision registered per round, not an override of the guard.

## 7. Sequencing (as of 2026-08-18)
1. r6 battery on the current instance — DONE 2026-08-19: NI PASS, defense
   gate NULL vs the search shooter (attacker-transfer failure); see
   docs/exploiter_league_r6_results.md.
1b. Attacker-transfer diagnostic on the current ensemble (AGG clone probe,
   ~25 min) and the specialist ladder with r1-t3 (search-validated
   defender) under the same router, each with a search-SEL n=64 check.
2. Specialist ladder under the moon-head router (r1-t3, arm a, arm-b
   epochs) — an afternoon, cheap loop only.
3. Router refinements ladder on the best specialist — an afternoon.
4. Round 7 prereg: specialist PPO with the default frozen (4.2.2).
5. Round 8: search-judged router (4.1.4) if the fixed router is the
   limiter; then further domains (4.3) behind audits.
Ops: cheap loops at ≤ 8 workers BelowNormal in the day; batteries at
night (defense-gate candidate arms 4-wide from 00:00; guards ~4.5 h);
never stack > 3 arms per probe call (2026-08-18 incident).

## 8. Honest limits
An ensemble is a better RAW PLAYER (the project's goal as stated
2026-08-18), not a better single network; folding it back into one net
would be a distillation from a stronger, different-lineage teacher and
needs its own registration against the closed distill directions. The
clone-attacker probe may overstate defense against the search attacker
(base 1.48 vs 2.48) — r6's defense gate is the first check of that for
ensembles. Router quality is specialist-relative. Serving cost grows
with specialists unless the router runs first.
