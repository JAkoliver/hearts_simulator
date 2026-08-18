# Experiment rules — measurement and ops discipline

Hard-won rules. Each exists because its violation cost real time or nearly
produced a false conclusion. Datestamps mark the incident that taught it.

## Measurement

1. **Neutral opponents for every strength comparison.** Evaluating vs a
   net's own policy (or its training opponent) hands one side a perfect
   opponent model — measured flipping a +1.24 "regression" into parity
   (2026-07-15) and understating a raw gain 3x (2026-07-21).
2. **Paired duplicate deals, powered n.** The old n=600 search gate had
   ~25% power vs a true -0.3 and made "continual failure"
   uninterpretable (re-powered to n=2400, 2026-07-19). Quote SE and n
   with every verdict.
3. **Gate against an EXPLICIT snapshot path, never a mutable "current"
   file.** During PPO stages, hearts_model_final.pth IS the candidate;
   gating against it silently compares a net to itself (2026-07-23).
   **Symptom: SE of exactly 0.000 = you gated a net against itself.**
4. **Holdout splits must be by DEAL.** Per-record splits leak same-deal
   siblings (fake 0.99 EVs, 2026-07-15); per-file TAIL splits on
   many-small-file banks keep only late-trick records (fake 0.099 BCE,
   2026-07-23).
5. **Telemetry informs, never gates.** Every promotion candidate gets an
   analyzer-style behavioral diff vs baseline (lite per-trial, full on
   promotion), appended to telemetry history; large deltas are FLAGGED
   in reports and the ledger. Tactical metrics are heuristics — the
   2026-07-23 comparison showed the strongest model deliberately scores
   worse on several "textbook" rates — so promotion stays score-based;
   gating on telemetry would Goodhart play style. (Telemetry found the
   2x moon-concession hole that a week of scalar gates missed.)
6. **Promotions only through the gated drivers** (orchestrator /
   promote_raw_line.py): baseline copy + Hall of Fame milestone +
   optimizer carry-through + trace re-export, hash-verified.
7. **Durations are quoted from docs/speed_ledger.md only**, same net
   generation, same convention (A/B vs steady — never mixed).

## Ops (see also launcher-discipline memory)

8. Unbuffered (`python -u` + PYTHONUNBUFFERED=1 for children), output to
   a FILE, never through pipes (grep/tee buffer silently). Launch via a
   dedicated script file; NEVER `chain && cmd &` (backgrounds the whole
   chain — violated again twice on 2026-07-21/23; the two-step "write
   script, run script alone" pattern is mandatory).
9. Multiprocessing-pool scripts live in the repo with the __main__
   guard (Windows spawn re-imports __main__; a scratchpad driver
   crash-respawned workers for 2.3h, 2026-07-18).
10. Every long-running launch gets a monitor on milestones AND a
    watchdog at ~2x the ledger estimate. Watchdog grep patterns must
    not collide with normal output (an "ABORT" pattern matched
    "PROMOTION ABORTED", 2026-07-22).
11. End-time-capped runs are sized from the VERIFIED launch clock, plus
    a hard-stop guard that kills at a chunk boundary (2026-07-22).
12. After any hard kill of the orchestrator: hearts_model_final.pth may
    be smoke-test-polluted; md5-verify against baseline temp / last
    milestone before trusting it.
13. **No cloud spend of any kind without explicit user approval of a
    specific rental.**
14. **TRAIN IN CLOUD, EVALUATE LOCALLY** (2026-07-24, supersedes any
    hardware-matching rule): checkpoints may be produced anywhere;
    every comparison that decides anything runs on ONE local machine
    with both arms sharing hardware. Never split a paired comparison
    or a single dataset generation across heterogeneous hardware.
    Every cloud-trained checkpoint is evaluated locally against a
    locally-run baseline before any promotion decision. (The 2026-07-17
    H100 equivalence failure was portable MEASUREMENT, not portable
    training.)

15. **Search standard = K=64 base + K=256 endgame** (adopted 2026-07-25:
    --k-endgame 256, triggers when any seat >= 85). Measured cost +37%
    per match (19.2 -> 26.2 s/pair); confident-flip rate in tension
    2.6x'd vs flat K=64. All match-level search comparisons run BOTH
    arms on this schedule; the per-deal search guard is unaffected (no
    match context there).

16. **Match-aware search is the CEILING CONFIG (validated 2026-07-27):**
    equity leaf scoring (hearts_equity.pt) + K=64/256 schedule on the
    556-ctx search trace. N=8000 cloud validation vs the frozen
    match-blind reference: +4.44 win-pts (SE 0.68), McNemar one-sided
    p~5e-11, all 8 shards positive; converts P2->P1 AND P2->P4 (mean
    place WORSE) = win-equity optimization, not generic strength.
    Consequences: (a) the search GUARD runs BOTH arms match-aware
    (556 traces + equity leaves; single-deal context so K stays 64 -
    the endgame path is validated separately); (b) match-level
    comparisons follow rule #15; (c) the Phase 2 teacher is
    match-aware search. Fleet instrumentation rule: match CSVs now
    carry stratum flags (tension/max85 reached, tension-deal count) -
    never run a validation-scale measurement without the columns the
    pre-registered analysis needs.

## Closed directions (measured; do not revisit without NEW evidence)

**What counts as admissible new evidence (2026-08-01).** A closed
direction is a powered, mechanism-attributed measurement, not a mood.
Reopening one requires naming which recorded premise no longer holds.
Admissible: (a) a change to a component the closure's mechanism
depends on (the PPO-on-MLP closure - critic EV stuck at 0.30 - did
not survive the v5 architecture and was correctly reopened, first
promotion 2026-07-15); (b) variation of a variable the closure itself
recorded as untested (e.g. visit-count targets on the imitation
closures); (c) a measured diagnostic contradicting the closure's
mechanism. Inadmissible by default: more compute, more data, more
seeds, or a rerun of the same recipe - closures are powered
measurements, so a re-roll only exploits variance. ("More data" IS
admissible exactly when the closure recorded quantity as the untested
variable.) Every reopening gets its own pre-registration.

- PPO fine-tune of near-ceiling nets: one-shot; 255k further games flat
  (2026-07-23). PPO gains scale with distance from the ceiling
  (-0.9 fresh distill, ~0 at ceiling).
- Same-lineage distillation of own-search targets: degrades the student
  at 3.5k AND 12.5k fresh deals, soft or sharpened (sharpen saturates
  at 2.0; 4.0 worse without entropy collapse).
- Reaching a LARGER net by imitation of existing teachers, any data mix
  (v5-L +3.3..+5.3, 2026-07-22/23). Untested variable: tree-search
  visit-count targets.
- Search amplification past the net's ceiling: K>64, ISMCTS, learned
  leaf evaluation (both variants).
- Anchored supervised imitation of search-defender decisions AT THE
  ~60k-decision corpus scale (2026-08-09, exploiter round 2,
  docs/exploiter_league_r2_results.md): both drift-screened candidates
  dead-null at the defense gate (+0.016 / +0.094 moons/match, p=.56 /
  .80) despite a real, large teacher behavior gap (search defenders
  concede 50% fewer moons). Untested variables recorded: corpus scale
  (rerun must pre-register a mid-training defense probe - see the
  results doc's containment-tension note), non-argmax teacher signals
  (visit counts, sequence-level), aimed-RL vehicles.
- Equity-scored-target distillation, ANY recipe (2026-08-05, expert-iter
  v2 comparative stage, docs/expert_iter_v2_results.md). Binary
  confidence-filtered hard-CE at the strongest health-passing anchor
  (lambda=4.0): ALL five compositions degrade placement +0.10..+0.16
  (hier. SE 0.003-0.017; one-sided max-T p_adj=1.0; 6400 CRN units x 2
  blocks). Continuous-certainty weighting (erf(z/2), same lambda; incl.
  2x enrichment and 2x data arms): statistically identical to baseline
  (|delta| <= 0.009, p_adj 0.81-0.99; registered enrichment and size
  contrasts both null). MECHANISM: refutes the v1 noise hypothesis -
  the harm lives in the confident teacher signal itself, and
  neutralizing noise converges to no-change, so the equity-teacher
  ordering signal is inert for match play. Lifeline branch (prereg
  2026-08-02) not triggered: no continuous arm helpful at familywise
  p<.05. Per the prereg stop rule the closure covers binary AND
  continuous variants; v1+v2 are two independent designs with the same
  verdict. Untested variables: none within the recipe family (teacher
  = search-equity ordering; a future teacher with a DIFFERENT signal
  source, e.g. visit counts or exploiter-league games, is outside this
  closure).
- Visit-count-target distillation at 7.6M (2026-08-11, Phase 2,
  docs/phase2_visitcount_results.md): both registered picks WORSE
  (ep3 match +0.083 p~1.0 + guard UB +0.427; ep2 +0.042 p=.997 + guard
  UB +0.318 vs +0.3), dose-response consistent with the mechanism -
  visit distributions encode where search LOOKED, not what it endorsed
  (entropy 0.43 -> ~1.0 onto explored-but-rejected moves). With
  expert-iter v2, BOTH available encodings of the searched teacher are
  measured non-improving at 7.6M. Untested variable: a genuinely new
  signal source (exploiter-league games qualify).
- Mutation-chain match-PPO on a FRESH, far-below-ceiling distill
  (2026-08-13, v6 Stage 4, ledger + docs/v6_next_plan.md): three
  consecutive run_loop trials from the champion-regime config
  neighborhood (lr pinned ~1e-5) each made the candidate WORSE than its
  own lineage baseline (+0.296 -> +0.167 -> +0.080 placement, SE ~0.021;
  monotone mitigation, never crossed; moon-shot suppression every
  trial). The sanctioned retry shape is docs/v6_next_plan.md Path C
  (KL-anchor-to-distill-init + registered lr x lambda factorial), not
  more mutation roulette. Untested variables: regime-appropriate
  recipes (Path C), and a non-starved starting distill (Stage-2b).
- CONDITIONAL: scale-without-data (2026-08-13, v6 Stage 3): arm a
  (19.37M, 2.55x the champion) TIED arm b (v5-M architecture + obs-v2
  channels + aux heads; the a-vs-b contrast is ~2.4x) on imitation from
  the 1.52M-record bank - capacity idled.
  Closure is conditional on that bank size; records-per-param fell
  ~4.9x vs the 2.93M bank that built v5-M. Per the admissibility rule
  above, "more data" IS the recorded untested variable here: the
  data-scaling probe (v6_next_plan Step 1) is the registered reopening
  test, and a data-bound verdict reopens scale via Stage-2b.
  **REOPENED 2026-08-15: probe verdict DATA-BOUND** (docs/
  v6_data_scaling_prereg.md §9) — imitation improves at every measured
  size, slope diminishing; Stage-2b proposable. Closure stands ONLY for
  the 1.52M bank. **THEN Addendum A (2026-08-16, paired strength,
  n=5000 x 4): +49% data bought NO detectable strength (S3-S2 pooled
  +0.018, SE 0.074, UB95 +0.163) while the control step did (-1.20,
  SE 0.08). Registered consequence: Stage-2b NOT PROPOSED; v6 SHELVED
  PENDING A NEW SIGNAL SOURCE.** Recorded untested variables: Path C
  (regime-appropriate PPO on the existing distill, own prereg); a
  teacher with a different signal source. Seed rule from the same
  record: >=2 training seeds per arm for any distill strength
  comparison (same-size seeds differed by 0.30-0.36/deal).
- Anchored placement-reward match-PPO for MOON DEFENSE on 8a89da90 at
  in-band drift (2026-08-17, league round 4, docs/exploiter_league_r4_
  results.md): 2x2 lambda {0.05,0.02} x shooter share {0.15,0.25} plus
  block-credit reward shaping b {2,4} - six trials, none gate-eligible
  on the validated fast defense probe (SE ~0.037). The lambda=0.05
  family is a stable ~-0.065 +/- 0.019 moons/match across four seeds
  (b=0/0/2/4): real, sub-bar, unmoved by shaping, density or anchor
  dose. Closes: anchored placement-reward PPO at 5-15% drift, shares
  <= 0.25, block credit <= 4 for defense on THIS baseline's inputs.
  Untested variables recorded: seat-attributed threat information
  added to the champion via zero-init adapters (obs v2) before anchored
  PPO; sequence-level league-game teacher signal; RL exploiters.
- Zero-init obs-v2 threat-information adapters trained from zero INSIDE
  the champion-regime anchored PPO (2026-08-18, league round 5, docs/
  exploiter_league_r5_results.md): E - C = +0.022 (SE 0.038) on the fast
  defense probe; the adapters never became a pathway (mean |w| ~30x below
  the trunk's at lr 9e-6 under the anchor). Closes that DELIVERY METHOD
  only. The information hypothesis is supported by T1 (obs-v2 students
  out-defend obs-v1 students ~4 SE) and remains open; recorded untested
  variable: adapter WARM-START — RUN 2026-08-18 (r5 Addendum W) and
  HALTED at acceptance: adapters pretrained with the trunk frozen
  disturb the champion at every dose tried (drift 34-36% vs a 3% bar;
  smaller adapters = more disturbance). Additive obs-v2 adapters on
  8a89da90 are closed by both delivery methods. Still open: adapters
  trained jointly with an anchored trunk fine-tune; a from-scratch
  obs-v2 generation (T1: obs-v2 students out-defend obs-v1 ~4 SE). Also on record: T0 audit -
  the champion already decodes moon-alive well (AUC 0.975 given points
  taken) but not per-seat points (R^2 0.62) or per-card capture (probe =
  raw channels).

## Strategy (2026-07-23): see docs/ROADMAP.md

Match objective first, prove the self-improvement loop at 7.6M, scale
only inside a working loop.

17. **Clock discipline + PC usability windows (2026-07-29).**
    (a) ALWAYS read the actual clock (`date`) BEFORE quoting any ETA,
    sizing an end-time-capped run, or scheduling work - never estimate
    from a stale planning-time assumption (bit us on installment 2's
    6:45pm miss and again on the 07-29 expert-iter overnight quote).
    (b) The PC must be USABLE 07:00-22:00 and 19:00-02:00 - i.e. the
    only unrestricted full-throttle window is 02:00-07:00. Outside it,
    GPU-saturating work (generation, unheadroomed training) needs the
    gentle profile (reduced threads, below-normal priority, headroom
    pacing) or explicit per-session user permission.
    (c) Suspension does NOT free VRAM - a paused process holding ~24GB
    still starves the compositor (measured 07-29). To make the PC
    usable, kill (records/checkpoints flush per unit) or run gentle;
    PAUSE_AI alone is not enough for VRAM-heavy processes.
