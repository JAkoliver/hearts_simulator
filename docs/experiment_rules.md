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
