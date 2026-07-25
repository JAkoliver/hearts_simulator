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

## Closed directions (measured; do not revisit without NEW evidence)

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
