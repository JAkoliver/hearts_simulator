# Phase 2 pre-registration — visit-count expert iteration at 7.6M
# (+ registered side-probe: anchored-PPO lambda=0.05)

Status: DRAFT — awaiting user signature. Nothing below executes until
signed. Baseline: 8a89da90 (5th match-era promotion, unchanged since
07-29).

## Why this, why now (the converging evidence)

Phase 2 is docs/ROADMAP.md's own make-or-break: "one demonstrated
generation of: search with current net -> richer targets -> train ->
measurably stronger at MATCH play", with visit-count targets named the
key untested variable. Three independent closures now point at it:
1. v5-L (07-22/23): imitation cannot reach a bigger net; closure
   recorded tree-search visit-count targets as the untested variable.
2. Expert-iter v1 post-mortem (07-31): search-teacher distillation
   needs targets that ENCODE PREFERENCE STRENGTH; v2 (08-05) closed
   equity-scored targets entirely, binary and continuous.
3. League r2 autopsy (08-09, diag 58685d1): per-decision argmax
   agreement is a dead currency in BOTH directions. Visit
   distributions are precisely a preference-strength encoding.
v6 stays gated: its trigger requires a compounding loop, and no loop
compounds today. Phase 2 is the experiment that either produces one or
proves the 7.6M net cannot host one.

## Stage A — instrument build (halt-default verification)

TreeSearchPlayer (built 07-15; PUCT+Max^n, c_puct 1.5 measured
optimal; kept for exactly this purpose) needs two integrations:
- MATCH-AWARE leaf scoring: the equity-leaf path flat search uses
  (rules #16 package), applied at tree leaves; per-acting-seat 556 ctx.
- Visit-count RECORDING: a generation driver emitting per-decision
  records (obs[556], legal mask, visit counts over legal actions,
  chosen action, seat, match id) with the same per-deal flush +
  kill-anytime/trim-resume contract as the r2 recorder.
Verification before any data counts (all halt-default):
- A/A determinism: same seed + fixed thread count twice -> identical
  records.
- Strength sanity on CRN deals vs flat K=64 at the CHOSEN budget:
  tree arm within +0.5/deal of flat (it converges to flat's level;
  materially worse = integration bug).
- Recorder self-consistency: chosen action = max-visit action;
  actions in mask; visits sum to iteration budget.

## Stage B — teacher-signal probe (cheap, decides everything)

Before mass generation, ~500 recorded decisions on ~40 natural deals:
- Visit-entropy distribution: registered HALT if the median visit
  distribution is effectively one-hot (>=90% of visits on the argmax
  for >=80% of decisions) - that would be the SAME dead currency as
  r2, just more expensive. Also HALT if near-uniform (no preference
  signal): median top-1 visit share < 0.35.
- Preference-strength validity: Spearman correlation between visit-
  share gap (top1 - top2) and flat-search value gap on the same
  decisions (CRN) >= 0.3. Visits must track value structure, not
  exploration noise.
Numbers in this stage may be re-banded by ONE pre-unblinding amendment
(the v2 freeze pattern) if the shapes are healthy but the guessed
thresholds are mis-set; the one-hot HALT is not amendable.

## Stage C — generation (sized by measurement, not hope)

- PACE PROBE first: 20 deals at iteration budgets {200, 400, 800},
  local; tree search is 15-55x flat cost at 1600 iters, so budget
  choice is a measured cost/signal trade. The budget used for the bank
  is chosen ONCE from the probe (signal metrics from Stage B at each
  budget) and recorded.
- Bank: natural self-play matches with the CURRENT baseline in all
  seats, target >= 150k play-decision records (every decision
  recorded, not confident-filtered - the filter WAS the r2 mistake);
  fresh seed block 210,000,000+ (stride 1M; audited disjoint from
  20-179M, 190M, 200M, 520/620M, 720-722M).
- All local by default under rule #17 pacing; the certified cloud
  path (H100+AOTI) may be proposed if the pace probe makes local
  sizing infeasible - separate per-rental user approval (#13).
- Passing decisions are NOT in scope (tree search has no pass
  integration; flat rewound-pass targets stay a recorded limitation).

## Stage D — distillation (recipe freeze, then candidates)

- Warm start from the 8a89da90 milestone (md5-verified in-harness; the
  working file holds a rejected candidate and is never trusted).
- Loss on play decisions: KL(student || visit distribution) - soft
  targets, never argmax CE. Value/belief heads train on match-outcome
  targets carried in the records (roadmap: "+ match-outcome value
  targets").
- Recipe freeze on holdout ONLY (v2 pattern, freeze-only seeds):
  epochs {1,2,3} x lr {1e-5, 3e-5}, entropy diagnostic (candidate
  entropy within 2x baseline both directions), teacher-KL on holdout.
  At most 4 candidates trained; freeze picks <= 2 for gates.
- Registered telemetry (informs, never gates): moon-defense holdout
  numbers from the b2 banks - IF visit-count distillation moves
  defense for free, that is round 4's opening, not this prereg's
  claim.

## Stage E — gates (the standard promotion battery, unchanged)

Match gate n=3200 alpha=0.05 (placement) + search guard n=4800 K=32
UB <= +0.3, exactly the orchestrator's standing promotion path.
PASS = promotion AND the Phase 2 loop question gets its answer
observed: re-run generation with the promoted net (one iteration) to
test COMPOUNDING - a second-generation gate pass is the roadmap's
"loop works" declaration and unlocks Phase 3/v6 sequencing.
FAIL on both candidates = visit-count targets close for this net;
combined with rounds 1-3, the capacity conversation (v6) is then
evidence-backed.

## Registered side-probe (independent): anchored PPO at lambda=0.05

Round 3's recorded untested variable, run once alongside Phase 2
(GPU-idle time), EXPLORATORY:
- One full-length anchored-PPO training, lambda=0.05, seed fresh;
  measure ordinary drift + defense-holdout (r3 harness unchanged).
- If drift lands in the 5-15% band: run the search guard n=4800 on it
  - the guard-tolerance-vs-drift curve point rounds 1-3 never
  measured. NOT candidate-eligible; gating beyond the guard datum
  would need separate authorization.
- Any outcome feeds (or kills) a possible round-4 dose-scheduling
  prereg. Cost: ~2.5h train + ~2.7h guard, bounded.

## Cost (rule #17)

Stage A build: session work. Stage B probe: <1h. Stage C: pace probe
~1-2h, bank hours-to-days depending on measured budget (quoted from
the probe before launch, per launcher discipline #3). Stage D:
minutes per candidate. Stage E: ~1.5h match gate + ~2.7h guard per
candidate. Side-probe: ~5h bounded. No cloud spend without separate
approval.

## What results are allowed to mean

- Stage-B HALT (one-hot visits): tree-visit targets share argmax's
  dead-currency failure at this budget - closes the formulation
  cheaply, before generation spend.
- Gate FAIL both candidates: visit-count distillation closes at 7.6M;
  Phase 2's answer is "no known target formulation compounds", which
  is the evidence the v6/capacity decision has been waiting for.
- Gate PASS + second-generation PASS: the loop compounds; Phase 3
  sequencing (scale inside the loop) and the league round-4 rider
  (visit targets on the defended corpus, r2's recorded reopening)
  both unlock, each behind its own prereg.
