# v6 aftermath — decision record and forward plan

Written 2026-08-14, immediately after the Stage-4 halt. This is a DECISION
RECORD and routing plan, **not a preregistration** — every path below
executes only under its own signed prereg (docs/experiment_rules.md), and
nothing is launched as of this writing. A new session should read this
before touching any v6 or league work.

Companion records: docs/v6_prereg.md (the concluded v6 prereg),
docs/speed_ledger.md 2026-08-13 entries (Stage-3 freeze/screens, Stage-4
trials 1-3 and the halt), equity_data/verdicts/.

---

**STATUS 2026-08-18 — this plan's routing is superseded for the main
line:** v6 shelved (data probe + Addendum A); league r4 (anchored PPO,
reward shaping) and r5 (threat adapters + warm-start) concluded without
promotion; the main compute program is now the GATED-ENSEMBLE program
(docs/gated_ensemble_program.md; r6 battery in progress). Path C stays
recorded as the only v6 continuation.

## 1. Where things stand (verified 2026-08-13)

- Baseline/champion **8a89da90** (milestone 1785322724, 5th match-era
  promotion) — md5-verified intact after the ladder's hard kill; served
  weights identical. `hearts_model_final.pth` holds an UNPROMOTED trial-3
  candidate (a9653255) — never serve it.
- **Stage 4 HALTED by the registered stop rule**: three consecutive failed
  trials under `run_loop` match-mode with `candidate_lineage: true`, each
  gated n=3200 against its own v6 lineage baseline:

  | trial | placement Δ (SE) | win% A/B | neutral raw (p) |
  |---|---|---|---|
  | 1 | +0.296 (0.022) | 35.2 / 46.4 | +0.630 (1.000) |
  | 2 | +0.167 (0.022) | 40.8 / 48.4 | −0.181 (0.127) |
  | 3 | +0.080 (0.021) | 44.1 / 48.1 | +0.187 (0.872) |

  Positive = candidate worse. Trial-to-trial differences are ~4-6 SE: the
  mitigation trend is real, but no trial came close to passing, and Stage 5
  was never entered.
- The Stage-4 amendment is moot: amendments are pre-unblinding, and Stage 4
  is fully unblinded. Any continuation is a NEW prereg.
- **The Stage-5 C++ blocker is CLEARED** (easy to miss reading only this
  doc): the obs-v2 search path was built and verified 2026-08-13 —
  ProbeObsDim speaks 882, FillObsRow assembles the v6 vector
  byte-identical to training records, A/A determinism + v5 byte-identical
  regression + end-to-end pipeline all PASS (ledger entry same date; two
  latent memory bugs found and fixed in the process). A v6 candidate CAN
  face the registered search guard; nothing in Stage 5 is
  tooling-blocked.
- The xinxin external benchmark is a separate, staged program (plan lives
  outside this repo with the eval workspace); it runs off frozen champion
  traces and does not interact with anything here.

## 2. What the v6 arc established (findings)

**F1 — Structure helps at fixed scale.** Arm b (v5-M architecture +
obs-v2 channels + aux heads) vs arm c (plain v5-M): CE 0.838 vs 0.868,
teacher-match 64.2% vs 63.0%. The obs-v2 information and aux heads are
real improvements to imitation.

**F2 — Scale alone bought nothing pre-PPO.** Arms a and b TIED on
imitation (a slightly worse per-deal on the screen, within noise).
Multipliers pinned (the two figures in circulation have different
denominators): arm a is 19.37M = **2.55×** the 7.6M champion (the
prereg's "~2.6×"), and **~2.4×** arm b, whose obs-v2 card channels +
aux heads sit slightly above plain v5-M. The a-vs-b tie is the
scale-isolating contrast. Consistent with a saturated training bank
(see F4).

**F3 — Match-PPO from a fresh distill, in the champion's config regime,
actively damages the net.** All three trials trained in a tiny config
neighborhood inherited from `config_chain_base.json` (lr pinned
1.0-1.2e-5, eps_clip 0.199, entropy_coef ~0.002) — a regime evolved for
gently refining the RL-MATURE champion at its ceiling. A fresh distill
0.83/deal below the ceiling is a different training regime, never before
attempted with match-mode reward. Telemetry signature: the candidate shot
FEWER moons than its baseline in every trial (492/571, 404/543, 528/556)
— aggressive teacher behavior erodes before reward signal replaces it.
This is the same drift class the league r2/r3 work measured and built
tools against.

**F4 — The untested confound: the bank may be too small.** v6 (19.37M
params) was distilled from 1.52M records. v5-M (7.6M params) was built
from **2.93M records**. Records-per-param fell ~4.9×. The a-b tie is
exactly what a saturated bank looks like (extra capacity idles), and
"data is the constraint" is already the registered lesson of the v5-L
closure. Nobody has measured whether 1.52M fresh records saturates even
the structured 8M net, let alone 19.4M.

**F5 — League round 4: each individual blocker has a measured answer,
but the JOINT result is not yet measured (corrected 2026-08-14 — the
first draft oversold this as "fully de-risked").** The chain:
- r1 trial 3: the ONLY defense-gate pass ever (−0.250 moons/match,
  p=0.029) — defense is teachable by PPO. But that was at λ=0 (22.4%
  unaimed drift), and it failed promotion there (search guard UB
  +0.453; match non-inferiority UB +0.034 vs +0.030).
- r3: the aimed-drift tool exists (KL-to-baseline on threat-dead states,
  in train.py); halted only on dose calibration (2 shots missed the
  5-15% band).
- Phase-2 side probe: the dose is found — λ=0.05 → 7.77% drift, in-band
  (curve ~20%/λ=0, 7.8%/0.05, 3.85%/0.4, 2.84%/1.0), and the guard on
  that candidate PASSES with margin (UB95 −0.087 vs +0.3, point
  estimate −0.27; verdicts/r3_probe005_guard.json).
- **The missing joint datum:** the same λ=0.05 candidate's DEFENSE gate
  was NULL — +0.016 moons/match, p=0.56, pass:false
  (verdicts/r3_probe005_defense_gate.json). So "defense pass" (λ=0)
  and "guard pass" (λ=0.05) have never been achieved together. The
  probe was underpowered (n=64 paired, SE 0.106: ~70% power against
  r1-t3's −0.250), so the null is weak evidence either way — but Path
  D's premise is a REASONABLE BET that teachability survives the
  in-band dose, not a measured fact. The registered mid-training
  defense probe in the r4 design is precisely the test of that bet.

## 3. The routing decision (agreed 2026-08-14)

```
Step 1  DATA-SCALING PROBE ($0, ~afternoon, own mini-prereg)
        │
        ├─ data-bound (slope still steep at 1.5M)
        │    → Path B: Stage-2b bank  (LOCAL hardware, ~60h machine time)
        │    → redistill → Path C recipe on the NEW distill
        │
        └─ saturated (slope flat)
             → Stage-2b DEAD
             → Path C on the EXISTING arm-a distill is the only v6
               continuation (or shelve v6 entirely)

Path D  LEAGUE ROUND 4 — recommended MAIN compute program on the
        champion, independent of every v6 branch above.
```

Ordering rationale, recorded so it isn't relitigated: the redesigned PPO
(Path C) is deliberately NOT run before Stage-2b in the data-bound branch.
Running a novel recipe on a suspected-starved distill is a confounded
experiment — a null could blame either the recipe or the start, and would
burn the recipe's clean first shot. The probe is nearly free and routes
both big spends. The recipe is DESIGNED now (§6) and executes in whichever
branch needs it; it is needed in every continuation branch, because even a
data-rich redistill is still a fresh distill (F3's regime).

Declined option, recorded: a $0 two-trial "mechanism mini-round" of Path C
on the existing distill before the probe. Declined because its result
would be interpretable only for the old start, and the recipe gets its
clean test in-branch anyway.

## 4. Step 1 — data-scaling probe (FIRST ACTION, needs mini-prereg + go)

**STATUS 2026-08-15: RUN AND CONCLUDED — verdict DATA-BOUND** (prereg
docs/v6_data_scaling_prereg.md §9; ledger 2026-08-15). Slope real but
diminishing (match +1.6 pp per doubling at the top of the range); the
strength screen did NOT resolve the 1.0→1.5M step. Routing follows the
data-bound branch: Stage-2b (Path B) is proposable; its go/no-go band
must be in strength units with paired screens.

**UPDATE 2026-08-16 (Addendum A, paired strength n=5000×4): the
decision step bought NO detectable strength (S3−S2 +0.018 ± 0.074,
UB95 +0.163) → registered band: STAGE-2B NOT PROPOSED; v6 SHELVED
PENDING A NEW SIGNAL SOURCE.** Path B is dead on measurement. Path C on
the existing arm-a distill is the only v6 continuation (own prereg).
Path D (league r4) is the main compute program.

Question: does the v6 distill improve materially with more data, holding
the recipe fixed? (Is F4 the real constraint?)

Sketch (the mini-prereg pins details before running):
- Subsets of the EXISTING bank at ~0.5M / 1.0M / 1.5M records,
  match-contiguous so the by-match holdout stays clean; same holdout for
  all three.
- Arm-a recipe exactly as frozen (lr 1e-4, ep 3; batch 512).
- Metrics: holdout CE + teacher-match at each size; optionally 2 neutral
  screens (n=2500, ~2.5 min each) on the 1.0M and 1.5M nets.
- Decision criterion (to be pinned in the mini-prereg, roughly):
  teacher-match still climbing 1.0M→1.5M beyond holdout noise AND CE still
  falling → DATA-BOUND; flat in both → SATURATED.
- Cost: 3 trainings at ~5 min/epoch scale + evals — an afternoon, $0, no
  gates touched, no claims burned.

## 5. Path B — Stage-2b: second bank installment (RECORDED; needs prereg + approval)

Runs ONLY on a data-bound probe verdict.

- Content: second generation installment with the SAME recorder and
  distribution as Stage 2 (format v3: obs[882] + outcome labels + belief
  labels; 1/8 shooter-clone defense pressure, defenders recorded).
  Target: bring the total bank to **≥3M records** (~doubling).
- Honesty about sizing: 3M puts records/param at ~0.155 — still well below
  v5-M's 0.386. The probe's slope at 1.5M→3M should be re-read after
  Stage-2b before anyone proposes a third installment; do not extrapolate.
- Venue: **LOCAL hardware (user decision 2026-08-14 — no cloud for
  Stage-2b).** Ledger basis (SAME-CONFIG row, corrected 2026-08-14 —
  the first draft, like the as-signed v6 prereg §Stage 2/§Costs (~44h),
  quoted the 2026-07-17 v5-era 6.32 s/deal row; the prereg's figure is
  history and stands as signed, this doc supersedes it):
  Stage 2's own local RTX 4090 chunks with the format-v3 recorder at
  K=64/256 measured **93-98 s/match** (ledger, chunks 20-23; chunk 23 ran
  dedicated and chunks 20-22 with the webapp co-resident, with no
  measurable difference — so no headroom is claimed). At ~662 records/match, a ~1.5M-record
  installment ≈ ~2,300 matches ≈ **~60h machine time full-throttle** —
  ~40% over the old quote. Under rule-17 windows (5h/night) that is
  ~12 nights; with standing full-throttle permission, ~2.5 days
  elapsed. Quote the pace probe first anyway (launcher discipline).
- Sequencing note: local generation owns the GPU — it serializes against
  any Path C/D PPO trials and the xinxin arms. Order them, don't overlap.
- Seeds: fresh block, audited disjoint at launch against all used blocks
  (the Stage-2 run used 220M+; the audit list lives in the v6 prereg §Stage 2).
- Then: redistill arm-a recipe (arm-b as cheap control), holdout + screen
  vs the recorded Stage-3 numbers (a: CE 0.8413, match 64.2%, screen
  +0.828 SE 0.155). Success = a materially better start; the go/no-go
  band gets pinned in the Stage-2b prereg.
- Independent value, worth remembering: the distill imitates the SEARCHED
  champion (stronger than the raw champion). v5-M's scratch distill landed
  AHEAD of its era baseline. A non-starved v6 distill could in principle
  gate-pass with NO PPO at all — Stage-2b is not merely PPO-enablement.

## 6. Path C — redesigned v6 PPO (recipe designed NOW; execution slotted by branch)

The registered replacement for the failed Stage-4 approach. Runs after the
Stage-2b redistill (data-bound branch) or immediately on the existing
arm-a distill (saturated branch). Own prereg before any trial.

**R1 — KL-anchor to the distill init.** The core fix, aimed at F3's
mechanism. train.py already has the machinery (league r3: KL-to-baseline
on selected states); here the anchor snapshot is the distill itself, and
the anchor applies broadly (the thing to preserve is teacher knowledge,
not just threat-dead states). Dose is calibrated, not guessed: register a
drift band (~5-15% ordinary-decision drift per 75k-deal trial, measured
the diag_pass_region way) with a 2-shot calibration allowance. Starting
λ from the measured champion-lineage curve: 0.05 (curve: ~20% at λ=0,
7.8% at 0.05, 3.85% at 0.4). The curve on a v6 net will differ — hence
calibration shots.

**R2 — a registered factorial replaces mutation roulette for round 1.**
The Stage-4 chain searched a tiny neighborhood evolved for a different
regime (F3). Round 1 of Path C is a pinned 2×2: lr {1e-5 control, 5e-5}
× λ {calibrated dose, 0 control} — 4 trials ≈ 14h from MEASURED
components (2h 75k-deal training + 83-86 min n=3200 lineage gate ≈ 3.4h
per trial) plus a pad for the 2-shot λ calibration allowance (each shot
≈ another trial), so budget ~16-22h for round 1; note the ledger's OTHER n=3200 figure, 51 min, is
the v6-vs-v5-CHAMPION config with only one 19.4M arm — Path C gates are
both-arms-v6 and must be costed at 86 min). The λ=0 /
lr=1e-5 cell reproduces the Stage-4 regime as an internal control. Only
after a first pass does the mutation chain resume (seeded from the
passing cell).

**R3 — optional dense-reward warmup arm (held in reserve).** Match
placement reward is ~1 signal per 11 deals; the historically successful
orderings either used deal-reward PPO on a fresh distill (v5-M's first
promotion, deal era) or match-reward on an RL-mature net (the 5 match-era
promotions). If the 2×2 is uninformative, a fifth trial with deal-reward
warmup (or match_reward_scale annealing) before match-mode is the next
axis. Not in round 1 — each arm is ~4h and the 2×2 answers first.

**Telemetry + fail-fast (all from measured failure modes):**
- Moon-shot rate vs baseline per trial — the erosion signature (F3);
  informs, never gates.
- Mid-training cheap gate probe (n=800, **~21 min** — DERIVED by
  n-scaling the measured 86-min n=3200 both-arms-v6 gate; the first
  draft's "~10 min" was a v5-era figure, ~2× optimistic for v6-vs-v6)
  at ~half of the 75k-deal TRAINING budget (~1h in), so a damaging trial
  halts at ~1.4h (probe included) instead of ~3.4h — "half budget"
  means half the training deals, to be pinned in the Path C prereg.
- Three-strikes stop rule as standing house rule; candidate_lineage: true;
  explicit-snapshot gating; gates via scripts/run_match_gate.py ONLY.

**Must NOT do (each burned us once):** inherit config_chain_base
mutations as round 1; gate against the v5 champion before a lineage pass
(Stage-4 trials 1-2's original sin); run any gate outside
run_match_gate.py; serve or gate hearts_model_final.pth as if promoted.

## 7. Path D — league round 4 (recommended main compute program)

Independent of every v6 branch; runs on the champion lineage; $0 cloud.
Needs its own prereg (docs/exploiter_league_r4_prereg.md when drafted).

Design sketch from the chain in F5 — noting per the F5 correction that
this is the best-evidenced bet on the board, not a measured certainty:
the λ=0.05 candidate passed the guard but its defense probe was null
(underpowered, n=64), so whether teachability survives the in-band dose
is exactly what round 4 must test, not assume.
- Anchored PPO on champion 8a89da90: KL-to-baseline on threat-dead states
  at λ≈0.05 (the measured in-band dose), certified shooter clones in the
  pool at 0.15/0.15 (r1's shares).
- REGISTER the mid-training defense probe (the r2 lesson: teacher-match
  and drift screens are dead currencies for defense; probe the defense
  gate metric itself mid-run) — and SIZE it: the r3-probe005 defense
  null came from n=64 paired (SE 0.106, ~70% power vs −0.250). The r4
  prereg must power its probe to detect the r1-t3 effect properly
  (n≈150-200 paired puts the SE near 0.06-0.07).
- Gates: the r1 prereg battery — defense gate (moons/match improvement),
  match non-inferiority (UB ≤ +0.030), search guard (UB ≤ +0.3). r1 t3
  failed the guard at 22.4% drift; the λ=0.05 candidate measured UB −0.087.
- Trials are hours each; three-strikes applies.
- Prize: the champion's one live, user-demonstrated hole (moon defense
  51.5% vs 62.3%, ~2× concessions) — closing it is both strength and the
  removal of the site's prime human exploit.

## 8. Closed by this arc (do not revisit without new evidence)

RECORDED in the register (docs/experiment_rules.md, 2026-08-14 — the
first draft said "adds to" without actually adding; the register now
also carries the previously-missing Phase-2 visit-count closure):
- **Mutation-chain match-PPO (champion-regime config neighborhood) on a
  fresh, far-below-ceiling distill: measured actively damaging** (3/3
  trials, monotone mitigation never reaching zero). Path C §6 is the only
  sanctioned retry shape.
- Scale-without-data (arm a 19.37M vs arm b, ~2.4× — F2's pinned
  contrast; 2.55× vs the champion): no pre-PPO effect, conditional on the
  1.52M bank — the F4 probe was the registered reopening test. UPDATE
  2026-08-16: probe run; imitation data-bound but strength saturated at
  1.0→1.5M (Addendum A) → Stage-2b NOT PROPOSED, v6 SHELVED.

Standing closures that constrain these paths (register has details):
equity-ordering distillation targets (expert-iter v2), visit-count targets
(Phase 2), same-lineage refresh recipes, anchored supervised defense
imitation at ~60k-decision scale (r2), K>64 escalation / ISMCTS.

## 9. State pointers for a fresh session

- Ladder record: logs/v6_stage4_ladder.log (3 verdicts + configs);
  stopwatch log logs/v6_ladder_stopwatch.log; ledger entry 2026-08-13.
- Lineage milestones: v6_stage4/milestones/ (rolled back; nothing promoted).
- Frozen Stage-3 nets + verdicts: v6_stage3/, equity_data/verdicts/.
- Champion: Hall_of_Fame/milestone_1785322724 = 8a89da90 (verified).
- The dose/guard/defense data Path D rests on:
  equity_data/verdicts/r3_probe005_guard.json (guard PASS),
  r3_probe005_defense_gate.json (defense NULL, n=64), r3_drift_*.json,
  exploiter_r1_defense_gate_r1t3.json + exploiter_r1_gates23_r1t3.json
  (the only defense pass and its two gate failures).
- Data-scaling probe record: docs/v6_data_scaling_prereg.md §9/§11,
  equity_data/verdicts/v6_data_probe.json, v6_probe/ (probe nets — never
  serve, never gate).
- Nothing in this plan is launched. Sequence on user go: Step-1 probe
  mini-prereg → probe → branch decision → (Stage-2b approval if
  data-bound) → Path C prereg → Path D prereg may proceed in parallel at
  any point.
