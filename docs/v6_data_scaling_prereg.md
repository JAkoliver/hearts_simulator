# v6 data-scaling probe — mini pre-registration

Status: **SIGNED by user 2026-08-15 21:44 PDT ("I sign off").** Written
2026-08-15. Design binding as written. The [FREEZE] instrument md5s in
§5 were filled AFTER signature but BEFORE any counted run and before
any result existed (the Study-2 [CAL]/[FREEZE] convention); the
signature covers the design, and the freeze record below covers the
instruments.

Context: Step 1 of docs/v6_next_plan.md §4 — the first action of the
post-Stage-4 routing plan. It is also the **registered reopening test**
named by the CONDITIONAL scale-without-data closure in
docs/experiment_rules.md (2026-08-14): "more data" is that closure's
recorded untested variable, and this probe is how it is varied. House
rules bind throughout (halt-default; telemetry informs, never gates;
one pre-unblinding amendment; durations quoted from
docs/speed_ledger.md).

## 1. Question

Does the v6 distill improve materially with more data, holding the
recipe fixed? I.e. is F4 (docs/v6_next_plan.md §2 — the bank may be too
small: 1.52M records for 19.37M params, records/param down ~4.9× from
the 2.93M bank that built v5-M) the real constraint?

The verdict routes two large spends:
- **data-bound** → Path B / Stage-2b becomes proposable (~60h local
  machine time, own prereg + approval) → redistill → Path C on the new
  distill.
- **saturated** → Stage-2b is dead; Path C on the existing arm-a
  distill is the only v6 continuation, or v6 is shelved.

## 2. What is NOT at stake

No gate is run. No promotion candidate is produced. Champion 8a89da90
is untouched, `hearts_model_final.pth` is not written, and no net
produced here may be served or gated. This is an imitation measurement
on a frozen bank.

## 3. Design

**Bank.** `expert_data/v6_bank/` — 768 files (24 generations × 32
threads), 1,524,821 records, 2,304 matches, format v3, all v3 checks
passed (ledger 2026-08-12). Each per-thread file contains whole
matches, so any file subset is match-contiguous by construction.

**Sizes — nested and generation-stratified.** Every subset takes the
same thread indices from ALL 24 generations, so the cloud/local venue
mix and the seed spread are proportional at every size, and each subset
is a strict subset of the next:

| tag | threads | files | records (approx) |
|---|---|---|---|
| S1 | t0–t9 | 240 | ≈ 0.48M |
| S2 | t0–t20 | 504 | ≈ 1.00M |
| S3 | t0–t31 (all) | 768 | 1.52M (exact) |

Approximate counts are from the uniform ~1,985 records/file; the exact
realized counts are recorded by the driver and reported.

**Recipe.** Arm-a exactly as frozen at Stage 3: HeartsNetV6 d448 L8,
**lr 1e-4, 3 epochs, batch 512**. No recipe search, no re-banding, no
hyperparameter variation. The ONLY thing that varies across runs is the
number of training records.

**Seeds.** Two per size (20260812, 20260813) → **6 trainings**. The
second seed exists because a 3-point slope from single runs cannot
separate a data effect from run-to-run training noise; the seed spread
is the registered noise floor.

**Holdout — identical for all six nets.** The full-bank holdout
(`md5(basename:match_id) % 20 == 0` over all 768 files) = **123
matches / 80,220 records**, the same split Stage 3 froze on. This is
clean for every net by construction: each subset training excludes its
own mod-20 matches, and those are a subset of the full holdout, so no
net has ever seen any full-holdout record.

## 4. Metrics and decision rule (frozen before any run)

Both primaries are computed **per holdout match** (123 clusters) and
compared **paired across sizes** (identical matches, identical
records):

- (a) **teacher-match rate** — top-1 agreement with the search-chosen
  action.
- (b) **policy cross-entropy** on the search-chosen action.

**Inference.** Paired by-match differences; one-sided t test at
α = 0.05 with exact Wilcoxon signed-rank as robustness; Holm over the
two metrics within each step. Records cluster hard by match, so
record-level binomial SEs are **inadmissible** — the by-match cluster
is the registered unit of analysis. The two seeds at a given size are
averaged per match before differencing (a seed is a nuisance replicate,
not a unit).

**Registered steps.**

1. **POSITIVE CONTROL — S1 → S2 (0.5M → 1.0M).** If neither metric
   improves significantly here, the probe is **UNINFORMATIVE**
   (instrument or recipe insensitive at this scale) — NOT "saturated".
   Report and halt; no routing, no closure change.
2. **DECISION STEP — S2 → S3 (1.0M → 1.5M)**, read only if the control
   passed:
   - **DATA-BOUND** — both metrics improve, Holm q < 0.05.
   - **SATURATED** — neither improves at q < 0.05.
   - **MIXED** (exactly one improves) — halt-default: no Stage-2b, no
     closure change; the numbers go back to the user for a decision.

Effect sizes and 95% CIs are reported for every step regardless of
verdict. No verdict is read from a point estimate alone.

**Diagnostics (inform, never gate).** Holdout play-phase entropy per
run (Stage-3 band [0.22, 0.87]; arm a froze at 0.832), train-loss
curves, per-size wall-clock.

**Registered optional screens (estimation only).** Neutral raw screen
n=2500 on ONE net per size (the seed-20260812 net), giving the slope in
strength rather than imitation units. Reference point: arm a's frozen
Stage-3 screen +0.828 (SE 0.155). No band, no gate, no significance
claim — CIs only, per the Study-1 sweep convention.

## 5. Instruments (frozen at signature)

- `v6_distill.py` md5 **fb3b0c4f5ac7c7b439045b7c656d0886** — UNCHANGED;
  the probe does not modify it.
- `hearts_net.py` md5 **1dd01de477d18a6b601704c0028d296b**.
- **NEW — probe driver**, to be built, self-tested and md5-frozen
  BEFORE signature: selects the file subsets by the §3 rule, invokes
  the frozen recipe unchanged, evaluates every produced net on the
  FULL-bank holdout, and emits a per-match CSV for the §4 analysis.
  Required because `v6_distill.py --data` takes a single glob and
  computes its holdout over the loaded files only, so fixed-holdout
  evaluation across sizes has no CLI entry today. BUILT + FROZEN
  2026-08-15 ~22:00 PDT, before any counted run:
  - `v6_probe_driver.py` md5 **98132ced63c7dc8f3506ecd3204706fb** —
    stages S1/S2 as HARDLINK dirs (`expert_data/v6_bank_probe/{S1,S2}`,
    basenames unchanged so the frozen mod-20 split is bit-identical to a
    filtered full-bank split; verified 240/504 files), and runs the
    frozen `v6_distill.py` UNCHANGED via subprocess with
    `--arm a --lr 1e-4 --epochs 3 --batch 512`, one file-logged run per
    (size, seed). Refuses to overwrite existing outputs.
  - `v6_probe_eval.py` md5 **c6151324f1375a170070f6a11d39325b** —
    fixed full-bank-holdout per-match scorer using the frozen
    `v6_distill` functions. SELF-TESTS PASSED before freeze: (1) its
    holdout is byte-identical to `v6_distill.load_bank`'s (80,220
    records / 123 matches); (2) it reproduces the frozen Stage-3 arm-a
    net's recorded holdout numbers — CE 0.8413 (ref 0.8413), match
    0.6417 (ref 0.642).
- **Analysis script** `v6_probe_analysis.py` md5
  **0dce14b1f645f2e60b05289504c0fa88** — paired by-match tests, exact
  Wilcoxon, Holm, CIs, verdict per §4. SELF-TEST PASSED on synthetic
  data: planted SATURATED / DATA-BOUND / UNINFORMATIVE all recovered.
- Pre-flight (2 files, 2 epochs, throwaway seed) ran end-to-end through
  the real subprocess path + evaluator: PASS.
- If the screens are run they use the existing `match_eval` path. The
  working tree carries uncommitted modifications to `match_eval.py`,
  `hearts_match_env.py`, `train.py` and `bindings.cpp`; their md5s and
  the built `hearts_env` extension are recorded at freeze so the screen
  provenance is explicit: `match_eval.py` e1a0cd01c5eb9cff5b1469443c1a9bc9,
  `hearts_match_env.py` 2589a8b1582ce805a8f06ddf176884f7, `train.py`
  07cb6e5301f3746ecf8b99280dfb05bb, `bindings.cpp`
  68d999edd06ff69235b56070e0031799, `hearts_env.cp313-win_amd64.pyd`
  5718e358b7e80fef5f0b86c3731fcdea, `headroom.py`
  6a0d4c74f43421c3a4ae402218ecace6, `cloud/shard_check.py`
  6170132337f548bd69253c1d6a5509a7.
- Launcher discipline applies: unbuffered, file-logged, launched from a
  script file with a `__main__` guard, monitor + watchdog at ~2× the
  quoted duration.

## 6. Costs

MEASURED (docs/speed_ledger.md, 2026-08-13): **5.1 min/epoch** at batch
512 on the full 1.52M bank; neutral screen n=2500 **≈ 2.5 min**.

DERIVED by linear scaling in record count (labeled as derived, not
measured): 3-epoch trainings ≈ 5 min (S1) + 10 min (S2) + 15 min (S3)
≈ 31 min per seed → **≈ 62 min for both seeds**; fixed-holdout
evaluations ~1–2 min each × 6; screens 3 × 2.5 min.

**Total ≈ 1.5 h GPU, $0 cloud, no gate touched, no claim burned.**
Rule 17: short enough to run inside a usability window with explicit
session permission; otherwise 02:00–07:00.

## 7. What results are allowed to mean

- **DATA-BOUND**: reopens the conditional scale-without-data closure by
  exactly the route the register's admissibility clause permits. Stage-2b
  becomes *proposable* — under its own prereg and approval, at the
  ~60h local quote. It does NOT license Path C, a promotion claim, or
  any statement about PPO.
- **SATURATED**: Stage-2b is dead and the conditional closure hardens;
  the a-vs-b tie becomes attributable to the architecture rather than
  the bank. Path C on the existing distill, or shelving v6, is the
  remaining v6 conversation.
- **UNINFORMATIVE control**: no routing, no closure change; the probe
  itself is the thing that failed.
- Either verdict is an **imitation** result. It says nothing about match
  strength; the screens, if run, are estimation only and gate nothing.

## 8. Deviations

One pre-unblinding amendment, per house rules. Crash-forced instrument
repairs follow the established precedent (fix, re-verify, document,
user approval, full record) and do not spend the amendment. Nothing is
unblinded until all six trainings and the fixed-holdout evaluation have
completed.

Signature: user, "I sign off" — 2026-08-15 21:44 PDT.

## 9. RESULT (2026-08-15 23:05 PDT — read only after all six trainings and the fixed-holdout evaluation completed)

**VERDICT: DATA-BOUND** by the §4 rule as written. Control S1→S2: CE
−0.0703 [−0.0732, −0.0674], match +2.75 pp [+2.51, +3.00]. Decision
S2→S3: CE −0.0217 [−0.0240, −0.0194], match +0.89 pp [+0.70, +1.09].
Holm q < 10⁻⁴ and exact Wilcoxon < 10⁻⁴ for all four; 123 paired
matches. Slope diminishing per doubling (CE −0.065 → −0.038; match
+2.5 → +1.6 pp). Screens (estimation only, unpaired, n=2500):
S1 +1.888 (0.154), S2 +0.439 (0.140), S3 +0.766 (0.149) — the 1.0→1.5M
step is unresolved on the strength axis (+0.33 ± 0.20). Full record:
docs/speed_ledger.md 2026-08-15; equity_data/verdicts/v6_data_probe.json;
per-match CSV v6_probe/holdout_by_match.csv; logs/v6_probe_*.log.

Instrument md5s re-verified unchanged after the run. No amendment used.
Meaning per §7: Stage-2b is PROPOSABLE under its own prereg; nothing
else is licensed.

Provenance note on the screens (correcting §5's wording, not the
design): they ran via `neutral_raw_eval.py` (md5
a1e20e7c1d9a36cb8cf9c4b7f4dfba63) → `orchestrator.evaluate_candidate_
neutral_raw` (orchestrator.py 6db0da7300487b49a86e1d797312cd9b) — the
same instrument and baseline path (`Hall_of_Fame/hearts_model_milestone_
1785322724.pth`, md5 8a89da90 verified pre-run) that produced the
Stage-3 screens; script logs/run_v6_probe_screens.sh, log
logs/v6_probe_screens.log.

## 10. ADDENDUM A — paired strength probe (SIGNED by user 2026-08-16 ("run it") — bands fixed before any pairing ran)

**Why.** §9's verdict is data-bound in IMITATION units. The decision it
routes — whether to spend ~60h of local generation on Stage-2b — is a
STRENGTH question, and the §9 screens left it unresolved because they
were unpaired across sizes (each net vs the champion on its own seed;
cross-size SE ≈ 0.20). This addendum converts the decision step into
strength units with a paired instrument, so Stage-2b is proposed or
declined on measurement. It reads the SAME six nets; nothing is
retrained. It is an addendum (new measurement on existing artifacts,
registered before it runs), not the §8 amendment; the amendment budget
stays untouched.

**Instrument.** `neutral_raw_eval.py` → `orchestrator.evaluate_
candidate_neutral_raw` (md5s recorded in §9), which seats CANDIDATE and
BASELINE one at a time at the same seat of identical deals against
three neutral v3-m7 anchors and tests the paired per-deal differential.
Here candidate/baseline are BOTH probe nets:

| pairing | candidate | baseline | measures |
|---|---|---|---|
| P1 | S3_s20260812 | S2_s20260812 | decision-step strength, seed A |
| P2 | S3_s20260813 | S2_s20260813 | decision-step strength, seed B |
| P3 | S2_s20260812 | S1_s20260812 | control-step strength, seed A |
| P4 | S2_s20260813 | S1_s20260813 | control-step strength, seed B |

n = **5000 paired deals** per pairing (SE ≈ 0.10; the champion screens
at n=2500 gave SE 0.14–0.15). Sign convention as the instrument prints
it: **negative = candidate (larger bank) better.** Seeds are the
instrument's own (`int(time.time())`, recorded in the log); the four
pairings are independent measurements — P1 and P2 are pooled by
inverse-variance weighting for the decision quantity, P3/P4 likewise
for the control.

**Registered readout (set BEFORE any pairing runs).** Let D = pooled
paired mean of P1+P2 (S3 − S2, per deal), with its 95% CI.

- **Control sanity (P3+P4).** The imitation control step was large
  (+2.75 pp match) and the unpaired screens showed −1.45/deal. If the
  pooled S2 − S1 strength is NOT significantly negative (UB95 ≥ 0),
  the paired instrument is inconsistent with everything already
  measured → HALT and report as an instrument problem; the decision
  readout is not read.
- **Decision (P1+P2), read only if the control is sane:**
  - **UB95(D) ≤ −0.20/deal** → **Stage-2b clearly worth pricing.**
    +0.47M records bought ≥ 0.2/deal at the top of the measured range;
    a doubling to 3M is credibly worth pursuing. Stage-2b prereg is
    drafted; its go/no-go band is set in strength units from D.
  - **−0.20 < UB95(D) ≤ −0.10/deal** → **Stage-2b worth pricing,
    marginal.** Drafted, but the prereg must state the projected gain
    honestly (D per +0.47M records, slope diminishing) and the user
    weighs it against league r4's queue time.
  - **UB95(D) > −0.10/deal** → **Stage-2b NOT PROPOSED.** The last
    +50% of data did not buy a detectable strength gain at SE 0.10; a
    doubling cannot credibly close the +0.77/deal gap to the champion.
    v6 is recorded as SHELVED PENDING A NEW SIGNAL SOURCE (the register
    entry gets that line); Path C on the existing distill remains the
    only v6 continuation, and only behind its own prereg.

The bands are on the UPPER confidence bound, so a noisy null cannot
promote Stage-2b; point estimates and CIs are reported for all four
pairings regardless. Nothing here is a gate; no net produced by the
probe is a candidate for anything.

**Cost (measured basis, ledger 2026-08-15).** n=2500 champion screen =
3.9 min incl. worker spin-up → n=5000 ≈ 8 min → **4 pairings ≈ 32 min
GPU/CPU, $0.** Runs from a script file, unbuffered, file-logged
(`logs/run_v6_probe_paired.sh` → `logs/v6_probe_paired.log`), watchdog
~1 h. Machine otherwise idle (12 workers → gentle-scaled).

**Provenance.** Champion NOT involved (both arms are probe nets);
`Hall_of_Fame` untouched; `hearts_model_final.pth` not written. Net
md5s recorded in the log at launch. Result appended as §11 with the
pooled D, CIs, and the band that fired.

Signature (Addendum A): user, "run it" — 2026-08-16.

## 11. ADDENDUM A RESULT (2026-08-16 03:46 PDT; four pairings n=5000, seeds in logs/v6_probe_paired.log)

| pairing | cand − base (per deal) | SE |
|---|---|---|
| P1  S3−S2 seed A | +0.198 | 0.105 |
| P2  S3−S2 seed B | −0.158 | 0.104 |
| P3  S2−S1 seed A | −1.049 | 0.112 |
| P4  S2−S1 seed B | −1.345 | 0.111 |

**Control (P3+P4 pooled): −1.198 (SE 0.079), UB95 −1.044** → sane; the
paired instrument sees the control step exactly where the unpaired
screens put it (−1.45 ± 0.2). Decision readout is read.

**Decision (P1+P2 pooled): +0.018 (SE 0.074), CI [−0.127, +0.163],
UB95 +0.163** → band 3: **UB95 > −0.10 → STAGE-2B NOT PROPOSED.** The
last +0.47M records (+49%) bought no detectable strength at SE 0.074
— point estimate on the wrong side of zero. A doubling to 3M cannot
credibly close the +0.77/deal gap to the champion. Per the registered
consequence: **v6 is SHELVED PENDING A NEW SIGNAL SOURCE**; Path C on
the existing arm-a distill remains the only v6 continuation, behind its
own prereg; the conditional closure's register line is updated.

**Finding on the record (not a band, informs):** the two training
seeds at the SAME size differ in strength by 0.36/deal at the decision
step (P1 vs P2, 2.4 SE apart) and 0.30 at the control step (1.9 SE).
Training-seed variance in strength (~±0.18/deal per net) exceeds the
effect of +50% data and is comparable to the Stage-3 arm gaps
(a +0.83 / b +0.63 / c +0.59, single seed each). Any future strength
comparison between distills of similar quality needs ≥2 training seeds
per arm; the imitation metrics (seed spread 0.3–0.6 pp) are far less
seed-sensitive than strength.

Champion untouched; hearts_model_final.pth not written; instrument md5s
unchanged. Addendum A closed. Amendment budget unused throughout.
