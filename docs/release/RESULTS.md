# RESULTS — every major measured claim

> STATUS: DRAFT - not yet released; project ongoing.

Conventions: per-deal deltas are points/deal, candidate minus
comparator on paired deals — NEGATIVE is better (fewer points is good
in Hearts). Placement deltas likewise negative-is-better. SE = standard
error of the paired mean. Claims discipline per docs/RELEASE_PLAN.md
sec. 4.5: measured claims only; user calibration matches are n=1
anecdotes and labeled so. Anything lacking a located primary source is
marked [NEEDS CITATION] rather than guessed.

## Architecture and per-deal era

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| v5-M (7.6M card-token transformer, distilled) beats v4-m10 baseline, raw gate | **-1.233** pts/deal | n=[NEEDS CITATION: gate n not in commit body; t=-9.27] | t=-9.27, p<1e-6 | 07-15 | commit a4136b5 |
| Same result is >3x best MLP candidate (B2 -0.388) at half B2's params | -1.233 vs -0.388 | — | — | 07-15 | a4136b5 |
| v5-S (1.9M) ties fully-PPO-trained v4 baseline from distillation alone | +0.058 (tie) | — | — | 07-15 | a4136b5 |
| v5-M searched play ties the v4 teacher | -0.099 | n=1000 | ±0.526 | 07-15 | 293c0fd |
| First PPO promotion on v5 (weak gate) | -0.712 | n=600 | p=0.016 | 07-15 | docs/ppo_v5_round2_findings.md |
| Re-gate of that promotion at powered n: it was real and UNDERestimated | pre-PPO **+1.025** worse | n=2400 | SE 0.180, t=5.68 | 07-21 | ledger 2026-07-21; ppo_v5_round2_findings.md |
| ISMCTS at 1600 iters ties flat K=64 at 55x wall-clock (closed) | +0.533 (tie) | 300 paired deals | — | 07-15 | b835897 |
| Oracle-leaf evaluator error vs action-value gaps (closed) | RMSE 6.8 pts vs gaps 0.5–1.5 | 7,200 leaf states | — | 07-14 | 7a3abc3 |

## Speed and cloud certification

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| SDPA attention promotion gate (pre-registered) | pooled **+0.068** | n=3600 paired deals | ±0.133 (PASS both criteria) | 07-17 | 24bb45d; docs/sdpa_gate_preregistration.md |
| Local steady generation after SDPA | **6.32 s/deal** | 400-deal run, seed 777 | — | 07-17 | ledger header table |
| Measured steady speedup vs prior config | 1.51x (clean); 1.93x vs original (soft) | — | — | 07-17 | ledger header |
| H100 JIT steady generation | 3.24 s/deal (1.95x local); $2.69/1k deals | 300 deals | — | 07-17 | ledger, H100 session |
| H100 cross-hardware gate, JIT stack | FAIL on power: mean -0.010, UB +0.331 vs +0.30 | n=3000 | SE 0.2072 | 07-17 | ledger |
| H100+AOTI steady generation | **2.22 s/deal** (2.85x local); $1.84/1k deals | 300 deals | — | 07-18 | ledger, H100 AOTI session |
| H100+AOTI cross-hardware gate (certification) | mean **-0.139**, UB **+0.064** < +0.30 → PASS | n=8000 | SE 0.123 | 07-18 | ledger |
| First cloud iteration: ops clean, strength FAIL | gate +0.450 (n=600); $7.02 | 3,500 deals gen | p 0.90 | 07-18/19 | ledger |

## Gate re-powering and PPO campaigns

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| Search gate re-power 600→2400; re-verdict on cloud candidate | **+0.679** definitively worse | n=2400 | SE 0.169, t=4.0 | 07-19 | ledger; 7f484f9 |
| PPO round 2: searched strength flat | pooled **-0.078** | 9,600 paired deals (4 trials) | SE 0.081; 95% CI [-0.24, +0.08] | 07-19/20 | ledger; ppo_v5_round2_findings.md |
| PPO raw gains genuine vs neutral opponents | -0.654 and -0.636 | n=2500 each | SE ~0.143/0.144, p<1e-5 | 07-21 | ledger; ppo_v5_round2_findings.md |
| First raw-line promotion | neutral raw **-0.619**; guard UB +0.187 | n=2500 / n=2400 | SE 0.141, p=1e-5 | 07-21 | fff75a2; ledger |
| PPO raw gain is one-shot vs new baseline | pooled -0.093 | 7,500 paired deals | SE 0.074 | 07-21 | ledger |
| v5-L from-bank distills fail | +3.988 / +3.776 | n=2500 each | SE ~0.175 | 07-22 | ledger |
| Fresh-bank warm-start distill fails (12.5k deals) | **+0.479** | n=2500 | SE 0.144, p=0.9996 | 07-23 | ledger |
| Sharpen 2.0 halves the degradation; saturates (4.0 worse) | +0.248 / +0.254 / +0.470 (s=2/3/4) | n=2500 each | SE ~0.14 | 07-23 | ledger |
| 255k further PPO games on best candidate: flat-to-negative | +0.24 → +0.428 across stages | n=2500/stage | SE ~0.14 | 07-23 | ledger |
| Deployed player vs 07-14 calibration opponent | **-1.016**/deal stronger | 1,200 paired deals | SE 0.234, p=0.00002 | 07-23 | ledger |

## Match era — promotions and the pivot

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| Score-blind baseline (+~2 pts/deal over anchors) wins only ~30–37% of 4-seat matches | ~30–37% vs 25% chance | 60-match calibration | — | 07-23 | ledger, Phase 1 |
| **Match promotion #1** | placement **-0.085**; raw -0.233 | n=800 matches | SE 0.041, p=0.018; raw p=0.043 | 07-24 | ledger; milestone 1784888158 (25db6f93) |
| **Match promotion #2** (compounding) | placement **-0.133**; win 36.0% v 31.5% | n=800 | SE 0.040, p=0.00044; win p=0.011 | 07-24 | ledger; milestone 1784900322 (9cb0ba9f) |
| **Match promotion #3** | placement **-0.087**; win 39.8% v 35.9% | n=800 | SE 0.036, p=0.008; win p=0.025 | 07-24 | ledger; milestone 1784920549 (10abe622) |
| Cumulative vs fixed anchors after night 1 | win 26.8%→39.8%; placement 2.44→2.05 | — | — | 07-24 | ledger |
| Trial 6: first search-guard veto (match PASS, substrate FAIL) | match -0.121 (p=0.0006); searched +0.477, UB +0.739 | n=800 / n=2400 | SE 0.037 / 0.159 | 07-24 | ledger |
| Bridge: match-blind SEARCH still beats match-aware RAW at match play | win 50.5% vs 38.0%; placement -0.175 | n=200 matches | discordant 51:26, McNemar p=0.006; placement p=0.068 | 07-24 | ledger; 1130e38 |

## Match-aware search program

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| Equity model calibration: ECE below clustered noise floor | ECE 0.0037 vs floor 0.0575 (agg); Brier 0.614 agg / 0.336 near-terminal | 5k-match natural holdout (2,000 S2 matches) | match-level cluster bootstrap | 07-25 | verdicts/diagnostics.json |
| Equity net selected over both frozen lookups | Brier 0.614 vs 0.645 | holdout | — | 07-25 | verdicts/selection.json |
| Flip/SNR spine gate: **HALT** | tension flip 38.9% (≥5% floor OK) but SNR 0.62 < 1.0; deal-point reference 0.86 | 660 tension / 11,245 decisions | — | 07-25 | verdicts/flip_snr.json (ledger entry quotes 36.8% / 0.41 / 0.57 — see note below) |
| Confident flips at K=64 (flips are ~97% noise) | tension 1.59%, runaway 3.13%, early 1.32% | probe decisions | — | 07-25 | ledger 2026-07-25 |
| K=256-endgame: confident-flip rate in tension | 2.6x vs flat K=64; cost +37%/match (19.2→26.2 s/pair) | probe v2 | — | 07-25 | docs/experiment_rules.md #15; 003a846 |
| Behavioral diagnostics (moon-attempt / leader-dump directions) | [NEEDS CITATION — tooling and behave_*.csv exist; no recorded verdict found] | — | — | 07-25/26 | ea2d33e |
| **N=8000 validation: match-aware search beats frozen match-blind reference** | **48.91% vs 44.47% = +4.44 win-pts (SE 0.68)** | N=8000 paired matches | discordant 1668:1313 (q=0.373), McNemar one-sided **p≈5e-11**; all 8 shards positive (+2.4..+6.6) | 07-27 | ledger 2026-07-27; 02551fc; data equity_data/validation_v1/ |
| Placement structure: P2→P1 AND P2→P4; mean place WORSE | P1 3913v3558, P2 1028v1838, P4 1780v969; mean 2.098 v 1.980 | N=8000 | — | 07-27 | ledger |
| Exploratory: effect by match-length tercile | +13.3 / +0.9 / -5.9 win-pts (short/mid/long) | N=8000 | exploratory, endogenous split | 07-27 | ledger |
| Validation cloud cost | $62.72 actual vs ~$46–48 projected | — | — | 07-26/28 | ledger |

## Evolved regime

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| Trials 1–4 pooled under evolved guard (motivated match-gate re-power) | -0.027 | n=3200 pooled | ±0.017, p~.06 | 07-28 | ledger |
| Match gate re-power 800→3200: power vs true -0.05 | 43% → 90% | — | — | 07-28 | ledger |
| **Match promotion #4** (first at n=3200) | placement **-0.029**; guard +0.029, UB +0.292 (pass by 0.008) | n=3200 / n=2400 | SE 0.017, p=0.0456 | 07-28 | ledger; milestone 1785273667 (cbfde942) |
| Search guard re-power 2400→4800: neutral-candidate pass rate | ~61% → ~86% | — | — | 07-28 | ledger |
| **Match promotion #5** (re-power vindicated) | placement **-0.031**; win **53.3% v 50.5%**; score -0.98; guard +0.072, UB +0.258 (old n would give ~+0.335 = false veto) | n=3200; guard n=4800 | p=.0247; win discordant 499:407, p=.0012; score p=.0078; guard SE 0.113 | 07-29 | ledger; milestone 1785322724 (8a89da90) |
| Analyzer before/after vs v4-m10 (2,000 deals, 16,000 games/side) | baseline 6.11 avg pts / 36.1% deal wins vs v4-m10 7.35 / 27.3%; solo diff -2.134 | 16,000 games | solo_diff p≈0 | 07-28 | analyzer_history.csv 2026-07-28 rows; 222acfe |
| Known weakness (unchanged): moon defense | 51.5% vs v4-m10 61.1%; moons conceded 272 vs 151 | 2,000 deals | — | 07-28 | analyzer_history.csv; RELEASE_PLAN sec. 3 |

## Expert iteration (era 8)

| Claim | Number | N | CI / p | Date | Citation |
|---|---|---|---|---|---|
| Wedge fix: row cap bounds VRAM | peak 23.9 GB pre-fix → 12.9 GB at 14 threads; no wedge over ~4.5 h | — | — | 07-30/31 | ledger; 45821a6 |
| Row cap throughput surprise | 44 s/match at 5 paced threads vs 37 at 14 unpaced pre-fix | — | — | 07-30 | ledger (clean A/B still pending) |
| Expert-iter bank | 333,415 records; 72.8% natural, 27.2% seeded (~4.5x tension-enriched) | — | — | 07-31 | ledger |
| Soft-target distills un-sharpen the champion | entropy 0.32 → 1.04–1.08; teacher-match below baseline | holdout | — | 07-31 | ledger |
| Hard-argmax distill imitation metrics | holdout teacher-match 60.1%; tension +10.2 over baseline | holdout | — | 07-31 | ledger |
| **Expert-iter v1 one-shot gate: FAIL (decisive)** | win **39.9% v 50.3%**; placement **+0.292** | n=3200 matches | discordant 377:710; placement SE 0.019 (~17 SE) | 07-31 | ledger; 88ff398 |
| Expert-iter v2 | pre-registered, in progress — no results | — | — | 08-01 | docs/expert_iter_v2_prereg.md (DRAFT) |

### Note on the flip/SNR figures

The ledger entry (2026-07-25) quotes tension flip 36.8%, SNR 0.41,
deal-point reference 0.57; the verdict JSON (equity_data/verdicts/
flip_snr.json) records 38.9%, 0.62, 0.86. Both record the same verdict
(HALT: flip floor passed, SNR threshold failed, reference also < 1.0).
The discrepancy between the two primary sources is flagged here per the
INDEX contract and should be resolved against the analysis script
before release; this table reports the machine-readable verdict values
with the ledger variant noted.

Cross-references: narrative context in docs/release/JOURNEY.md;
how these numbers were produced in docs/release/METHODOLOGY.md.
