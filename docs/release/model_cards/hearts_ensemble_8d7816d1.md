# Model card — gated ensemble champion (milestone 1787333162, md5 8d7816d1)

Promoted 2026-08-21 (6th match-era promotion; first ensemble). One
882-input TorchScript-able module (`HeartsHybrid`, hearts_net.py):

| component | file (frozen) | md5 | role |
|---|---|---|---|
| default | Hall_of_Fame/hearts_model_milestone_1785322724.pth | 8a89da90 | plays ~90% of decisions (556-dim prefix) |
| specialist | v6_stage3/arma_lr3e-4… (arm a, 19.37M, obs v2) | a9653255 | plays gated moon-threat states |
| router | arm a's aux moon head, τ=0.1 + moon-alive check | — | public information only; fire rate is DISTRIBUTION-DEPENDENT: ~11.7% of plays on the search-bank holdout, ~5–6% in champion self-play, ~2.5% vs weak anchors; ~0.4% of PASS decisions can also gate (no points term — corrected 2026-08-21) |

Inputs: obs v2 (882 public dims — v1 obs 550 + match ctx 6 + extension
326). Outputs: masked policy logits + value (belief via forward_all).
Training: NONE at the ensemble level — a composition of frozen nets;
the specialist is a from-scratch imitation student of the searched
8a89da90 (v6 Stage 3, single seed).

Registered evaluation (docs/exploiter_league_r7_prereg.md + Amendment 1;
results docs/exploiter_league_r7_results.md):
- Match non-inferiority n=3,200 mixed anchors: Δplace −0.011 (SE 0.007).
- SEL search-shooter defense gate, fresh-seed primary n=256: 2.559 →
  1.816 moons conceded/match (−0.742, SE 0.071; pooled n=320 −0.697).
- Guard: promotion is raw-only; served search substrate = the champion's
  traces (3a2abd36 / efdfee07), verified unchanged. Ensemble-as-rollout
  telemetry +0.016 (SE 0.167, n=2,400) — neutral; not a serving config.
- Clone-probe estimates (informs): SEL −0.360, AGG −0.647; paired
  per-deal strength vs champion −0.010 (SE 0.046).

Limits, honestly: defense measured against ONE attacker family (the
search shooter + its clones); no claim about human attackers; the
specialist is one training seed; τ chosen on clone probes (battery ran
on disjoint seeds); serving cost ≈ champion + specialist-aux per play
decision until the cheap pre-gate lands (program doc §7 step 1-2).
Weights: models-v1 release assets `hearts_ensemble_8d7816d1.pth`
(checkpoint, md5 8d7816d1) and `hearts_ensemble_8d7816d1_882trace.pt`
(served-form trace, md5 9d9a4f49), with ENSEMBLE_MD5SUMS /
ENSEMBLE_SHA256SUMS alongside; never tracked in git.
