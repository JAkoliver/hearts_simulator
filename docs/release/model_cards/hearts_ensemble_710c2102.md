# Model card — v6.1 gated ensemble champion (milestone 1789949580, md5 710c2102)

Promoted 2026-09-20 (7th match-era promotion; second ensemble). One
882-input TorchScript-able module (`HeartsHybrid`, hearts_net.py, gate
`moonhead2:0.1:0.26`) — the v6 ensemble (8d7816d1) with a second router
tier and a trained specialist:

| component | file (frozen) | md5 | role |
|---|---|---|---|
| default | Hall_of_Fame/hearts_model_milestone_1785322724.pth (v5 champion, 7.6M) | 8a89da90 | plays every decision the router does not hand off (~95%) |
| tier-1 specialist AND router | v6_stage3/arma_lr1e-4.ep3.pth (v6 arm a, 19.37M, obs v2) | a9653255 | its aux moon head is the router; it plays when 0.10 < p ≤ 0.26 (exactly the v6 behaviour) |
| tier-2 specialist | equity_data/exploiter_r9/B1/specialist_B1_end.pth (arm a further trained inside the ensemble, league round 9 cell B1) | 50492c6d | plays when p > 0.26 — the top quarter of gated decisions, ~1.2% of all |

p = the router's maximum opponent moon probability, conjoined with an
opponent moon-alive flag (public information only; no points term).
Same three forwards as the v6 module — the router net's policy is a
by-product of its own forward — so serving cost is unchanged. Inputs:
obs v2 (882 dims). Outputs: masked policy logits + value (belief via
forward_all from the default net).

Training: the tier-2 specialist is v6 arm a trained by PPO inside the
ensemble with the default net and router frozen (gradients reach only
gated decisions), against a population of moon-shooting attackers,
recording only shooter matches, with a −2.0 penalty on the deal's last
gated decision whenever an opponent shot the moon
(docs/exploiter_league_r9_prereg.md, cell B, run 1). The router tier
threshold 0.26 was fixed by an outcome-blind exposure census and
selected mechanically on a registered screen
(docs/exploiter_league_r11_prereg.md §3–§4).

Registered evaluation vs the v6 ensemble 8d7816d1
(docs/exploiter_league_r11_results.md):
- Match non-inferiority n=6,400 mixed anchors: Δplace +0.0020 (SE
  0.0034, UB95 +0.0075 vs bar +0.030); win rate 51.05% vs 50.98% (LB95
  −0.003 vs bar −0.025); Δscore −0.04/match.
- SEL search-shooter defense gate, fresh-seed primary n=256: 1.816 →
  1.500 moons conceded/match (−0.316, SE 0.091, p=0.0003; pooled n=320
  −0.325). Versus the v5 champion on the same shards: 2.544 → 1.522
  (−1.02, SE 0.07).
- Guard: promotion is raw-only; the served search substrate stays the
  champion's traces (3a2abd36 / efdfee07), verified unchanged.

Lineage of the number: v5 → v6 cut moons conceded by 29%; v6 → v6.1 by
a further 17% (40% below v5), with ordinary play statistically
indistinguishable at every step.

Limits, honestly: defense measured against ONE attacker family (the
search shooter + its clones); no claim about human attackers; the
tier-2 specialist is one training seed (its sibling run B2 was weaker);
the arm was the cheapest of three on a noisy screen, so its screen
cost (−0.022/deal) is biased low — the NI number (+0.002 placement) is
the honest one; this was the specialist's second NI attempt (the
single-tier version failed round 10 by 0.002 placement; disclosed
before the run). Weights: models-v1 release assets
`hearts_ensemble_710c2102.pth` (checkpoint) and
`hearts_ensemble_710c2102_882trace.pt` (served-form trace, md5
85ee0851), with ENSEMBLE_V6_1_MD5SUMS / _SHA256SUMS alongside; never
tracked in git.
