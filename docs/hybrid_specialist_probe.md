# Hybrid "defense specialist" probe — registration (2026-08-18) + result

**Question.** Does the obs-v2 student's defense (T1: arm b −0.117 vs the
champion's defenders) live in its THREAT-STATE play, and what does it
cost in strength to hand only those decisions to it? A hard switch
between two frozen raw nets — the champion (8a89da90, 556 inputs) on
every decision except moon-alive threat states, arm b
(`v6_stage3/armb_lr3e-4.ep4.pth`, 882 inputs) on those — is not a
perturbation of either net (r5's fragility failure does not apply). If
it works it is both a product candidate ("the best raw AI may be several
raw nets that switch") and the calibration a round-6 aimed-teacher design
needs. Registered BEFORE running; measurement of existing artifacts,
$0, ~40 min; nothing trains; champion untouched.

**Instrument.** `HeartsHybrid` (hearts_net.py) with two gates:
- **threat** (primary): some OPPONENT relative seat is moon-alive
  (obs-v2 872–875) AND holds ≥ 1 penalty point (taken-by planes ×
  penalty). Fires on 19.1% of play decisions on the v6 holdout, 0% in
  passing. Checkpoint `hybrid_champ_armb_threat.pth` md5 689cf4a4.
- **any_alive** (secondary, informs): any opponent moon-alive flag set;
  51.2% of play decisions. `hybrid_champ_armb_anyalive.pth` md5 bbb7cd84.
Null contracts verified on 80,220 holdout states: non-gated outputs ==
champion, gated outputs == arm b, exactly. Harnesses unchanged: fast
defense probe (n=1,000 CRN, block 740M, base = champion defenders) and
paired neutral-raw strength vs the champion (n=5,000).

**Readouts (registered).**
- R1 defense: Δ moons conceded (threat hybrid vs champion). "Threat-state
  play carries the defense" if Δ ≤ −0.06 with UB95 < 0 (≥ half of arm b's
  −0.117); Δ ≈ 0 → the defense lives in non-threat play (passing /
  early-deal shaping) and switching cannot capture it.
- R2 strength: paired per-deal Δ vs the champion with 95% CI. UB95 ≤ 0 →
  not weaker on the screen (a match NI n=3200 would then be the real
  test before any product claim); UB95 > 0 → the cost is quantified and
  weighed against R1.
- any_alive gate: same two numbers, informs (a broader switch trades more
  strength for possibly more defense).
- Nothing here gates or promotes; a promising result → round-6 prereg
  (aimed teacher and/or hybrid NI/guard battery).

**Result (2026-08-18 12:35→13:34; equity_data/verdicts/hybrid_probe_fastprobe.json,
logs/hybrid_probe.log):**
| hybrid | gate rate (plays) | defense Δ vs champion defenders (SE), CI | p | strength vs champion (per deal, SE) |
|---|---|---|---|---|
| threat | 19.1% | **−0.107 (0.036), [−0.178, −0.036]** | 0.0016 | **+0.104 (0.055)**, UB95 +0.21 |
| any_alive | 51.2% | −0.147 (0.041), [−0.228, −0.066] | 0.0002 | +0.246 (0.093), UB95 +0.43 |
R1: **threat-state play CARRIES the defense** — 19% of decisions recover
essentially all of arm b's −0.117. R2: strength cost +0.10/deal on the
screen (UB > 0) — quantified; the match NI (placement) is the real
"overall better player" test and is the registered next measurement. The
broader gate buys +0.04 defense for 2.4× the cost — the value sits in the
threat states. Defender placement vs the shooter field: −0.038 / −0.035
(better). Nothing gated; nothing trains; champion untouched.
