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

**Match NI gate (2026-08-18 14:20→15:05, scripts/run_match_gate.py, n=3200
mixed anchors, 12 workers; equity_data/verdicts/hybrid_threat_ni_n3200.json):**
threat hybrid vs champion placement Δ **+0.001 (SE 0.010), UB95 +0.017 ≤
+0.030 → NON-INFERIOR**; win 52.4% vs 52.6% (discordant 201:208); 2,675 s.
The tiny SE reflects that the hybrid IS the champion on 81% of decisions.
Together with the fast probe (−0.107 defense) this is the first candidate
since r1-t3 to clear both the defense direction and match non-inferiority.
Not yet run: the registered SEL defense gate n=320 and the search guard on
the ensemble as one traced module.

**Gate ladder: INTERRUPTED (ops incident 2026-08-18 15:05→16:19).** The
7-rung ladder was launched as ONE fast-probe call (8 arms per decision, 15
network forwards per step per worker, 12 workers at Normal priority,
daytime). The machine became unresponsive (display would not wake) and was
hard-powered-off at 16:19; no kernel fault or resource-exhaustion event was
logged - CPU starvation. Fixes: CPU pools now BelowNormal by default
(headroom.apply_process_priority), hybrid loader in-memory, ladder to be
rerun as separate ≤3-arm probes at ≤8 workers. Nothing corrupted; champion
verified 8a89da90.

**Gate ladder (rerun GENTLY 2026-08-18 16:47→22:26: 3 probes of ≤3 arms + 7 paired-
strength reads, 8 workers BelowNormal, desktop usable throughout;
equity_data/verdicts/hybrid_ladder_fastprobe_{pts,moonhead,uncertain}.json,
logs/hybrid_ladder_gentle.log). Base arm on the 8-worker layout: 1.520.**
| gate | fires (plays) | defense Δ (SE), CI | strength Δ/deal (SE) |
|---|---|---|---|
| threat:3 | 12.8% | −0.169 (0.032), [−0.232, −0.106] | −0.015 (0.032) |
| threat:6 | 10.9% | −0.168 (0.031), [−0.228, −0.108] | −0.019 (0.037) |
| threat:10 | 9.8% | −0.148 (0.030), [−0.207, −0.089] | −0.003 (0.027) |
| **moonhead:0.1** | **9.6%** | **−0.247 (0.032), [−0.310, −0.184]** | **+0.014 (0.031)** |
| moonhead:0.3 | 3.3% | −0.182 (0.021), [−0.223, −0.141] | −0.030 (0.026) |
| moonhead:0.5 | 1.5% | −0.072 (0.013), [−0.097, −0.047] | −0.015 (0.017) |
| uncertain:1.0 | 3.7% | +0.026 (0.027), [−0.026, +0.078] | +0.049 (0.039) |
Reading: the specialist's own aux MOON HEAD is the router — at τ=0.1 the
ensemble concedes −0.247 moons/match (≈ r1-t3's −0.235; 2× arm b's own
−0.117) at zero measurable strength cost; every stricter gate beats the
≥1-point rule on both axes (that rule's +0.10/deal was the price of
switching in low-value states); the champion-uncertainty gate is dead.
Estimation only (8 rungs, selection noise modest at these SEs); the
chosen rung(s) go to the registered battery (NI n=3200 → SEL defense gate
n=320 → search guard) as one traced 882 module.
