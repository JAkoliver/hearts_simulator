# SDPA gate result report — 2026-07-17 (early morning)

## Verdict: PASS. SDPA promoted into production traces.

## Pooled numbers (criterion from docs/sdpa_gate_preregistration.md, commit e48d3ec)

| | n | delta mean (SDPA − old, + = SDPA worse) |
|---|---|---|
| Run 1 (seed 20260716) | 600 | +0.4700 |
| Run 2 (seed 555001) | 3000 | −0.0123 (SDPA vs anchor −0.0447, old vs anchor −0.0323) |
| **Pooled** | **3600** | **mean +0.0681, SE 0.1326, one-sided 95% UB +0.2862** |

- Criterion 1 — pooled mean < +0.20: **MET** (+0.0681)
- Criterion 2 — mean + 1.645×SE < +0.30: **MET** (+0.2862)
- Branch fired: **PASS**

Run 2 alone is dead parity (−0.012 ± 0.145); run 1's +0.47 was the noise the
pre-registration anticipated by pooling it against SDPA anyway.

## Commands executed (PASS branch, in order)

1. Pooled-criterion computation (Python/numpy over the four CSVs;
   parity_sdpa/old.csv n=600, parity2_sdpa/old.csv n=3000):
   printed run means, pooled mean/SE/UB, per-criterion MET/NOT MET, verdict.
2. Appended RESULT section to docs/sdpa_gate_preregistration.md.
3. `python export.py`
   → hearts_ai_grandmaster.pt, hearts_ai_search.pt re-traced (SDPA graph,
   same weights). v4m10 legacy traces untouched.
4. Post-export sanity (Python): traces vs eager `net_from_checkpoint` on
   4096 random obs/masks — max |diff| = 0.0 on logits (unmasked), value,
   belief for BOTH traces; argmax agreement 100.000%. (Threshold was
   < 1e-4; result was bit-exact since trace and eager now share one graph.)
5. Smoke: `SelfPlayGen.exe --model hearts_ai_search.pt --deals 10 --k 64
   --pass-k 24 --threads 14 --cuda --bf16 --seed 91` → exit 0, 640 records
   (matches the prior smoke count at this seed).
6. `git add hearts_net.py docs/sdpa_gate_preregistration.md && git commit`
   → 24bb45d "Promote SDPA attention: gate PASS (pooled +0.068 +/- 0.133,
   n=3600)".
7. Auto-memory updated (teacher-rate numbers, promoted stack, dead ends).
8. This report; committed separately. No further work started (no C1).

## Deviations from the pre-registered plan

Two, both operational, neither statistical:

1. **Run 2's sides ran sequentially, not concurrently.** The concurrent
   launcher had a shell scoping bug (`cd && VAR=… && cmd1 & cmd2 &`
   backgrounds the whole `&&` chain including the variable assignment, so
   cmd2 saw an empty output path and died at launch). The SDPA side
   completed normally; the reference side was relaunched alone with the
   identical command and seed. SearchEval outcomes are seed-deterministic,
   so scheduling affects wall-clock only. (Run 1 had hit the same bug and
   was also completed sequentially, before pre-registration.)
2. **The SDPA side's marginal mean was observed before the reference side
   ran**, unavoidably, while diagnosing that failure. No degrees of freedom
   remained: the criterion was already committed (e48d3ec) and the
   reference side's data was fixed by seed 555001. Disclosed here rather
   than omitted.

The criterion itself, the pooled data set, and the executed branch match the
pre-registration exactly.

## Standing follow-up (pre-registered as "not part of this gate")

A 400-deal steady-state run to replace the ~6.4 s/deal projection with a
measurement. Not started, per instruction to stop after this report.
