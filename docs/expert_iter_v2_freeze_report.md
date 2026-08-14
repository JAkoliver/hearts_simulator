# Expert-iter v2 recipe freeze report - anchor coefficient

The anchor coefficient weights KL(candidate || champion) on
non-confident (flat) states: LOW = absorb more teacher signal,
risk overwriting real knowledge (the v1 failure); HIGH =
protective, may damp learning through the shared trunk.

**Caveat that makes this a human decision:** v1's disaster
had GOOD imitation metrics. Nothing below measures played
strength - only the comparative stage does that.

| coef | seed | conf match | value EV | belief BCE | entropy ratio | non-conf KL |
|---|---|---|---|---|---|---|
| 0.25 | 909 | 0.5872 | 0.292 | 0.1774 | 3.044 | 1.66352 |
| 0.25 | 910 | 0.6008 | 0.299 | 0.1760 | 3.006 | 1.57762 |
| 1.0 | 909 | 0.6592 | 0.299 | 0.1768 | 2.361 | 0.67001 |
| 1.0 | 910 | 0.6592 | 0.302 | 0.1757 | 2.339 | 0.65175 |

**Means:**

| coef | conf match | entropy ratio | non-conf KL |
|---|---|---|---|
| 0.25 | 0.5940 | 3.025 | 1.62060 |
| 1.0 | 0.6592 | 2.350 | 0.66090 |

- conf match: holdout teacher agreement on confident states (signal absorbed)
- non-conf KL: policy drift from the champion in flat states (knowledge disturbed; the thing the anchor protects)
- entropy ratio: candidate/baseline on flat states; hard constraint <= 2.0 (v1's un-sharpening signature)

## Recommendation: **None**

BOTH coefficients violate the entropy constraint (ratio > 2x baseline) - the recipe itself looks broken; HALT and investigate before any mix runs.

To continue the pipeline with your chosen coefficient:

    nohup bash ops/auto_v2_continue.sh <0.25|1.0> &

## HALT EXPLORATION (2026-08-04 ~03:00, holdout-only, pre-unblinding)

Both pre-specified coefficients violated the entropy diagnostic, halting
the pipeline. Under the halt, additional HOLDOUT-ONLY recipe arms were
measured (no mix has been evaluated; nothing here touches match play):

| arm                | conf match | entropy ratio (<=2.0) | non-conf KL |
|--------------------|-----------:|----------------------:|------------:|
| coef 0.25, ep3     |     0.594  |              3.025 X  |      1.621  |
| coef 1.0,  ep3     |     0.659  |              2.350 X  |      0.661  |
| coef 1.0,  ep2     |     0.659  |              2.318 X  |      0.653  |
| coef 1.0,  ep1     |     0.600  |              2.574 X  |      0.991  |
| coef 2.0,  ep3     |     0.679  |              2.011 X  |      0.381  |
| coef 3.0,  ep3     |     0.687  |              1.859 OK |      0.282  |
| coef 4.0,  ep3     |     0.688  |              1.762 OK |      0.229  |

Findings:
1. MONOTONE DOSE-RESPONSE: a stronger anchor improves EVERY axis at
   once - teacher match rises, flat-state drift falls, entropy falls.
   The anchor behaves as a regularizer, not a brake on learning.
2. Epoch count is NOT the lever: entropy inflation happens in epoch 1
   (ep1 is WORSE on all axes); later anchored epochs partially recover.
3. Saturation between coef 3.0 and 4.0 (match plateaus ~0.687-0.688).
4. coef 4.0 dominates 3.0 (match tied, less drift, lower entropy).

RECOMMENDATION: freeze anchor coef = 4.0 (dominant, passes the
entropy diagnostic with margin). Requires a USER-SIGNED amendment: the
prereg's candidate set {0.25, 1.0} is superseded by this holdout
measurement - both original values are diagnostically inadmissible, so
an amendment is needed no matter what; the exploration was conducted
entirely on holdout data before any comparative evaluation existed.

To continue after approving:  nohup bash ops/auto_v2_continue.sh 4.0 &
