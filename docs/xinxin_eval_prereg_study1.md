> *Published as registered (Study 1 of the Perilune v5 vs. xinxin report, docs/xinxin_eval_report.md). The approval signature sentence and the author email address have been replaced by a registration date, and the in-text references to signing now read "registered"; the registered text is otherwise unchanged (md5 of the original file: 4e304f217fa0a4de54c77a37f9662952). The title carries the working name "DRAFT" the document had at registration; "the user" throughout is the author, who directed and approved the study the AI drafted.*

# PRE-REGISTRATION (DRAFT) — Perilune vs xinxin

Status: PILOT-FILLED 2026-08-14, awaiting registration. All `[PILOT]` and
`[FREEZE]` blanks are filled from the variance pilot (HANDOVER.md step 1)
and the instrument md5s; the design is then registered and becomes binding.
House rules apply (halt-default, one pre-unblinding amendment, registered
analysis frozen before data).

## 1. Question and claims to be supported

Primary: which engine is stronger at four-player hearts played as matches
to 100 points (pass rotation L/R/A/H, 2♣ leads, Q♠=13, moon 0/26/26/26,
hearts break only on a heart)?
Secondary: how does the answer depend on search budget, and how much of
xinxin's search does Perilune's raw policy net equal?

Engines: Perilune champion (hearts_ai_search_match.pt md5 3a2abd36 +
hearts_equity.pt md5 efdfee07, lineage 8a89da90; both md5s re-verified
unchanged 2026-08-14 before the pilot) vs xinxin (git
6ba1154-dirty; local patches: cassert include, kLead2Clubs enabled;
rules mask 0xda1, kQueenBreaksHearts OFF to match Perilune).

## 2. Design

Complement-paired duplicates: each seed played twice with engine seat
assignment flipped (verified: identical deals, so deal luck cancels
within a pair). Unit of analysis = the pair. Strata run and reported
separately, never pooled as primary:
- 2v2 (rows 4-9): the symmetric headline.
- 1v3+3v1 (rows 0-3 with complements): the challenger view.

Arms (M = matches to 100; D = single-deal vacuum with pass rotation and
start-of-match context):
- M-A: both at shipped defaults (Perilune K=64/256 pass_k=24; xinxin
  sims=10,000 worlds=30 ε=0.1 OM-2).
- M-B: no search either side (Perilune raw sequential policy; xinxin
  DoMinPlay + selectPassCards).
- M-C: xinxin at sims=93,000 (wall-clock-parity point, RE-VERIFIED on
  the idle machine immediately before the run; playout counts published
  beside it) vs default Perilune.
- D-A: deal vacuum at both defaults.
- SWEEP (estimation, not testing): xinxin sims x ∈ {300, 1k, 3k, 10k,
  30k, 93k} vs Perilune raw, 2v2 pairs, same seeds at every x (CRN);
  crossing of D(x)=0 estimated by monotone fit in log x, CI by
  bootstrap over pairs.

## 3. Registered metrics

Per arm and stratum, computed by the FROZEN analyze.py:
- PRIMARY: paired mean-placement difference (tie-averaged), one-sample
  t over pairs. CO-PRIMARY: paired match-win difference (tie-split),
  t over pairs. Holm correction over the two, per arm.
- ROBUSTNESS (reported alongside, same data): Wilcoxon signed-rank on
  the paired placement differences and an exact sign test on pair
  winners - D is bounded in [-2, +2] and n is modest, so the t's
  normality lean deserves a distribution-free check.
- SECONDARY (reported with CIs, never gating): mean-score difference;
  per-deal points (unit = per-match deal-mean); deals/match; moons
  completed; budget table (playouts + ms per decision, both engines).
- Sign convention: negative placement/score favours Perilune; positive
  win difference favours Perilune.

## 4. Sample sizes (rule fixed BEFORE the pilot; no optional stopping)

Pilot: **12 complement pairs** (24 matches, rows 4-9 cycled exactly
twice - row-balanced), 2v2, M-A settings, seed 20260814. The pilot
yields ONLY the paired SD; its effect direction is not evidence and its
pairs are EXCLUDED from every arm. (8 pairs was the first draft; at
n=8 the SD estimate carries a ±40-90% multiplicative CI, which can
mis-set n by 2-4x. 12 is the compromise; the ~50 extra minutes are
cheap insurance.)

**n rule (mechanical, no discretion):**
- M-A 2v2 (primary): n = ceil((2.80 · SD_pilot / 0.35)²), rounded UP to
  a multiple of 6 (the six 2v2 rows differ in seat ADJACENCY, which
  changes passing targets - unbalanced rows would confound), min 18,
  cap 54. Powered at d = 0.35 places: a third of a place per seat is
  the smallest gap we call clearly meaningful. Reference points from
  the SD band 0.5-0.9: n = 16-52.
- **Minimum-detectable-effect clause:** the writeup states the MDE at
  the chosen n. A null result claims "any advantage is smaller than
  [MDE] places", never "no difference". Detecting a razor-thin true
  gap (d < 0.15, n > 150-280 pairs) is out of scope for this study
  and said so.
- M-A 1v3+3v1 (secondary): n = half of the 2v2 n, rounded up to a
  multiple of 4. It answers a robustness question, not the headline.
- M-B: same n as M-A (costs minutes).
- M-C: half of M-A's n (min 12 pairs, multiple of 6). Its registered
  purpose is DIRECTIONAL - does the M-A result survive giving xinxin
  313x the playouts at equal wall-clock - so it is powered for
  d = 0.5, not 0.35. At ~6 min/match this is still the most expensive
  arm; if the pilot SD lands high, the cap is 24 pairs and the writeup
  says so.
- D-axis: 150-200 complement-paired deals at M-A settings (~2-3 h),
  paired per-deal SD taken from the pilot's logged deals; powered for
  ~1.5 points/deal. [PILOT fill: SD_paired_per_deal = 5.844 over 107
  paired deals -> required n = ceil((2.80*5.844/1.5)^2) = 120; registered
  n_D = 156 paired deals (multiple of 12: balances the six 2v2 rows x
  four pass directions; inside the registered 150-200 envelope, >= 120).]
- SWEEP: 8 pairs per x-level (48 matches total across 6 levels),
  estimation only - no per-level significance is claimed.
- **D-axis joint coverage note (author, at registration):** rows cycle mod 6
  and pass directions mod 4, so only the 12 parity-consistent
  (row, direction) combinations occur — e.g., seating row 4 never plays
  a Hold deal. Marginal balance (each row 26×, each direction 39× at
  n=156) holds exactly; the joint cross is half-covered by construction
  of the frozen runner. The D-axis estimand is the mean over this
  realized design.

**Seeds: ONE shared block across M-A, M-B, M-C** (base 30260814 +
7919·i per pair, disjoint from the pilot): identical deals across arms
make cross-arm contrasts (the value of Perilune's search on the same
deals: A vs B) computable as paired statistics. Cross-arm contrasts
are labeled exploratory. The sweep reuses the block at every x (CRN).

**Seed-sharing note (author, at registration):** seed sharing across arms and
strata is deliberate CRN: M-A/M-B/M-C and both strata draw from the
same block, so the same deals recur across arms — cross-arm and
cross-stratum contrasts are paired by design, and the report will
describe this rather than treat arms as independent samples.

Expected totals if SD_pilot ≈ 0.65: M-A 30+16 pairs, M-B 30, M-C 15,
D ~180 deals, sweep 48 pairs → **~250 matches ≈ 2,100 deals ≈ 110k
play decisions per engine, ~11-14 h GPU.**

**PILOT RESULTS (filled 2026-08-14, mechanical rule applied, no
discretion).** Pilot ran exactly as registered (12 pairs, 24 matches,
seed 20260814, rows 4-9 cycled twice, arm M-A settings); verifier: 237
deals, 0 hard errors, 0 warnings, 0 logger err records, 0 incomplete
pairs.
- SD_pilot (paired 2v2 D_placement, n=12 pairs) = **0.555192**
- n rule: (2.80 · 0.555192 / 0.35)² = 19.73 → ceil 20 → up to multiple
  of 6 = **n_MA-2v2 = 24 pairs** (within min 18 / cap 54)
- **M-A 1v3+3v1 = 12 pairs** (half of 24, up to multiple of 4)
- **M-B = 24 pairs** (same as M-A)
- **M-C = 12 pairs** (half of M-A, min 12, multiple of 6; under the
  24-pair cap)
- **D-axis = 156 complement-paired deals** (see §4 D-axis fill;
  SD_paired_per_deal = 5.844)
- **MDE at the chosen n (80% power, two-sided α=0.05):** M-A 2v2:
  2.80·0.555192/√24 = **0.317 places/seat**; M-A 1v3+3v1 and M-C
  (n=12): **0.449 places/seat** (M-C registered target d=0.5:
  satisfied); D-axis (n=156): 2.80·5.844/√156 = **1.31 points/deal**.
  Any null is reported as "any advantage is smaller than [these]",
  never "no difference". (SD estimates come from the 2v2 pilot; the
  1v3+3v1 stratum's own SD may differ — its MDE is indicative.)
- Pilot pairs (seed block base 20260814) are EXCLUDED from all arms;
  the arms use base 30260814 + 7919·i per §4. The pilot's effect
  direction is not evidence and is not repeated here.

## 5. Instruments (frozen at registration)

- perilune_tourney binary: md5 339baa34ad4908f5c70b41c43a386d08
  (amendment 1 build, 2026-08-14; supersedes fdce0ac33cf3cf041385a302078ccddf
  — see AMENDMENT 1 below; all three smokes green on this binary)
- analyze.py: md5 200ad38599a64f75c0009df243ddbad9
  verify_games.py: md5 41a6440ff7fac9264967ec731701350d
- verify_games.py MUST report 0 hard errors on every arm's games.jsonl;
  any error voids the arm (fix, re-run the arm fresh - no patching data).
- Bridge integrity: any [perilune-bridge] FATAL or [gamelog] WARNING
  voids the affected run.

## 6. What results are allowed to mean

- M-A decides the headline claim at defaults; M-B isolates policy
  quality; M-C bounds the "compute excuse" (xinxin dominant in BOTH
  budget currencies); the sweep prices the raw net in sims.
- If Perilune wins M-A but loses M-C: the claim degrades to "stronger
  at practical budgets"; report both, no cherry-picking.
- If arms disagree with the pilot direction, the pilot is NOT evidence
  (it sized variance only).
- Exploratory §10.4 analyses (search-judged decision quality, belief
  calibration, moon microscopy, equity trajectories) are labeled
  exploratory and never support the headline claims; evaluator bias
  (Perilune judging both sides) stated wherever used.

## 7. Threats to validity (to appear in the writeup)

Single opponent (claims are "vs xinxin", not "vs MCTS"); single machine
(93k is machine-specific; playouts published); objective mismatch
(match-aware vs point-minimizing - two-axis reporting is the
mitigation); opponent-model mismatch both directions at mixed tables;
his in-search passing vs heuristic passing in M-B; ε=0.1 playout noise
is part of his agent; GPL/MIT separation constrains the shared arena to
his harness (by design).

## ADDENDUM A (pre-unblinding repair, 2026-08-14 — not the amendment)

§3 registers Wilcoxon signed-rank, an exact sign test on pair winners,
Holm over the two primaries, and CIs for the secondaries — but the
frozen analyze.py emits only means/SE/t. `supplement_stats.py`
(md5 00f0f084134364b39ace56a76b535404) implements these
already-registered final steps WITHOUT touching any frozen instrument
(it imports analyze.py read-only and reuses its contrast construction,
verified bit-identical on the pilot). Written, self-tested (t CDF and
quantile vs known values; exact Wilcoxon and binomial on hand-checked
cases), and md5-frozen BEFORE any arm result was read; validated only
against the pilot, whose direction is registered non-evidence.
Conventions pinned in its header: all tests two-sided; exact
signed-rank (midranks for ties; zeros discarded with count reported;
exact DP null distribution); exact binomial sign test (split pairs
discarded with count reported); Holm for m=2; 95% t CIs;
dependency-free Python stdlib (no scipy). This is a repair (implements
registered analyses), like the Stage-4 stop-rule clarification — the
single pre-unblinding amendment remains AMENDMENT 1 below.

## AMENDMENT 1 (pre-unblinding, 2026-08-14 — the one permitted amendment)

Found after registration, before any arm ran: the frozen runner rotated the
vacuum pass direction by MATCH index, and with --pair-complement the
match index increments within a pair — so the two sides of every D-axis
pair would have passed in different directions (side 0 always Left/
Across, side 1 always Right/Hold): engine assignment confounded with
direction parity, and §2's within-pair deal-luck cancellation false for
the D-axis. This contradicts the premise of the registration-time D-axis
note, which assumed within-pair direction consistency.

Change (user-approved): one line in perilune_tourney.cpp — pass_offset
now rotates by PAIR index (pair_id % 4), identical across both sides of
a pair. With the fix, the registration-time note describes the code exactly:
row parity locks to direction parity, 12 parity-consistent (row, dir)
combinations, each direction 39 pairs at n=156. M arms are unaffected
(offset stays 0 without --rotate-vacuum-pass; smoke_v2 reproduces
pre-fix results bit-identically on deals and scores).

Verification on the amended binary: smoke_v2 (8/8 paired deals
identical hands, 0 FATALs), smoke_vacuum (dirs 0,1,2,3), new
smoke_vacpair (--pair-complement + --rotate-vacuum-pass: within-pair
same direction, dir == pair%4, verifier 0 errors, 0 warnings).
Binary md5: fdce0ac33cf3cf041385a302078ccddf →
339baa34ad4908f5c70b41c43a386d08. analyze.py and verify_games.py
unchanged. No counted data existed at amendment time.

## AMENDMENT 2 (2026-08-14, crash-forced repair; user-approved with
## required record below — exceeds the one-amendment budget, documented
## openly; blinding intact at approval)

**(1) Triggering state (evidence, captured verbatim by a diagnostic
build with pre-fix semantics; deterministic reproduction of the arm
crash):** arm M-A 2v2, match 13/24, pair 12 side 0, seating row 4,
seed 30355842 (= 30260814 + 7919·12), deal_idx 5, trick 1 (n_play=3,
4th card of the first trick), acting seat 2. Acting hand (Perilune
ids): [36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51] = Q♠ + 12
hearts (all-penalty hand; K♥ absent). Perilune chose action 36 (Q♠);
xinxin's getAllMoves listed only the 12 hearts. Root cause: Perilune's
HeartsEnv legalizes penalty cards on trick 1 when the hand holds ONLY
penalty cards; xinxin's kNoQueenFirstTrick (bit 0x100 of the registered
mask 0xda1) excludes Q♠ unconditionally and his all-penalty fallback
readmits hearts only. The repair was crash-forced: the same seeds
deterministically reach the same state, so re-running without a fix
cannot complete the arm.

**(2) Blinding statement:** 12 of 24 pairs (24 matches) had completed
when the guard aborted. During diagnosis, nothing was unblinded: no
analyze.py or supplement_stats.py run touched any arm file; no RESULT
or DEAL line of any arm was read; extraction used only match-header
markers ("=== match ..."), line COUNTS (RESULT count 24, games.jsonl
line count 251, deals-in-crash-match count 5), stderr (BUDGET/FATAL
lines), and the evidence reproduction in runs/amd2_evidence (a
non-counted directory).

**(3) The fix (bridge files only; xinxin's tree and the research repo
untouched):** the arena's move generator (getAllMoves) now defines root
legality for Perilune. PeriluneHeartsPlayer fills Replay.arena_legal
BEFORE the decision. In perilune_engine.cpp: (a) raw path — the policy
mask is env-legal ∩ arena-legal; (b) search path — if the search's pick
is arena-illegal, re-select by the search's OWN rollout scores
restricted to arena-legal candidates (equal determinization counts per
candidate make argmax(mean) identical to ChooseAction's argmax(sum)
restricted to the subset); (c) if the intersection is ever empty,
selection among arena-legal actions by raw-policy logits under an
arena-only mask (the trace accepts arbitrary masks; the search mode now
holds a raw copy of the policy module for exactly this path). Passing
takes no arena filter (both rule sets pass anything). Every other
cross-check (hand/seat/legality/desync) stays fatal — no silent
substitutions anywhere else; the post-decision legality check remains
and now carries a full state dump. Discovered and fixed in the same
pass: ChoosePass's mode test updated (raw_mod alone no longer
distinguishes raw from search mode) so search-arm passing still goes
through the pass search — verified by smoke budgets; and build_bridge.sh
now removes stale bridge objects before compiling (a stale object
briefly masked the fix during verification — caught by the fix-check).

**(4) Data disposition:** the partial M-A 2v2 output was moved to
runs/arm_ma_2v2.VOID.amd2 (kept as evidence, excluded from all
analysis) and the arm re-runs FRESH on the same registered seeds under
the amendment-2 binary. No other arm had started. Every counted match
in the study therefore shares one binary provenance:
md5 8b155bb97286599262d756ff5aefcfa5.

**(5) Verification on the amendment-2 binary:** fix-check on the
crashing seed (pair 12): exit 0, zero FATALs, verifier 0 hard errors,
and the formerly-fatal state resolves to a heart play (verifier
warn-line "point card on trick 1", the anticipated warn-only class).
smoke_v2: identical deals AND identical scores to the pre-fix binary
(8/8 paired deals identical hands, 0 FATALs); smoke_vacuum: dirs
0,1,2,3; smoke_vacpair: within-pair same direction, dir == pair%4;
verifier 0 hard errors, 0 warnings on both vacuum smokes. Binary md5:
339baa34ad4908f5c70b41c43a386d08 (amendment 1) →
8b155bb97286599262d756ff5aefcfa5 (amendment 2). analyze.py,
verify_games.py, supplement_stats.py unchanged.

**(6) Pilot standing:** the pilot ran with ZERO bridge FATALs (its 237
deals never reached the divergent state), so its data is unaffected by
either the bug or the fix; the n-rule inputs (SD_pilot = 0.555192,
SD_paired_per_deal = 5.844) stand as registered.

**(7) Reporting:** the report carries a "Deviations from
preregistration" section listing Amendment 1 and Amendment 2 with
dates and rationale, citing the v6-prereg "instrument bugs: fix and
re-verify" precedent for why a crash-forced instrument repair does not
spend a scientific claim.

Amendment 2 APPROVED by user in session, 2026-08-14, with the seven
record requirements above.

## AMENDMENT 3 (2026-08-14, crash-forced repair; user-approved
## "fix + full re-run"; blinding intact)

**Trigger:** the amendment-2 chain's arm M-A 1v3 FAILED verification
with 93 hard errors, all "hands differ across sides", every deal index
≥ 1, every pair (deal 0 always identical). The 2v2 arm before it (478
deals) verified clean. Detected by the registered verifier's
complement-hand check; the arm is void under §5.

**Diagnosis (probe evidence, hands only — no scores read):** a
diagnostic probe recovered the exact Deck::Shuffle seeds reproducing
the logged hands. Old dealer law: deal k = shuffle(base + 82 +
(82 + 16·x)·k) where x = number of xinxin seats at the table.
Measured: 1v3 side 0 (x=3): +82, +212, +342, +472; side 1 (x=1):
+82, +180, +278, +376; 2v2 arm (x=2 both sides): +82, +196, +310,
+424 identical across sides. Mechanism: HeartsCardGame machinery
creates child states via cs->init(..., SEED++) (CardGameState.cpp:846,
Hearts.cpp:978), post-incrementing the PARENT's shuffle counter at a
composition-dependent rate. Consequences: (a) complement pairing was
broken for every unequal-composition stratum (1v3/3v1) beyond deal 0;
(b) the registered cross-arm CRN premise ("identical deals across
arms", §4 and registration note 2) held ONLY for deal 0, since arms with
different xinxin search settings clone at different rates. The pilot,
smokes, and 2v2 arms could never have caught this: equal xinxin counts
across sides kept the drift synchronized.

**Change (one line in OUR faithful Play copy, perilune_tourney.cpp;
no xinxin source touched):** the per-deal Reset now passes an explicit
seed: Reset(match_seed + 257·deal_index), using his existing
Reset(NEWSEED) path. Deal k of pair i is a pure function of (i, k):
identical within pairs in every stratum, and identical across all
arms/sweep levels on the shared seed block (CRN exactly true for all
deals). 257·deal < 7919 for any realistic match length, so deal-seed
windows never collide across pairs; pilot windows (base 20260814)
remain disjoint.

**Verification on the amendment-3 binary:** new smoke_1v3pair (the
divergence-triggering configuration: 1v3 complement pairs, SEARCH
engines both sides): exit 0, 0 FATALs, verifier 0 hard errors
0 warnings; probe confirms deal k = base + 257·k EXACTLY on both sides
for both pairs. smoke_v2 / smoke_vacuum / smoke_vacpair: all green
(smoke_v2's specific deals change BY DESIGN under the new law; its
invariants — 6/6 paired deals identical hands, 0 err records,
0 FATALs — hold). Binary md5: 8b155bb97286599262d756ff5aefcfa5
(amendment 2) → 5f5a905ad947667137213171ae7a287f (amendment 3).
analyze.py, verify_games.py, supplement_stats.py unchanged.

**Data disposition:** BOTH completed arms voided and moved aside —
runs/arm_ma_2v2.VOID.amd3 (it verified clean, but its consumption-based
deal streams match neither the fixed dealer nor the registered
cross-arm pairing, and single-binary provenance is required) and
runs/arm_ma_1v3.VOID.amd3 (the failed arm). The full chain re-runs
from scratch on the registered seeds under the amendment-3 binary; no
other arm had started.

**Blinding:** no analyze.py/supplement_stats.py run has touched any arm
file; no RESULT or DEAL line of any arm has been read. Diagnosis used
verify.txt error lines (hand-identity failures only), dealt-HAND
extracts (never scores), and the seed probe.

**Pilot standing:** the pilot's variance inputs (SD_pilot = 0.555192,
SD_paired_per_deal = 5.844) remain valid: it ran 2v2 (synchronized
drift, pairing intact, 0 FATALs, verifier clean). Its deal streams
differ from the new dealer's, but a variance estimate over exchangeable
deal streams does not depend on which streams were drawn; the n rule
stands as registered.

Amendment 3 APPROVED by user in session, 2026-08-14 ("Fix + full
re-run"), before any counted data was unblinded.

Registered 2026-08-14, with two notes incorporated at registration: the D-axis joint
coverage note (§4) and the seed-sharing/CRN note (§4). Binding from
this point: halt-default, no changes to the frozen instruments (§5)
without a registered amendment.
Amendment 1 APPROVED by user in session, 2026-08-14 ("Fix, amend
freeze"), before any counted run.
