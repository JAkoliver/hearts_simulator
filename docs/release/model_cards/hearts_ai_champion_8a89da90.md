# Model card — champion 8a89da90 (raw network of the current era)

> STATUS: DRAFT - not yet released; project ongoing.

## Identity
- **Files:** hearts_model_final.pth (checkpoint, md5 8a89da90);
  traces: hearts_ai_match_8a89da90.pt (match-context, md5 6fe062fb),
  hearts_ai_grandmaster.pt (raw-play engine trace, md5 ccb71f51),
  hearts_ai_search.pt (search-path trace, md5 e755b9b3). Trace
  filenames are engine identifiers, not strength claims.
- **Architecture:** HeartsNetV5 card-token transformer, d=320, L=6,
  7.6M params; 550-dim observation + 6-dim match context
  (docs/release/ARCHITECTURE.md sec. 2).
- **Date promoted:** 2026-07-29, match promotion #5, milestone
  1785322724 (ledger 2026-07-29).

## Provenance
Fifth and final promotion of the match era: v5 architecture promotion,
first raw-line PPO promotion, then five match-mode PPO promotions
(#1-#3 at n=800 on 2026-07-24; #4-#5 at the re-powered n=3,200 on
07-28/29). Full trail in docs/release/RESULTS.md "Match era" and
"Evolved regime" tables.

## Role in the record
The deployed raw network: every era-8/9/10 candidate was gated against
it, and it seats the AI on the live site. Every later self-improvement
recipe failed to beat it (RESULTS.md eras 8-10), producing the capacity
verdict that motivates the v6 campaign.

## Measured strength
- Promotion gate #5: placement **-0.031** (n=3,200 paired matches,
  p=0.0247), match wins **53.3% vs 50.5%** (McNemar discordant 499:407,
  p=0.0012), search guard UB +0.258 (n=4,800) (ledger 2026-07-29).
- Analyzer vs v4-m10 field: 6.11 avg pts/deal and 36.1% deal wins vs
  v4-m10's 7.35 / 27.3%; solo diff -2.134 (2,000 deals, 16,000
  games/side; analyzer_history.csv 2026-07-28).
- Opponent-field caveat: all numbers are vs the stated anchor fields
  and configurations; nothing generalizes past them without new
  measurement.

## Known weaknesses (stated honestly)
- Moon defense: 51.5% vs v4-m10's 61.1%; moons conceded 272 vs 151
  over 2,000 deals (analyzer_history.csv 2026-07-28). Confirmed
  in-instrument by the exploiter league (v4-m10 holds the SEL attacker
  to 0.237 moons/deal vs this lineage's defenders at 0.367;
  docs/exploiter_league_phaseA.md). League rounds 1-3 improved defense
  only at the cost of failed protection gates; the hole stands in the
  deployed net.

## Intended use
Raw argmax play (engine/web), or inside the search player with the
K=64 / K=256-endgame schedule. As a gate baseline, use exactly these
files; the promotion hash identifies the artifact.
