# Release documentation (DRAFTS - project not yet open-sourced)

Contract for everything in this directory:
- DERIVED documents. Primary sources (docs/speed_ledger.md,
  docs/experiment_rules.md, docs/ROADMAP.md, preregs, verdict JSONs,
  git history + notes) remain canonical; on conflict, the primary
  source wins and the release doc gets fixed.
- Every quantitative claim carries a citation: (ledger entry date),
  (verdict JSON), or (commit hash). No untraceable claims.
- Claims discipline (RELEASE_PLAN sec. 4.5): measured claims only,
  with N and CI; user calibration matches are n=1 anecdotes; no
  "superhuman" language.
- Updated as eras complete, not reconstructed at release time.

Reading order for an outside observer:
1. README_RELEASE.md - front door: what, headline results, build, play.
2. JOURNEY.md - the narrative: eras, experiments, failures, pivots.
3. ARCHITECTURE.md - the system: engine, net, search, training, gates,
   data formats, cloud, web app.
4. METHODOLOGY.md - the measurement discipline (paired deals, CRN,
   neutral anchors, pre-registration, halt-is-default, re-powering).
5. RESULTS.md - claim table: every major result, N, CI, citation.
6. REPRODUCING.md - toolchain pins, builds, rerunning gates,
   regenerating data from seeds.
7. model_cards/ - per released checkpoint.

Status: skeleton 2026-08-01. JOURNEY/ARCHITECTURE first drafts pending.

## Known documentation debts (from the 2026-08-01 drafting audit)
- [NEEDS CITATION] v5 -1.233 gate sample size (commit a4136b5 lacks n).
- [NEEDS CITATION] behavioral spine-gate pass/fail was never given a
  ledger entry / verdict JSON (tooling ea2d33e, behave_*.csv exist).
- flip_snr numbers differ between ledger 07-25 (36.8%/0.41/0.57) and
  equity_data/verdicts/flip_snr.json (38.9%/0.62/0.86); same HALT
  verdict both places - resolve against the analysis script pre-release.
- REPRODUCING.md needs a clean-machine build verification + Python pins.
- Stale figures in match_aware_search_design.md (N=8000 "local ~5.8h")
  are superseded by the ledger - noted in JOURNEY, doc left as-is
  (historical record).
