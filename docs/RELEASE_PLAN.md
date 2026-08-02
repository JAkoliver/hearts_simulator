# Open-Source Release Plan (living document)

Status: PLANNING ONLY. The project is NOT being open-sourced now. Release
happens only on the user's EXPLICIT go-ahead, as a deliberate final step.
This document exists so the project accumulates release-readiness as it
goes instead of paying an archaeology bill at the end.

## 1. Documenting the trajectory (the thing outside observers value most)

The project's primary sources are already append-only and honest - keep
them that way; they ARE the release documentation:
- docs/speed_ledger.md - every measurement, failure, wrong-then-corrected
  diagnosis, and cost, in order. Never rewrite history in it; corrections
  are appended (see the 07-30 wedge entry for the pattern).
- docs/experiment_rules.md - the measurement discipline and closed
  directions, each tied to the evidence that closed it.
- docs/ROADMAP.md - strategy with its revisions visible.
- docs/*_prereg.md + equity_data/verdicts/*.json - pre-registrations and
  machine-readable outcomes.
- Full git history - releases WITH history (it is the trajectory).
  Prerequisite: the pre-release secrets audit (sec. 4) passes.

At release, ADD a synthesis layer (do not replace primary sources):
- README: what this is, headline results, how to play it, how to build.
- docs/JOURNEY.md: narrative arc for outsiders - the v5 architecture
  insight, the match-objective pivot, the N=8000 validation, the expert-
  iteration failures and what they taught, with links into the ledger.
- A results paper/post if desired: per-claim, cite the ledger entry and
  verdict JSON that back it.
- Explicitly document the AI-assisted workflow and its EVOLUTION (of
  real interest to observers): the project began in the Antigravity IDE
  with Gemini 3.1 Pro as the coding assistant (early July 2026, the
  first commits through the early automation era), then moved to
  Claude Code (VS Code extension) with the Fable 5 model doing
  ops/implementation under human direction (the era documented
  throughout docs/speed_ledger.md). Switch date: 2026-07-09 (user
  recollection, corroborated by git: last old-toolchain commit 83f7437
  07-09 23:41, first structured-style commit f4963a2 07-13 08:39 after
  a transition gap). Both toolchains credited.

## 2. Data: preserve, release, organize

RELEASE (small, high-value, goes in git or a data release):
- equity_data/validation_v1/ (the N=8000 validation dataset) - committed.
- equity_data/verdicts/*.json, analyzer_history.csv, probe CSVs.
- Mix-experiment manifests + eval CSVs + results doc (v2 prereg sec.).
- All seeds/configs needed to REGENERATE any bank (the reproducibility
  contract (seed, chunk) -> data makes raw banks technically optional).

RELEASE AS BINARY ARTIFACTS (too big for git: GitHub Releases or a
HuggingFace dataset repo):
- expert_data/ v2 banks (~GBs; format documented in selfplay_gen.cpp
  header + MATCH_RECORD_V2). Include per-file sha256 manifest.
- Training banks for the distill lineage if desired (same route).

DO NOT RELEASE:
- hearts_web/match_logs.jsonl and hearts_match_log.csv (the user's
  personal play data) unless the user explicitly opts in.
- Any log containing machine paths/usernames beyond what the repo
  already shows; scrub logs/ from release artifacts (they are local
  working files, not tracked).

ORGANIZATION/SECURITY until release:
- Primary repo stays private. Big artifacts: private GitHub release
  assets (the pilot-bundle-v1 pattern) or an external encrypted backup.
- BACKUP discipline (do this soon, not at release): periodic copy of
  Hall_of_Fame/, equity_data/, expert_data/, and the repo to a second
  disk or private cloud. A single-disk project is one failure from
  losing the trajectory it wants to publish.

## 3. Models to save and release

Champion lineage (all already in Hall_of_Fame/ with hashes in the
ledger/experiment_ledger) - release the milestones that mark eras:
- v3-m7 and v4-m10 anchors (they define every measurement's opponent
  field - without them nobody can reproduce our numbers).
- v5-M first promotion (the architecture-insight artifact).
- Each match-era promotion milestone (1st..5th; hashes 25db6f93 ...
  8a89da90), or at minimum the 1st and the final champion.
- hearts_ai_search_ref_matchblind_20260724 (.pt + .pth, md5 a1a0be31) -
  the frozen reference of the N=8000 validation. NON-NEGOTIABLE for
  reproducibility of the headline result.
- hearts_equity.pt + equity_v1.pth (the validated equity model).
- Final champion: .pth + all traces (raw/search/match/equity) so both
  the raw net and the full match-aware search player run out of the box.
- Model card per released net: size/arch, training provenance, measured
  strength vs which field, KNOWN WEAKNESSES stated honestly (e.g. the
  moon-defense hole 51.5% vs v4's 61%), intended use.

## 4. Pre-release checklist (executed only on the user's go-ahead)

1. SECRETS AUDIT: scan full git history + all release artifacts for
   rpa_/github_pat_/gho_/PUBLIC_KEY-style material (2026-08-01 scan:
   clean; .mcp.json with the RunPod key was never tracked and is now
   gitignored). Rotate the RunPod API key regardless at release time.
2. PERSONAL DATA: decide on email/username in git commits (fine to keep
   or re-author; user's call), remove personal match logs (sec. 2).
3. LICENSES: code (suggest MIT or Apache-2.0), weights + data (suggest
   a permissive open-weight license); LICENSE files + per-artifact
   notice. third_party/cuda_include is NVIDIA-licensed - it must be
   EXCLUDED from the release and replaced with fetch instructions
   (check every third_party item's license before release). Ship an
   EXPLICIT fetch helper (e.g. scripts/fetch_cuda_headers.py: pip
   download nvidia-cuda-runtime-cu12==12.6.77, extract include/ into
   third_party/cuda_include) plus a CMake configure-time existence
   check whose error message states that exact command - NOT a silent
   configure-time auto-download (breaks offline builds, and keeping
   the user's fetch explicit keeps the NVIDIA license boundary clean).
4. REPRODUCIBILITY: pin the toolchain in README (MSVC/libtorch 2.12.1
   +cu126, the cloud/Dockerfile recipe already encodes the Linux path);
   PIN PYTHON DEPS: freeze the working env into a lock file
   (requirements.lock via pip freeze, torch matching the cu126
   libtorch flavour) - closes the [NEEDS CITATION] gap in
   docs/release/REPRODUCING.md sec. 1; scope it honestly (it makes
   training/eval run on a clean machine - the bit-identical data
   contract lives in the C++/libtorch side and rule #14's caveats);
   verify a clean-machine build from the public repo alone.
   DONE 2026-08-01: equity_data/validation_v1/MD5SUMS manifest +
   `analyze_validation.py --verify-md5` (shard integrity check
   documented in REPRODUCING.md sec. 4).
5. CLAIMS DISCIPLINE: no "superhuman"/"best-human" claims - measured
   claims only (vs anchor fields, vs the frozen reference, N and CIs).
   The user calibration matches are n=1 anecdotes and labeled so.
6. DEMO PATH: hearts_web/ ships as the way outsiders can play the
   final net; document localhost setup; optional hosted demo later.
7. COMMUNITY SURFACE: issues policy, a CONTRIBUTING note (or an
   explicit "archive, not maintained" statement), citation entry.
8. FINAL: user reviews this checklist executed, then flips the repo.

## 5. Habits starting now (cheap, avoid end-loading)
- Never commit secrets/tokens/keys; .mcp.json is gitignored.
- Keep every prereg, verdict JSON, and results doc in-repo (ongoing).
- When a session-only insight matters to the trajectory, append it to
  the ledger - chat/agent memory is NOT part of the releasable record.
- Tag era-defining commits (validation, promotions) for navigation.
- Back up Hall_of_Fame/ + data dirs (sec. 2) on a schedule.
