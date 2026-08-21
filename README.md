# Perilune: a match-aware Hearts AI

A neural network that taught itself Hearts as a **match to 100**, not a
series of isolated deals, plus everything around it: the C++ engine and
self-play generator, belief-weighted determinized search, the training
and gate machinery, the measurement record, and the public web app
where you can play it ([play.perilune.ai](https://play.perilune.ai)).

The repository is released with its full history on purpose. The
trajectory, including the experiments that failed and were closed with
evidence, is most of what there is to learn here.

## Headline results (measured claims only)

Every claim below carries its sample size and is traceable to the
append-only ledger ([docs/speed_ledger.md](docs/speed_ledger.md)), a
pre-registration, or a verdict JSON
([equity_data/verdicts/](equity_data/verdicts/)). The claim table with
citations is [docs/release/RESULTS.md](docs/release/RESULTS.md).

- **Match-aware search beats match-blind search: 48.91% vs 44.47%
  match wins** (+4.44 win-points, SE 0.68, McNemar one-sided
  p about 5e-11) over N=8,000 paired matches against a frozen
  reference, all 8 shards positive. The match-aware player takes more
  firsts and more fourths and has *worse* mean placement: it trades
  expected placement for win probability, which is the objective.
- Making the network score-aware compounded immediately at adoption:
  cumulative win rate against a fixed anchor field went from 26.8% to
  39.8% across three promotions (n=800 paired matches per gate).
- A capacity verdict, run to completion and taken seriously: several
  self-improvement recipes (expert iteration, PPO variants, visit-count
  distillation) passed their intermediate stages and still failed the
  promotion gates, with a clean dose-response in the final case. The
  current 7.6M-parameter network is at capacity from this baseline;
  the successor campaign (v6, [docs/v6_prereg.md](docs/v6_prereg.md))
  trained a larger, structurally extended network from scratch and was
  concluded 2026-08-16 without a promotion — post-mortem in
  [docs/v6_postmortem.md](docs/v6_postmortem.md). The capacity answer
  arrived by a different route: the current champion (promoted
  2026-08-21) is a **gated ensemble of raw nets** — the 7.6M champion
  playing ~90% of decisions, with the 19M obs-v2 network from the v6
  campaign playing only the moon-threat states its own threat head
  detects. That composition cut moons conceded to a competent attacker
  by 29% at no measurable cost in ordinary play — capacity paying off
  exactly where the whole-net evaluation could not see it
  ([docs/gated_ensemble_program.md](docs/gated_ensemble_program.md),
  [docs/exploiter_league_r7_results.md](docs/exploiter_league_r7_results.md)).

No claims about human-relative strength are made anywhere in this
repository; human games on the site are n=1 anecdotes and labeled so.

## Play it

- **Live:** [play.perilune.ai](https://play.perilune.ai). Free, no
  account: solo vs the AI, daily challenge, multiplayer tables,
  spectating, and full match reviews with on-device deep search.
- **Local web app:** `python -m uvicorn hearts_web.server:app --port
  8642` from the repo root (models listed below must be present), then
  open `http://localhost:8642`.
- **Terminal game:** the `hearts_game` target of the C++ build.

## Build

Toolchain pins and full procedures:
[docs/release/REPRODUCING.md](docs/release/REPRODUCING.md).

**Windows (development platform):**
1. Install Visual Studio (MSVC, C++17) and CMake 3.14+.
2. Download **libtorch 2.12.1+cu126** and unzip to `./libtorch`.
3. Fetch the CUDA runtime headers (NVIDIA-licensed, not tracked here):
   `python scripts/fetch_cuda_headers.py`
4. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
5. `cmake --build build --config Release`

**Linux / cloud:** `docker build -f cloud/Dockerfile -t hearts-worker .`
builds the headless generation/eval targets (the Dockerfile pins the
same libtorch and the matching CUDA wheels).
`-DHEARTS_CLOUD_ONLY=ON` does the same outside Docker.

**Python side:** Python 3.13, `pip install -r requirements.lock`
(torch 2.12.1+cu126 to match libtorch). Training, evaluation, gates,
and the web app all run from these pins.

## Models

The released checkpoints and traces are in **GitHub Releases**
(`models-v1`), each with md5/sha256 manifests. The important ones:

| File | Role |
|---|---|
| `hearts_ai_search_match.pt` | Deployed match-aware search trace: the strongest configuration, and the v6 teacher (md5 3a2abd36) |
| `hearts_ai_match_8a89da90.pt` + `hearts_model_final.pth` | The 8a89da90 champion network (5th match-era promotion), match-context trace + checkpoint — now the DEFAULT component and search substrate of the promoted ensemble |
| `hearts_ensemble_8d7816d1.pth` + `_882trace.pt` (models-v1 release) | Current champion: gated ensemble (8a89da90 default + v6 arm a specialist + moon-head router, one 882-input module; checkpoint md5 8d7816d1, trace 9d9a4f49) — [round-7 record](docs/exploiter_league_r7_results.md), [model card](docs/release/model_cards/hearts_ensemble_8d7816d1.md) |
| `hearts_ai_grandmaster.pt` / `hearts_ai_search.pt` | Champion raw-play / search traces (filenames are engine identifiers, not strength claims) |
| `hearts_ai_search_ref_matchblind_20260724.pt` + `.pth` | The frozen match-blind reference of the N=8000 validation (md5 a1a0be31); required to reproduce the headline result |
| `hearts_equity.pt` + `equity_v1.pth` | The equity model: score state to placement probabilities |
| `hearts_ai_grandmaster_v3_milestone7.pt` | The v3-m7 anchor (goes in `legacy_v3_pass238/`, the path the eval scripts bind) |
| `hearts_ai_grandmaster_v4m10.pt` / `hearts_ai_search_v4m10.pt` | The v4-m10 anchor: with v3-m7, the fixed opponent field of every measurement |
| `shooter_agg_v1b.pt` / `shooter_sel_v1.pt` | Certified moon-shooter clones: the exploiter league's frozen attack instruments |

Model cards: [docs/release/model_cards/](docs/release/model_cards/).
Place downloaded models in the repo root; the engine and web app load
them by these exact filenames.

## Documentation

Reading order for an outside observer
([docs/release/INDEX.md](docs/release/INDEX.md)):

1. This README.
2. [JOURNEY.md](docs/release/JOURNEY.md): the narrative in eras, with
   the failures kept in.
3. [ARCHITECTURE.md](docs/release/ARCHITECTURE.md): engine, network,
   search, training, gates, data formats, cloud, web app.
4. [METHODOLOGY.md](docs/release/METHODOLOGY.md): the measurement
   discipline (paired deals, CRN, neutral anchors, pre-registration,
   halt-is-default).
5. [RESULTS.md](docs/release/RESULTS.md): every major claim with N,
   CI, and citation.
6. [REPRODUCING.md](docs/release/REPRODUCING.md): builds, gates,
   regenerating data from seeds.

Primary sources outrank all of the above on conflict: the ledger, the
rules file ([docs/experiment_rules.md](docs/experiment_rules.md)), the
pre-registrations, the verdict JSONs, and the git history.

## The workflow, credited

The project is AI-assisted end to end, and the toolchain evolution is
part of the record: the first week ran in the Antigravity IDE with
Gemini 3.1 Pro as coding assistant; from 2026-07-09 onward, Claude Code
with the Fable 5 model has done ops and implementation under human
direction. The measurement discipline in
[docs/experiment_rules.md](docs/experiment_rules.md) exists in large
part so that an AI-assisted project cannot fool its operator.

## Data recorded by the site

Matches on the live site are recorded anonymously (cards, timing,
outcomes, a random browser identifier; no accounts, no personal data)
and used to improve the AI. Details on the site's
[about page](https://play.perilune.ai/about). Players' raw logs are
not part of this repository.

## License

MIT for code, released weights, and released data (see
[LICENSE](LICENSE)). Fonts are OFL, emote artwork is Twemoji (CC-BY
4.0), and the CUDA runtime headers are fetched separately under
NVIDIA's license.
