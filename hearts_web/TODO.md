# Perilune webapp - deferred features (so they aren't forgotten)

Review mode extensions (2026-08-04 design session):
- BELIEF HEATMAP in review: what the net BELIEVED about hidden cards at
  each step vs reality (belief head per opponent) - unique feature;
  heavy payload + UI, deferred.
- SHAREABLE REVIEW LINKS: encode match + position; pairs with history
  (DONE 2026-08-05: match history menu list + /api/history).
- EXPORT match JSON from the review page.
- SEARCH-VERIFIED DISAGREEMENTS: re-judge top disagreements offline
  with the K=64 search player ("verified mistakes" - the raw-policy
  gap metric conflates player error with net overconfidence).
- CLIENT-SIDE SEARCH (the hosting-ready answer, post-release): compile
  the C++ engine + determinized search to WASM via Emscripten; run the
  net in-browser with ONNX Runtime Web (WebGPU, WASM/CPU fallback).
  "Deep check this position" then costs the CLICKER's hardware, not the
  server's GPU - required before the site moves to a hosting provider
  (chess.com/lichess model: WASM Stockfish client-side, metered server
  analysis). Caveat: ships the weights to every visitor, so only nets
  that are (or may as well be) public; a stronger private site net must
  stay server-side. UI: search results are a distinct layer - own color
  and icon, labeled with the budget (K), VALUE units (points / win%)
  never probabilities; disagreement entries upgrade in place to
  "verified (-x pts)" or "retracted - your move was fine".
  Interim while self-hosted: server-side probe with per-user daily
  quota + idle-GPU-only batch (needs a search_probe C++ CLI: seed +
  action prefix -> per-move values JSON, reusing the replay contract).
- Mobile pass for the review page (game page done; review is
  desktop-first).
- Log index for review/history (DONE 2026-08-05: in-memory offset
  index built at startup, updated on append; history precomputed
  per pid).

Gameplay / table:
- Big-batch animation fast-forward on tab refocus (background-tab
  throttling delivers a burst; snap to state instead of replaying).
- Canned emotes for tables; lobby host controls (kick, seat order);
  host turn-timer option (AI plays for AFK seat); spectator mode.
- Difficulty tiers (v3/v4 anchor nets as easier opponents).
- Moon-defense challenge mode (scenarios seeded from logged exploit
  games; every attempt is exploiter-league data).
- Skill rating + leaderboard (needs anti-cheat design first - open
  weights make 'consult the AI' cheating trivial; server-side
  statistical detectors belong in the PRIVATE ops layer, see
  site_config boundary).

Polish:
- PWA manifest + installable icon; sounds toggle; animation-speed
  setting; keyboard play in the game itself; colorblind 4-color deck;
  prefers-reduced-motion; collapsible side panels; green-mode felt
  polish pass.

Ops:
- Localhost-only admin/status page (active tables, matches/day, unique
  pids, log size).
- Session-restart table persistence (optional; light pickle).
