# Perilune webapp - deferred features (so they aren't forgotten)

Review mode extensions (2026-08-04 design session):
- BELIEF HEATMAP in review: what the net BELIEVED about hidden cards at
  each step vs reality (belief head per opponent) - unique feature;
  heavy payload + UI, deferred.
- MATCH HISTORY: list a pid's past matches (from telemetry) so any old
  game can be re-reviewed, not just the one just finished.
- SHAREABLE REVIEW LINKS: encode match + position; pairs with history.
- EXPORT match JSON from the review page.

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
