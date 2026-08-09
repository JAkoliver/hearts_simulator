# Perilune site security design (hearts_web)

Distilled from the 2026-08-08 security-audit session; each claim traces
to a commit whose body records the threat, the fix, and how it was
verified (chain `c703baf..f1a90e0`, cited per section below).

WRITING RULE for this document: it must stay publishable. It describes
invariants that hold even with all code public (Kerckhoffs's principle:
the client is served to every visitor anyway, and the repo will be
open-sourced). Anything whose secrecy matters - tuned thresholds,
credentials, player data, detection methods, open weaknesses - lives
OUTSIDE the repo (sec. "Release boundary") and must never be added
here. If a note would help an attacker only while unfixed, it belongs
in the private ops layer, not in this file.

## 1. Identity and credentials

- One credential exists: the server-minted 128-bit bearer key. It lives
  in the player's browser (localStorage) and optional key file - the
  server stores only sha256(key). A leaked server file impersonates
  nobody. (`6b94171`)
- The canonical id IS sha256(key). Canonical ids are identifiers, never
  credentials: presenting one authenticates nothing (403); only the
  hash preimage does. `resolve_pid()` is the single auth chokepoint
  every key intake threads through. (`6b94171`, `300bbef`)
- Display names are server-assigned immutable codenames - unique by
  construction, no user-chosen names, hence no impersonation or
  moderation surface. The codename doubles as the public handle
  (slugified for profile URLs); no new identifier and no credential
  exists anywhere in the public path. (`c703baf`, `8b0624c`)
- Identities are created ONLY via `POST /api/identity/new`
  (rate-limited); every lookup endpoint is strictly non-minting, so
  visits can never orphan a returning player's key. (`01ca084`,
  `dacb6a0`)
- Credentials never appear in URLs. Share links and public profiles use
  minted share tokens and slugs; the progress page hosts share UI and
  therefore carries no key UI (that moved to /account). (`01ca084`,
  `60b7208`, `8b0624c`)
- Key reveal is two-step (explicit warning, second click shows);
  compromised keys rotate: the replacement resolves to the SAME
  identity, the old key is revoked and fails loudly at every endpoint.
  Rotation events store credential hashes only. (`94afd01`, `300bbef`,
  `6b94171`)

## 2. Sessions and tables

- A solo session id (sid) is an identifier, NOT a credential: pid-bound
  sessions require the owning key on state reads and plays (403
  otherwise). This closed the audit's real finding - review URLs
  carried the sid, so a stream viewer could have read a live hand or
  injected moves. (`dacb6a0`)
- Table views are strictly per-seat: the state a player receives
  contains their own hand and public information only.
- The AFK auto-play is a deliberately dumb heuristic (lowest legal
  card), so timing out is never a way to have the strong AI play a
  seat. (server.py `check_timeout`)

## 3. Game integrity (what makes the leaderboard trustworthy)

- The game is server-authoritative end to end: server-drawn secret
  seed, server-validated moves, per-seat state. Scores cannot be
  fabricated client-side. (`f1a90e0` body records the audit)
- Seed-leak sealed: one seed drives all deals of a match and the review
  payload contains it, so reviews, insights, and share-minting gate on
  a LOGGED match summary - nothing with the seed is servable
  mid-match. (`f1a90e0`)
- The site serves promoted weights only: a dedicated
  `hearts_web_model.pth` advanced exclusively by the promotion path -
  never the training chain's working file, which can hold unpromoted
  candidates mid-trial. Leaderboards are per model era, keyed by the
  logged model md5, eligibility server-verified from the match log,
  sole first place only; every listed score links to its match via a
  share token, so every score is verifiable by inspection. (`584d794`)
- Cheating by consulting the AI is inherently possible once weights are
  open; statistical detection is therefore a PRIVATE ops-layer concern
  by design - its methods and thresholds are deliberately not part of
  the public instrument. (hearts_web/TODO.md, site_config boundary)

## 4. Abuse limits

- Per-IP rate buckets on tunneled traffic: a general bucket plus a
  creation bucket that covers session/table creation AND identity
  mint/rotate (unbounded minting would burn the codename pool).
  Public-profile lazy compute is capped per request and cached.
  Registries (sessions, tables) are capped with oldest-first eviction;
  stale tables are reaped. (`dacb6a0`, site_config_example.py)
- The concrete limit VALUES are operations, not instrument: they live
  in the gitignored `site_config.py`. The committed
  `site_config_example.py` ships safe generic defaults. (release-
  boundary decision 2026-08-04)

## 5. Release boundary - public instrument, private operations

Public (committed): server + client code, this document,
`site_config_example.py` with safe defaults.

Private (gitignored, never committed):
- `hearts_web/site_config.py` - tuned production thresholds/policies
- `hearts_web/.share_secret` - share-token signing secret
- `hearts_web/player_keys.jsonl`, `player_names.jsonl` - identity
  stores (hashes only, but still operational data)
- `hearts_web/match_logs.jsonl` - players' personal match data
- Tunnel infrastructure and machine-specific launchers
  (`RUN_TUNNEL.cmd`, `TUNNEL_HELPER.ps1`, schedule scripts), `logs/`

The published client code is in every visitor's browser regardless of
repo status; the server code's safety must not depend on secrecy, and
after the 2026-08-08 session it does not. What remains secret is
exactly the set of things that are secrets: credentials, signing
material, player data, and operational tuning.

## 6. Maintenance rules

- New security-relevant changes: record threat + fix + verification in
  the commit body (the chain above is the pattern), then update this
  document's invariants if they changed.
- Never note an OPEN weakness here or in TODO.md beyond what is already
  safe-public; open items with exploit value go to the private ops
  layer until fixed.
- Pre-release: sweep this doc, TODO.md, and RELEASE_PLAN.md for
  anything that migrated from "invariant" back to "open hole" (see
  RELEASE_PLAN sec. 4 checklist).
