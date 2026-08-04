"""Site operations config - EXAMPLE (committed; safe generic defaults).

The live deployment copies this to site_config.py (gitignored) and tunes
it there: production thresholds, caps, and policies are deliberately
NOT published (release-boundary decision 2026-08-04 - publish the
instrument, keep the operations).
"""

# Checkpoint the AI seats play with. The live site may run weights that
# are newer than any released checkpoint.
MODEL_PATH = 'hearts_model_final.pth'

# Per-IP rate limits, applied only to tunneled traffic (CF-Connecting-IP
# present): (max requests, window seconds).
RL_GENERAL = (60, 10.0)
RL_CREATE = (10, 60.0)      # session/table creation + join

# In-memory registries.
SESSION_CAP = 200           # solo sessions kept before oldest eviction
TABLE_CAP = 100             # tables kept before oldest eviction
CODE_LEN = 4                # join-code length

# Table lifecycle (seconds).
REAPER_INTERVAL_S = 60      # sweep cadence
STALE_S = 120               # all humans silent this long -> table deleted
AWAY_S = 10                 # heartbeat silence before a seat shows 'away'

# Dev controls (reset button, ?player= identity override):
# 'localhost' = injected only for direct localhost requests; 'off' = never.
DEV_MODE = 'localhost'
