#!/usr/bin/env bash
# Deploy the live site (play.perilune.ai) to the VPS.
#
# The VPS serves from a plain clone of the PUBLIC GitHub repo, so a
# deploy is only ever "fast-forward the clone" - it can never carry an
# uncommitted local file. Everything the site needs that is NOT in git
# (site_config.py, .share_secret, the .pth weights, static/models +
# static/ort, the jsonl data files, the VPS-built hearts_env .so) is
# untracked there and a pull leaves it alone. That is the whole safety
# argument for this script; keep it true.
#
# Restart policy: index.html and the other static pages are read from
# disk per request (server.py's _og_serve + the /static mount), so a
# front-end change is live the moment the pull lands. Only a change to
# tracked .py files needs the service bounced - this decides that from
# the actual diff rather than asking you to remember.
#
#   ops/deploy_site.sh              # pull, restart if .py changed, verify
#   ops/deploy_site.sh --restart    # force the bounce (e.g. new weights)
#   ops/deploy_site.sh --dry-run    # show what would deploy, change nothing
#
# The TARGET HOST is infrastructure and never lives in this public repo
# (same rule as RUN_TUNNEL.cmd: gitignored here, versioned in the
# private ops repo). Supply it in ops/deploy_target.env - gitignored,
# and in the ops-repo sync set:
#
#   HEARTS_VPS=root@<vps-ip>          # required
#   HEARTS_VPS_REPO=/home/hearts/hearts_simulator
#   HEARTS_VPS_USER=hearts
#   HEARTS_SITE_URL=https://play.perilune.ai/
#
# An environment variable of the same name overrides the file.
set -euo pipefail

CONF="$(dirname "$0")/deploy_target.env"
# shellcheck source=/dev/null
[ -f "$CONF" ] && . "$CONF"

HOST="${HEARTS_VPS:-}"
if [ -z "$HOST" ]; then
  cat >&2 <<'MSG'
No deploy target. The VPS address is private - it is not in this repo.
Create ops/deploy_target.env (gitignored) with:

  HEARTS_VPS=root@<vps-ip>

The current value lives in the private ops repo; see hearts_web/DEPLOY.md.
MSG
  exit 1
fi
APP_USER="${HEARTS_VPS_USER:-hearts}"
REPO="${HEARTS_VPS_REPO:-/home/hearts/hearts_simulator}"
SITE="${HEARTS_SITE_URL:-https://play.perilune.ai/}"
SERVICE=hearts-web

FORCE_RESTART=0
DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --restart) FORCE_RESTART=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---- preflight: the VPS can only pull what GitHub already has -------------
say "preflight (local)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ]; then
  echo "on branch '$BRANCH', not main - the VPS tracks main. Aborting." >&2
  exit 1
fi
git fetch --quiet origin main
UNPUSHED="$(git log --oneline origin/main..HEAD | wc -l | tr -d ' ')"
if [ "$UNPUSHED" != "0" ]; then
  echo "$UNPUSHED local commit(s) not on origin/main - push first:" >&2
  git log --oneline origin/main..HEAD >&2
  exit 1
fi
LOCAL="$(git rev-parse origin/main)"
echo "origin/main = ${LOCAL:0:9}  $(git log -1 --format=%s "$LOCAL")"

# ---- the deploy itself ----------------------------------------------------
say "deploying to $HOST"
# shellcheck disable=SC2087  # deliberate local expansion into the heredoc
ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd "$REPO"

# The clone is owned by $APP_USER; we arrive as root (systemctl needs
# it). Drop to the app user for every git call so the working tree
# keeps one owner - a root-written object here bites the NEXT deploy.
asapp() {
  if [ "\$(id -un)" = "$APP_USER" ]; then "\$@"; else sudo -H -u "$APP_USER" "\$@"; fi
}

# A dirty TRACKED file on the VPS means someone hot-edited the box; the
# pull would conflict and we would rather stop than guess. Untracked
# files are the normal state (models, config, data) - never touched.
DIRTY="\$(asapp git status --porcelain --untracked-files=no)"
if [ -n "\$DIRTY" ]; then
  echo "VPS repo has modified tracked files - refusing to pull:" >&2
  echo "\$DIRTY" >&2
  exit 1
fi

BEFORE="\$(asapp git rev-parse HEAD)"
if [ "\$BEFORE" = "$LOCAL" ]; then
  echo "already at \${BEFORE:0:9} - nothing to pull"
else
  if [ "$DRY_RUN" = "1" ]; then
    asapp git fetch --quiet origin main
    echo "[dry-run] would fast-forward \${BEFORE:0:9} -> ${LOCAL:0:9}:"
    asapp git diff --stat "\$BEFORE" "$LOCAL"
    exit 0
  fi
  asapp git pull --ff-only --quiet origin main
fi
AFTER="\$(asapp git rev-parse HEAD)"
[ "\$AFTER" = "$LOCAL" ] || { echo "VPS HEAD \$AFTER != origin/main $LOCAL" >&2; exit 1; }

if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] no restart evaluated"; exit 0; fi

# Restart only when server-side code moved (see the header note).
NEED=0
if [ "$FORCE_RESTART" = "1" ]; then
  NEED=1; echo "restart: forced"
elif [ "\$BEFORE" != "\$AFTER" ] \
     && asapp git diff --name-only "\$BEFORE" "\$AFTER" \
        | grep -qE '\.py\$'; then
  NEED=1; echo "restart: python changed"
  asapp git diff --name-only "\$BEFORE" "\$AFTER" | grep -E '\.py\$'
else
  echo "restart: not needed (static/front-end only)"
fi

if [ "\$NEED" = "1" ]; then
  systemctl restart $SERVICE
  sleep 3
fi
systemctl is-active --quiet $SERVICE || { systemctl status $SERVICE --no-pager -l; exit 1; }

# Origin health, before the edge gets a chance to mask a broken box.
CODE="\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8642/api/leaderboard)"
echo "origin /api/leaderboard: \$CODE"
[ "\$CODE" = "200" ] || exit 1
EOF

if [ "$DRY_RUN" = "1" ]; then say "dry run complete"; exit 0; fi

# ---- verify through the tunnel -------------------------------------------
say "verifying $SITE"
CODE="$(curl -s -o /dev/null -w '%{http_code}' "$SITE")"
TIME="$(curl -s -o /dev/null -w '%{time_total}' "$SITE")"
echo "public: HTTP $CODE in ${TIME}s"
[ "$CODE" = "200" ] || exit 1
echo "deployed ${LOCAL:0:9}"
