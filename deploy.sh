#!/usr/bin/env bash
# Deploy shop_assistant to the GPU server (ssh alias in DEPLOY_HOST) as a user-level systemd service (ticket #15).
# Also installs + enables the nightly ingestion timer (ticket #18): deploy/shop-assistant-ingest.{service,timer}.
# Usage: ./deploy.sh [--data]     --data also syncs data/ (products, embeddings, faq, log)
set -euo pipefail

HOST=${DEPLOY_HOST:-gpu-host}   # ssh alias of the GPU server
DEST='~/shop_assistant'
UNIT=shop-assistant.service
INGEST_SERVICE=shop-assistant-ingest.service   # nightly ingestion (ticket #18), files in deploy/
INGEST_TIMER=shop-assistant-ingest.timer
SYNC_DATA=0

for arg in "$@"; do
  case "$arg" in
    --data) SYNC_DATA=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"

# The bot must never run twice (Telegram delivers each update to one session only).
if pgrep -f shop_assistant.main >/dev/null 2>&1; then
  echo "refusing to deploy: shop_assistant.main is running locally — stop it first" >&2
  exit 1
fi

EXCLUDES=(--exclude '__pycache__' --exclude '.git' --exclude '.env' --exclude '.venv' --exclude 'session/')
if [[ $SYNC_DATA -eq 0 ]]; then
  EXCLUDES+=(--exclude 'data/')
else
  # Server-owned files: the live customer log, run logs, backups, eval runs. Excluded paths are also
  # protected from --delete, so a data sync never wipes them.
  EXCLUDES+=(--exclude 'data/log.jsonl' --exclude 'data/*.log' --exclude 'data/*.bak*' --exclude 'data/eval_*')
  # The nightly run's Gemini request log (#18) is written on the server only.
  EXCLUDES+=(--exclude 'data/gemini_ingest.jsonl')
fi

echo "==> rsync to $HOST:$DEST"
rsync -az --delete "${EXCLUDES[@]}" ./ "$HOST:$DEST/"

echo "==> install deps + (re)start service on $HOST"
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"   # uv is not on the non-interactive ssh PATH
cd $DEST
uv venv -q --allow-existing
uv pip install -q -r requirements.txt
mkdir -p ~/.config/systemd/user
cp $UNIT ~/.config/systemd/user/$UNIT
cp deploy/$INGEST_SERVICE deploy/$INGEST_TIMER ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now $UNIT
systemctl --user restart $UNIT
# Nightly ingestion: only the timer is enabled; it starts the oneshot service at night.
systemctl --user enable --now $INGEST_TIMER
sleep 5
systemctl --user is-active $UNIT
journalctl --user -u $UNIT -n 5 --no-pager
systemctl --user list-timers $INGEST_TIMER --no-pager
REMOTE

echo "==> done"
