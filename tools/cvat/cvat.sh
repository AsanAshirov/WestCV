#!/usr/bin/env bash
# cvat.sh — the WestCV CVAT instance (separate Docker project "westcv-cvat").
# usage: tools/cvat/cvat.sh up|down|ps|logs [service]|admin|tunnel|untunnel|url
#   up     pull images (first run ~2.5 GB) and start; UI at http://localhost:${WESTCV_CVAT_PORT:-8080}
#   down   stop containers, keep data (volumes westcv-cvat_*)
#   admin  create the superuser interactively
#   tunnel start a Cloudflare quick tunnel and print the public https URL (changes on every restart)
#   url    print the current public URL; untunnel stops the tunnel
# env: CVAT_DIR (upstream checkout, default ~/cvat, tag v2.76.0), WESTCV_CVAT_PORT (default 8080)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CVAT_DIR="${CVAT_DIR:-$HOME/cvat}"
share="$REPO/DataSets/proxies"
command -v cygpath >/dev/null && share="$(cygpath -m "$share")"   # Git Bash -> C:/... for Docker Desktop
export WESTCV_SHARE="$share"

if [ ! -f "$CVAT_DIR/docker-compose.yml" ]; then
  echo "no CVAT checkout in $CVAT_DIR; run: git clone --depth 1 --branch v2.76.0 https://github.com/cvat-ai/cvat.git \"$CVAT_DIR\"" >&2
  exit 1
fi

dc() {
  docker compose -p westcv-cvat --project-directory "$CVAT_DIR" \
    -f "$CVAT_DIR/docker-compose.yml" -f "$REPO/tools/cvat/docker-compose.westcv.yml" "$@"
}

case "${1:-}" in
  up)    dc up -d && echo "CVAT: http://localhost:${WESTCV_CVAT_PORT:-8080}" ;;
  down)  dc down ;;
  ps)    dc ps ;;
  logs)  shift; dc logs --tail 100 "$@" ;;
  admin) docker exec -it westcv_cvat_server bash -ic 'python3 ~/manage.py createsuperuser' ;;
  tunnel) dc --profile tunnel up -d cloudflared && sleep 8 && "$0" url ;;
  untunnel) dc --profile tunnel stop cloudflared ;;
  url)   docker logs westcv_cvat_tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1 ;;
  *)     sed -n '2,11p' "$0"; exit 1 ;;
esac
