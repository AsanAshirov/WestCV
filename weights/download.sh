#!/usr/bin/env bash
# Fetch the model weights if they are missing, then verify SHA-256. Idempotent.
# The weights are also committed to the repository, so this is a fallback.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.4.0
for f in yolo26m.pt yolo26n.pt; do
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$BASE/$f"
done
sha256sum -c SHA256SUMS
