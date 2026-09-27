#!/usr/bin/env bash
# download.sh — fetch the model weights used by solution.py (run once, before going offline).
#   YOLO26m (Ultralytics, AGPL-3.0, COCO-pretrained), 44 MB
set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
SHA256="401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
if [ ! -f yolo26m.pt ]; then
  curl -L --fail --retry 3 -o yolo26m.pt.part "$URL"
  mv yolo26m.pt.part yolo26m.pt
fi
echo "$SHA256  yolo26m.pt" | sha256sum -c -
ls -l yolo26m.pt
