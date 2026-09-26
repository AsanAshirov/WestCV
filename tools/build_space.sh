#!/usr/bin/env bash
# Assemble the Hugging Face Space (Gradio SDK) in dist/space and upload it:
#   bash tools/build_space.sh
#   pip install -U huggingface_hub && hf auth login
#   hf upload <user>/antigradient-demo dist/space . --repo-type space
# `hf upload` stores the .pt weights correctly (a plain git push of binaries is rejected).
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
OUT="$ROOT/dist/space"
rm -rf "$OUT"
mkdir -p "$OUT/weights"
cp "$ROOT/app/app.py" "$ROOT/app/requirements.txt" "$ROOT/app/README.md" "$ROOT/LICENSE" "$OUT/"
cp -r "$ROOT/src" "$ROOT/configs" "$OUT/"
cp "$ROOT/weights/yolo26n.pt" "$OUT/weights/"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
echo "Space ready in $OUT"
