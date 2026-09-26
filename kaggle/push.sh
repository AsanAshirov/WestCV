#!/usr/bin/env bash
# Run the pipeline on the sample videos on a PRIVATE Kaggle GPU kernel.
#
#   KAGGLE_USER=<your kaggle username> bash kaggle/push.sh             # videos from Google Drive
#   KAGGLE_USER=<user> SAMPLES_DIR=DataSets bash kaggle/push.sh        # also upload local videos (slow, ~20 GB)
#
# Needs: pip install kaggle; token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN).
# Everything is created private: datasets without --public, the kernel with is_private=true.
# Only committed files are uploaded (git archive), never DataSets/ or caches.
set -euo pipefail
: "${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
ROOT=$(git rev-parse --show-toplevel)
BUILD=$(mktemp -d)
KERNEL=antigradient-samples-t4
SOURCES="\"$KAGGLE_USER/antigradient-code\""

upload() {  # upload <folder> <slug>: create the private dataset or add a new version
  if kaggle datasets status "$KAGGLE_USER/$2" >/dev/null 2>&1; then
    kaggle datasets version -p "$1" -m "$(git -C "$ROOT" rev-parse --short HEAD)" -r zip
  else
    kaggle datasets create -p "$1" -r zip
  fi
}

mkdir -p "$BUILD/code"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$BUILD/code"
git -C "$ROOT" rev-parse HEAD > "$BUILD/code/COMMIT"
cat > "$BUILD/code/dataset-metadata.json" <<EOF
{"title": "antigradient-code", "id": "$KAGGLE_USER/antigradient-code", "licenses": [{"name": "other"}]}
EOF
upload "$BUILD/code" antigradient-code

if [ -n "${SAMPLES_DIR:-}" ]; then  # upload in place: no 20 GB copy on a small disk
  cat > "$SAMPLES_DIR/dataset-metadata.json" <<EOF
{"title": "antigradient-samples", "id": "$KAGGLE_USER/antigradient-samples", "licenses": [{"name": "other"}]}
EOF
  upload "$SAMPLES_DIR" antigradient-samples
  SOURCES="$SOURCES, \"$KAGGLE_USER/antigradient-samples\""
fi

mkdir -p "$BUILD/kernel"
cp "$ROOT/kaggle/run_samples.py" "$BUILD/kernel/"
cat > "$BUILD/kernel/kernel-metadata.json" <<EOF
{
  "id": "$KAGGLE_USER/$KERNEL",
  "title": "$KERNEL",
  "code_file": "run_samples.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [$SOURCES],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF
sleep 20  # let Kaggle finish processing the new dataset version
kaggle kernels push -p "$BUILD/kernel"
rm -rf "$BUILD"
echo
echo "status:  kaggle kernels status $KAGGLE_USER/$KERNEL"
echo "results: kaggle kernels output $KAGGLE_USER/$KERNEL -p kaggle_out"
