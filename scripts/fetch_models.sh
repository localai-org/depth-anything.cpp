#!/usr/bin/env bash
# Pre-download the curated DA3 model set plus the DA2 depth-upscale default (~13 GB).
# One weight per capability, not the full 31 GB of redundant quants.
# Idempotent: skips files already present at the right size; resumes partials.
set -uo pipefail

REPO="mudler/depth-anything.cpp-gguf"
BASE="https://huggingface.co/${REPO}/resolve/main"
DIR="$(cd "$(dirname "$0")/../models" && pwd)"

# capability                       file
CURATED=(
  "relative+pose (fast)           depth-anything-small-f32.gguf"
  "relative+pose (default)        depth-anything-base-q4_k.gguf"
  "relative+pose (quality)        depth-anything-large-f32.gguf"
  "relative+pose+gaussians        depth-anything-giant-f32.gguf"
  "metric+pose (anyview branch)   depth-anything-nested-anyview.gguf"
  "metric+pose (metric branch)    depth-anything-nested-metric.gguf"
  "relative depth (upscale default) depth-anything2-base-f32.gguf"
)

echo "Downloading curated DA3 models into $DIR"
for row in "${CURATED[@]}"; do
  cap="${row% *}"; file="${row##* }"
  out="$DIR/$file"
  if [ -s "$out" ]; then
    echo "  [skip] $file (present, $(du -h "$out" | cut -f1))"
    continue
  fi
  echo "  [get ] $file  — $cap"
  curl -fL -C - --retry 5 --retry-delay 3 -o "$out" "$BASE/$file" \
    || { echo "  [FAIL] $file"; rm -f "$out"; }
done
echo "Done. Models present:"
ls -lh "$DIR"/*.gguf 2>/dev/null
