#!/usr/bin/env bash
# M9 fresh-clone audit (plan §8): clone the pushed repo into a temp dir,
# sync the environment, run the quality gates, and produce an explanation
# from a committed checkpoint without any retraining.
set -euo pipefail

AUDIT_DIR=$(mktemp -d)
echo "=== cloning into $AUDIT_DIR ==="
git clone --quiet https://github.com/vivri161803/graphlime-rdf.git "$AUDIT_DIR/graphlime-rdf"
cd "$AUDIT_DIR/graphlime-rdf"

echo "=== uv sync ==="
uv sync --quiet

echo "=== just check ==="
just check

echo "=== load-and-explain from committed checkpoint (no retraining) ==="
uv run python -m graphlime_rdf.cli load-and-explain \
    --checkpoint checkpoints/aifb_best.pt --node 5766 --top 5

echo "=== AUDIT PASSED ==="
