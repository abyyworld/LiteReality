#!/bin/bash
# Local-only scoring of v2 and the partial A1 snapshot (no Modal). Subsets: SwiftShader takes
# ~60-110 s a view, so 10 views at 480 and 4 at full resolution per set, same views for both.
cd ~/LiteReality
S="env PYTHONUNBUFFERED=1 uv run --frozen --with playwright --with lpips==0.1.4 python results/stray_quality/score.py eval"
O=results/fix_v4/scores
run() { echo "=== $(date -u +%T) $*"; $S "$@" --results $O 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"; }
for set in sharp random; do
  run v2 run/atrium_stray_q/v2 --eval-set $set --views 10
  run a1p1250 run/atrium_stray_q/v4_a1_nogrid --eval-set $set --views 10
done
for set in sharp random; do
  run v2 run/atrium_stray_q/v2 --eval-set $set --full --views 4
  run a1p1250 run/atrium_stray_q/v4_a1_nogrid --eval-set $set --full --views 4
done
echo "=== $(date -u +%T) ALL DONE"
