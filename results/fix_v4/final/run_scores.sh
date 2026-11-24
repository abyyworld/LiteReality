#!/bin/zsh
# Score v2 and v4 on both eval sets at 480 and full res, room-call Spark settings, Metal.
cd ~/LiteReality
for full in "" "--full"; do
  for set in sharp random; do
    for v in v2 v4; do
      echo "=== $v $set ${full:-480} $(date +%T)"
      uv run --frozen --with playwright --with lpips==0.1.4 python results/stray_quality/score.py \
        eval $v run/atrium_stray_q/$v --eval-set $set $full --gl metal \
        --results results/fix_v4/final/scores 2>&1 | grep -v '^  row'
    done
  done
done
echo "=== ALL DONE $(date +%T)"
