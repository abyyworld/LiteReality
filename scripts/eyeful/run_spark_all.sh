#!/usr/bin/env bash
# Every Spark score in results/eyeful/spark/ for STEP 3, all 71 test views, in order.
cd "$(dirname "$0")/../.." || exit 1
R=scripts/eyeful/run_spark.sh
for run in A_30000 B_5000 B_1250; do
  $R trainer "$run" || echo "FAILED $run trainer"
done
for run in A_30000 B_5000 B_1250; do
  presets="call std3 blur limits matched"
  [[ $run == B_* ]] && presets="call limits matched"   # B: "blur" = std3 (already antialiased)
  for p in $presets; do
    $R spark "$run" --preset "$p" --chunk 71 || echo "FAILED $run $p"
  done
done
for run in B_5000 B_1250; do
  $R spark "$run" --spz splats_sh1.spz --preset call --chunk 71 || echo "FAILED $run sh1 call"
done
echo ALL DONE
