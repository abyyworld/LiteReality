#!/bin/bash
cd /Users/abyyworld/LiteReality
S() { uv run --frozen --with playwright --with lpips==0.1.4 python results/stray_quality/score.py run "$1" "run/atrium_stray_q/$1" > "results/stray_quality/logs/$1.log" 2>&1; }
( S v0; S v2 ) &
( S v1; S v3 ) &
wait
