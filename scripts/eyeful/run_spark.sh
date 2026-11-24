#!/usr/bin/env bash
# scripts/eyeful/spark_score.py with the extra packages it needs (Playwright, lpips, torchmetrics).
cd "$(dirname "$0")/../.." || exit 1
exec uv run --frozen --with playwright --with lpips==0.1.4 --with torchmetrics==1.6.0 \
    python scripts/eyeful/spark_score.py "$@"
