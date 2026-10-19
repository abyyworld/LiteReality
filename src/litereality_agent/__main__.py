"""Legacy module entrypoint; prefer ``uv run litereality``.

The console script and this compatibility module both dispatch to :mod:`litereality_agent.cli`.
"""

from __future__ import annotations

from litereality_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
