"""`python -m litereality_agent.room_ops.roomcall [SCENE] [--demo] [--host] [--port] [--label]
[--no-open]` — the same as `litereality roomcall`."""

from __future__ import annotations

from litereality_agent.room_ops.roomcall.server import main

if __name__ == "__main__":
    raise SystemExit(main())
