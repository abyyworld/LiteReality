# tea_room: first real `--realistic` run on Modal (2026-11-13)

```bash
uv run litereality setup
uv run litereality roomcall ~/LiteReality-Agent/example-scans/tea_room --realistic --deploy
```

MacBook (Apple Silicon, Python 3.10). TRELLIS on Modal H100s (up to 4 at once). The room-call
server ran as one Modal CPU container. Every time below is wall-clock time on this run.

## Time

| Step | Time |
|---|---|
| `litereality setup` (image cache warm: nothing rebuilt) | 10 s |
| Quick room built on the laptop | 2.0 s |
| Room-call server image, first build since the scan-upload patch | ~25 s |
| Room-call deploy | 28 s |
| First generated object (Chair0 preview) on the link, after generation started | 122 s |
| All previews | 275 s |
| All finals (10 objects) | 430 s |
| Whole command, start to exit | 7 min 52 s |

Each figure in the table below is that call's own latency, from when it was sent to Modal until
its result came back. With 4 GPUs for 11 objects × 2 passes, later calls also include time spent
waiting in the queue.

| Object | Preview (s) | Final (s) |
|---|---|---|
| Chair0 | 121.8 | 129.7 |
| Sink_Storage0 | 131.9 | 112.4 |
| Storage1 | 143.4 | 55.9 |
| Chair2 | 215.0 | 53.7 |
| Sofa0 | 225.8 | 47.9 |
| Chair1 | 235.9 | 111.0 |
| Storage3 | 115.8 | 41.0 |
| Storage0 | 260.1 | 36.3 |
| Dishwasher0 | 275.3 | 144.0 |

## What came out

- 10 objects generated (final pass).
- Chair3 borrows Chair1's mesh, because its own view was too partial.
- Table0 kept its template: the capture saw too little of it (best view score 0.21) and no other
  object looks like it.
- 11 live versions were published to the link. The download size grows with each one: 0.9 MB
  for the first, 62 MB for the last. **62 MB is heavy for people joining a call on a phone.**

## Cost (Modal billing report, `modal billing report --for today --resolution h --show-resources`)

| App | Resource | Cost |
|---|---|---|
| litereality (TRELLIS) | H100 | $2.139 |
| litereality (TRELLIS) | memory + CPU + egress | $0.100 |
| litereality-roomcall (server) | CPU + memory + egress | $0.002 |
| **Total** | | **≈ $2.24** |

Read an hour after the run, so it includes the GPU containers' idle time before they shut down
(about $0.38 more than the report taken right after the run showed).

Notes:
- Before this run, the month's metered cost was $2.10, almost all of it Volume storage, which the
  free storage allowance covers. The amount billed stayed at $0.
- The docs' estimate was "~$2" and "all finals about 5 min". The measured run took 7.2 min to
  the last final and cost $2.24.
