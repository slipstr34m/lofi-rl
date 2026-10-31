"""The rating UI: a stdlib server on 127.0.0.1 and a vanilla-JS page (docs/RATING.md).

- `manifest`: batch manifests (public and hidden), validation, and the synthetic demo batch.
- `votes`: the event and vote schemas, the fsynced append-only log, and the queue derived from it.
- `audit`: aggregates over a finished batch (no per-clip scores, no model output).
- `serve`: the HTTP server (`make rate`).

Nothing here imports a model, numpy or any audio library at import time.
"""
