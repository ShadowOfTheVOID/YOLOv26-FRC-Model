#!/usr/bin/env python3
"""Standalone entrypoint for the scouting API. Standard library only.

`run.py serve` works too, but run.py imports the whole pipeline, which drags in
numpy and requests. This imports the API and the database layer and nothing
else, so a host needs no pip install at all -- no wheels, no build step, no
compile step on a small instance.

    python3 serve.py                 # 127.0.0.1:8781
    HOST=0.0.0.0 PORT=8080 python3 serve.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tbavid.api import serve
from tbavid.db import DB_PATH

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8781))
    if not DB_PATH.exists():
        raise SystemExit(
            f"No database at {DB_PATH}.\n"
            "Build it on the machine that has the videos:\n"
            "    python3 run.py db sync\n"
            "then upload data/scouting.db next to this file.")
    serve(host, port)
