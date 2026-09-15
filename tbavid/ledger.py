"""The never-pull-twice record.

Keyed on YouTube video ID, because one stream can be linked to many TBA
matches -- keying on match key would let the same video in repeatedly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

from .config import LEDGER_PATH

# Statuses that mean "we got usable footage out of this".
OK = "ok"
# Terminal-ish failures. Skipped on future runs unless --retry-failed.
TOO_LONG = "too_long"
TOO_SHORT = "too_short"
UNAVAILABLE = "unavailable"
FAILED = "failed"
REJECTED = "rejected"  # downloaded fine, but camera classification untrustworthy

RETRYABLE = {FAILED, UNAVAILABLE}


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self.entries: Dict[str, dict] = {}
        if path.exists():
            try:
                self.entries = json.loads(path.read_text())
            except json.JSONDecodeError:
                backup = path.with_suffix(".json.corrupt")
                path.replace(backup)
                print(f"  ! ledger was unreadable, moved to {backup}")

    def seen(self, yt_key: str, retry_failed: bool = False) -> bool:
        entry = self.entries.get(yt_key)
        if entry is None:
            return False
        if retry_failed and entry.get("status") in RETRYABLE:
            return False
        return True

    def get(self, yt_key: str) -> Optional[dict]:
        return self.entries.get(yt_key)

    def record(self, yt_key: str, status: str, **extra) -> None:
        entry = self.entries.setdefault(yt_key, {})
        entry.update(extra)
        entry["status"] = status
        entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry.setdefault("first_seen_at", entry["updated_at"])
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.entries.values():
            s = e.get("status", "?")
            out[s] = out.get(s, 0) + 1
        return out
