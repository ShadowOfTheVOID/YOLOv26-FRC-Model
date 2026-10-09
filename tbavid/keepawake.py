"""Keep the computer awake while the hub counter is counting (stdlib).

A laptop left on the scoring table between matches goes to sleep after its
idle timeout -- nobody touches the keyboard while the cameras count -- and a
sleeping box sends nothing: bioarena shows OFFLINE until someone wakes it
and finds the cameras have to be reopened. Power settings changed by hand
are one more thing to forget on match day, so the counter holds the machine
awake itself, only while it counts, and lets go when it stops.

    release = hold("counting")      # call from the thread that counts
    ...
    release()

  * macOS: `caffeinate -di -w <pid>` (no idle sleep, display on; it dies
    with us even if we crash). Closing the lid still sleeps a Mac.
  * Windows: SetThreadExecutionState. The request belongs to the calling
    thread and lapses when that thread ends, so call `hold` from the thread
    that runs for the whole counting session.
  * Linux: `systemd-inhibit --what=idle:sleep`; without systemd nothing
    is held and the returned note says so.

Never raises: failing to hold the machine awake must not stop counting.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Callable, Optional, Tuple

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def _command(why: str, platform: str = sys.platform) -> Optional[list]:
    """The helper process that holds the machine awake, or None."""
    if platform == "darwin":
        if shutil.which("caffeinate"):
            return ["caffeinate", "-di", "-w", str(os.getpid())]
        return None
    if platform.startswith("linux"):
        if shutil.which("systemd-inhibit"):
            return ["systemd-inhibit", "--what=idle:sleep", "--who=hub counter",
                    f"--why={why}", "--mode=block", "sleep", "infinity"]
        return None
    return None


def hold(why: str = "counting fuel") -> Tuple[Callable[[], None], str]:
    """Hold the machine awake. Returns (release, note): `note` says what was
    done, for the log; `release()` undoes it and is safe to call twice."""
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            if not k.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED):
                return (lambda: None), "could not keep the computer awake"

            def release() -> None:
                try:
                    k.SetThreadExecutionState(ES_CONTINUOUS)
                except Exception:
                    pass
            return release, "keeping the computer awake while counting"
        except Exception as e:
            return (lambda: None), f"could not keep the computer awake: {e}"
    cmd = _command(why)
    if cmd is None:
        return (lambda: None), ("could not keep the computer awake (no caffeinate "
                                "or systemd-inhibit): turn off sleep by hand")
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        return (lambda: None), f"could not keep the computer awake: {e}"

    def release() -> None:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except Exception:
                p.kill()
    note = "keeping the computer awake while counting"
    if sys.platform == "darwin":
        note += " (closing the lid still puts it to sleep)"
    return release, note
