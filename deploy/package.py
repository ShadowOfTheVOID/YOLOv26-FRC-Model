#!/usr/bin/env python3
"""Build distributable archives, identically on macOS, Windows and Linux.

    python3 deploy/package.py release            -> dist/tbavid-<date>.{zip,tar.gz,tar.xz}
    python3 deploy/package.py release v0.2.0     -> dist/tbavid-v0.2.0.{...}
    python3 deploy/package.py code               -> tbavid_code.tgz     (no key)
    python3 deploy/package.py code --with-key    -> tbavid_code_WITH_KEY.tgz

Python, not shell, because the shell versions are bash-only and Windows was
told to package by hand. zip/tar/xz/sha256sum are not a portable set of
binaries -- macOS has no sha256sum, Git Bash usually has no zip -- but
zipfile, tarfile, lzma and hashlib are in the standard library everywhere,
and this project already requires Python.

Source only: no videos, frames, database or model weights. Every artifact is
checked for the TBA key before it is kept.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Explicit allowlist. A denylist would eventually miss something, and the thing
# it misses is the file with the key in it.
INCLUDE = ["run.py", "serve.py", "config.json", "requirements.txt", "README.md",
           "SCOUTING.md", "DATA.md", "CHANGELOG.md", "LICENSE",
           "tbavid", "train", "tests", "deploy", "docs"]

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".venv-train", "dist",
                "data", "state", "runs", "weights", "dataset", "logs"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}


def collect(root: Path = ROOT):
    """Every file to ship, as (absolute path, path inside the archive)."""
    out = []
    for item in INCLUDE:
        src = root / item
        if not src.exists():
            continue                      # deleted from the repo; not fatal
        if src.is_file():
            out.append((src, Path(item)))
            continue
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if set(rel.parts) & EXCLUDE_DIRS or path.suffix in EXCLUDE_SUFFIX:
                continue
            # *WITH_KEY* files live in deploy/ alongside everything else, and
            # deploy/ is bundled wholesale -- without this the "safe to upload"
            # archive quietly carries the key. That is not hypothetical.
            if "WITH_KEY" in path.name:
                continue
            out.append((path, rel))
    return sorted(out, key=lambda p: p[1].as_posix())


def tba_key(root: Path = ROOT) -> bytes:
    """The configured key, so artifacts can be proven not to contain it."""
    env = root / ".env"
    if not env.exists():
        return b""
    for line in env.read_text(errors="replace").splitlines():
        name, _, val = line.strip().partition("=")
        if name.strip() == "TBA_AUTH_KEY":
            return val.strip().strip("'\"").encode()
    return b""


def assert_keyless(blobs, key: bytes, what: str) -> None:
    """Refuse to ship anything containing the key, whatever the filename."""
    if not key:
        return
    for name, data in blobs:
        if key in data:
            raise SystemExit(f"ABORT: {what} contains your TBA key: {name}")


def _mode_for(path: Path) -> int:
    """Windows has no executable bit; keep .sh runnable when built there."""
    if path.suffix == ".sh" or path.name.endswith(".bash"):
        return 0o755
    return 0o755 if os.access(path, os.X_OK) else 0o644


def write_tar(dest: Path, files, prefix: str, compression: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, f"w:{compression}") as tar:
        for src, rel in files:
            info = tar.gettarinfo(str(src), arcname=f"{prefix}/{rel.as_posix()}"
                                  if prefix else rel.as_posix())
            info.mode = _mode_for(src)
            # Ownership from the building machine is noise in a source archive.
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            with src.open("rb") as fh:
                tar.addfile(info, fh)


def write_zip(dest: Path, files, prefix: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, rel in files:
            arc = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
            info = zipfile.ZipInfo(arc, date_time=dt.datetime.fromtimestamp(
                src.stat().st_mtime).timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (_mode_for(src) & 0xFFFF) << 16
            zf.writestr(info, src.read_bytes())


def read_back(path: Path):
    """Every member of a finished archive, for the second key check."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return [(n, zf.read(n)) for n in zf.namelist()]
    with tarfile.open(path) as tar:
        out = []
        for m in tar.getmembers():
            if m.isfile():
                fh = tar.extractfile(m)
                if fh:
                    out.append((m.name, fh.read()))
        return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_release(args) -> int:
    version = args.version or dt.date.today().strftime("%Y.%m.%d")
    name = f"tbavid-{version}"
    dist = ROOT / "dist"

    files = collect()
    key = tba_key()
    assert_keyless(((str(r), s.read_bytes()) for s, r in files), key,
                   "the staged tree")

    built = []
    for suffix, comp in ((".zip", None), (".tar.gz", "gz"), (".tar.xz", "xz")):
        out = dist / f"{name}{suffix}"
        if comp is None:
            write_zip(out, files, name)
        else:
            write_tar(out, files, name, comp)
        # Same check again, on the finished artifact rather than the staging
        # tree: the check that matters is the one on what actually ships.
        try:
            assert_keyless(read_back(out), key, str(out))
        except SystemExit:
            out.unlink(missing_ok=True)
            raise
        built.append(out)

    sums = dist / "SHA256SUMS"
    sums.write_text("".join(f"{sha256(f)}  {f.name}\n" for f in built))

    print("dist/")
    for f in built:
        print(f"  {f.name:<34} {f.stat().st_size/1024:.0f} KB")
    print(f"  {sums.name:<34} {len(built)} checksums")
    print(f"\nAll keyless and verified. {len(files)} files per archive.")
    return 0


def cmd_code(args) -> int:
    if args.with_key:
        if not (ROOT / ".env").exists():
            raise SystemExit("no .env to include")
        out = Path(args.out or "tbavid_code_WITH_KEY.tgz")
        files = collect() + [(ROOT / ".env", Path(".env"))]
        write_tar(out, files, "", "gz")
        try:
            os.chmod(out, 0o600)          # best effort; a no-op on Windows
        except OSError:
            pass
        print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
        print("\n  !! This archive CONTAINS your TBA key.")
        print("  !! Copy it only to machines you control (scp). Never upload it")
        print("  !! to Kaggle, Colab, GitHub or anywhere shareable.")
        print("  !! For those, run this without --with-key instead.")
        return 0

    out = Path(args.out or "tbavid_code.tgz")
    files = collect()
    key = tba_key()
    write_tar(out, files, "", "gz")
    try:
        assert_keyless(read_back(out), key, str(out))
    except SystemExit:
        out.unlink(missing_ok=True)
        raise
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    print("Contains no .env, no key, no data/, no state/ -- safe to upload.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("release", help="dist/ archives for a GitHub release")
    p.add_argument("version", nargs="?", help="defaults to today's date")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("code", help="one .tgz of the code for another machine")
    p.add_argument("out", nargs="?", help="output path")
    p.add_argument("--with-key", action="store_true",
                   help="include .env -- for machines you control only")
    p.set_defaults(func=cmd_code)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
