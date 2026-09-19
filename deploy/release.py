#!/usr/bin/env python3
"""Cut a release: run every gate CI runs, then hand you the tag command.

    python3 run.py release v0.2.1            # check and build, change nothing
    python3 run.py release v0.2.1 --tag      # ...and write the tag locally
    python3 run.py release v0.2.1 --push     # ...and push it, which publishes

Releasing here is "push a tag and let .github/workflows/release.yml do the
rest", which is a good arrangement with one sharp edge: every check lives on
the far side of the push. A missing CHANGELOG section or a failing test is
found *after* the tag is public, and the fix is deleting a pushed tag, which
is the one git operation nobody enjoys explaining.

So this runs the same gates first, locally, in the same order the workflow
does. Passing here is not a promise that CI will pass -- it runs on a clean
checkout with its own Python -- but everything that has actually gone wrong
with a release of this project would have been caught right here.

Nothing is tagged or pushed without the matching flag. The default is a
rehearsal.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))

import package                                              # noqa: E402

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")

# The workflow greps the tree for anything shaped like a committed key. Same
# pattern here: a key found locally costs a minute, one found in CI costs a
# tag, and one found by nobody costs the key.
KEY_PATTERN = (r'(TBA_AUTH_KEY|X-TBA-Auth-Key)[[:space:]]*[=:]'
               r'[[:space:]]*["'"'"']?[A-Za-z0-9]{40,}')


class Failed(Exception):
    """A gate said no. The message is what the human needs to do about it."""


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=check)


def step(name):
    print(f"\n=== {name} ===", flush=True)


def ok(msg):
    print(f"  ok    {msg}", flush=True)


def warn(msg):
    print(f"  warn  {msg}", flush=True)


# ------------------------------------------------------------------- gates

def check_version(version):
    if not VERSION_RE.match(version):
        raise Failed(f"{version!r} is not vMAJOR.MINOR.PATCH, e.g. v0.2.1.\n"
                     "  The workflow triggers on tags matching v*, and the "
                     "CHANGELOG\n  headings are written that way too.")
    ok(f"{version} is a well-formed version")


def check_changelog(version):
    path = ROOT / "CHANGELOG.md"
    if not path.exists():
        raise Failed("no CHANGELOG.md")
    try:
        body = package.changelog_section(version, path.read_text())
    except SystemExit as exc:
        raise Failed(f"{exc}\n"
                     "  The release notes ARE this section -- CI fails rather "
                     "than\n  publishing an empty body. Add:\n\n"
                     f"    ## {version} -- <date>\n\n"
                     "  ...describing what changed, then run this again.")
    if not body.strip():
        raise Failed(f"CHANGELOG.md has a {version} heading but nothing under it")
    lines = [l for l in body.splitlines() if l.strip()]
    ok(f"CHANGELOG.md has {version} ({len(lines)} lines of notes)")


def check_tag_free(version):
    local = git("tag", "--list", version).stdout.strip()
    if local:
        raise Failed(f"tag {version} already exists locally.\n"
                     f"  Delete it with:  git tag -d {version}\n"
                     "  ...or pick the next version.")
    remote = git("ls-remote", "--tags", "origin", f"refs/tags/{version}",
                 check=False).stdout.strip()
    if remote:
        raise Failed(f"tag {version} is already on origin -- that release is "
                     "published.\n  Pick the next version.")
    ok(f"{version} is not taken, locally or on origin")


def check_tree_clean():
    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        listing = "\n".join(f"    {l}" for l in dirty.splitlines()[:10])
        raise Failed(f"{n} uncommitted change(s). A tag points at a commit, so "
                     f"anything\n  not committed is not in the release:\n\n"
                     f"{listing}")
    ok("working tree is clean")


def check_branch():
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        warn(f"on branch {branch}, not main -- the tag will point there")
    else:
        ok("on main")
    git("fetch", "--quiet", "origin", check=False)
    behind = git("rev-list", "--count", "HEAD..@{upstream}",
                 check=False)
    if behind.returncode == 0 and behind.stdout.strip() not in ("", "0"):
        warn(f"{behind.stdout.strip()} commit(s) on the remote are not in "
             "this checkout")


def check_no_key():
    found = git("grep", "-nIE", KEY_PATTERN, "--", ".", check=False)
    if found.returncode == 0 and found.stdout.strip():
        raise Failed("something shaped like a TBA key is committed:\n"
                     + found.stdout.strip()[:500])
    tracked = git("ls-files", check=False).stdout.splitlines()
    withkey = [f for f in tracked if "WITH_KEY" in f.upper()]
    if withkey:
        raise Failed("a *WITH_KEY* file is tracked: " + ", ".join(withkey))
    ok("no key in the tracked tree")


def run_tests():
    proc = subprocess.run([sys.executable, str(ROOT / "tests" /
                                               "test_pipeline.py")],
                          cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    if proc.returncode != 0:
        raise Failed("tests failed -- CI runs these before it publishes, so "
                     "this\n  would have burned the tag:\n\n    "
                     + "\n    ".join(tail[-12:]))
    ok(tail[-1] if tail else "tests passed")


def build(version):
    args = argparse.Namespace(version=version)
    package.cmd_release(args)


def check_archive(version):
    """Unpack what was just built and prove it runs, as the workflow does."""
    import tarfile, tempfile
    tgz = ROOT / "dist" / f"tbavid-{version}.tar.gz"
    if not tgz.exists():
        raise Failed(f"expected {tgz} to exist after the build")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tgz) as tar:
            tar.extractall(tmp)
        unpacked = Path(tmp) / f"tbavid-{version}"
        for cmd, what in (([sys.executable, "tests/test_pipeline.py"], "tests"),
                          ([sys.executable, "run.py", "--help"], "run.py")):
            proc = subprocess.run(cmd, cwd=unpacked, capture_output=True,
                                  text=True)
            if proc.returncode != 0:
                raise Failed(f"the built archive's {what} failed:\n"
                             + (proc.stderr or proc.stdout)[-800:])
        ok("the archive unpacks, passes its tests and runs")


# ------------------------------------------------------------------ driver

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run.py release", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="e.g. v0.2.1")
    ap.add_argument("--tag", action="store_true",
                    help="write the annotated tag locally (still not pushed)")
    ap.add_argument("--push", action="store_true",
                    help="push the tag, which publishes the release")
    ap.add_argument("--skip-tests", action="store_true",
                    help="for a re-run after a known-good test pass")
    args = ap.parse_args(argv)
    version = args.version if args.version.startswith("v") else f"v{args.version}"

    try:
        step("version")
        check_version(version)
        step("changelog")
        check_changelog(version)
        step("git")
        check_tag_free(version)
        check_tree_clean()
        check_branch()
        check_no_key()
        if args.skip_tests:
            step("tests")
            warn("skipped at your request -- CI will still run them")
        else:
            step("tests")
            run_tests()
        step("build")
        build(version)
        step("archive")
        check_archive(version)
    except Failed as exc:
        print(f"\n  FAILED: {exc}\n", file=sys.stderr)
        print("Nothing was tagged or pushed.", file=sys.stderr)
        return 1

    step("result")
    print(f"  {version} is ready. Everything CI checks passes here.")

    if not (args.tag or args.push):
        print("\n  Nothing has been tagged. To publish:")
        print(f"    python3 run.py release {version} --push")
        print("  or by hand:")
        print(f"    git tag -a {version} -m {version} && git push origin {version}")
        return 0

    notes = package.changelog_section(version, (ROOT / "CHANGELOG.md").read_text())
    subject = next((l.strip() for l in notes.splitlines() if l.strip()), version)
    git("tag", "-a", version, "-m", f"{version}\n\n{subject}")
    print(f"  tagged {version} locally")

    if not args.push:
        print("\n  Not pushed. When you are ready:")
        print(f"    git push origin {version}")
        print(f"  Changed your mind:  git tag -d {version}")
        return 0

    print(f"\n  pushing {version} to origin -- this publishes the release")
    proc = subprocess.run(["git", "push", "origin", version], cwd=ROOT)
    if proc.returncode != 0:
        print(f"\n  push failed. The local tag is still there: "
              f"git tag -d {version} to undo.", file=sys.stderr)
        return 1
    print(f"\n  pushed. The release workflow is building it now:")
    print("    https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model/actions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
