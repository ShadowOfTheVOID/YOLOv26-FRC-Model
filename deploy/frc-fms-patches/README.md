# Patches for watchtower-fms (frc-fms)

Commits for [watchtower-fms](https://github.com/arnan-bajaj/watchtower-fms)
(formerly frc-fms), kept here because this repository's sessions cannot push
there. Apply them in a checkout of it with `git am`.

| patch | what | status |
|---|---|---|
| 0001 | `zone` counts only the class named `fuel` (a 3-class model counted robots) | **in its main** (77199c9) |
| 0002 | README docs for this repo's counter plugin | **in its main** (77199c9) |
| 0003 | `/control` shows a counter's own status line (detail / warning) | new |
| 0004 | A counter plugin system: plugin lookup by short name, `plugin_paths:`, `--list-counters`, option checks, `status()`, `close()` | new |
| 0006 | The Watchtower app: `.github/workflows/app.yml` builds it on each tag for Mac, Windows, Linux and Pi, from YOLOv26-FRC-Model's `apps/watchtower/` at the tag in `.github/hubcounter-release` (v0.4.3) | new |
| 0005 | Easier plugins: drop a `.py` into `vision/plugins/`, or `run_vision.py --add-plugin <repo folder>` | new (needs 0004) |

## 0003 + 0004: the plugin system

- **Lookup.** `counter:` takes a built-in (`zone`, `linecross`, `mock`), a
  plugin's short name, a `plugins:` alias, or `"module:Class"`. Plugins come
  from two places:
  - installed packages registering the `watchtower.counters` entry points
    (`pip install -e ../YOLOv26-FRC-Model`, then `counter: tbavid-colour`);
  - folders in `plugin_paths:` (no install, no `PYTHONPATH`).
- **`python run_vision.py --list-counters`** lists every counter with its
  description, needs and options.
- **Optional plugin attributes:** `PLUGIN_API`, `NAME`, `DESCRIPTION`,
  `NEEDS` and `OPTIONS`.
  - A misspelt option is flagged at startup, e.g. `ball_aera` → "did you
    mean `ball_area`?".
  - A plugin written for a newer API is refused with a message.
- **`status()`** goes into each second's report and shows under the hub in
  `/control` → Setup → Vision.
- **`close()`** is called once when vision stops.
- **A plugin that fails to load** shows its error on `/control`. Before,
  the hub thread died silently.
- **Older plugins** with none of the new attributes work unchanged.

Checked: both patches apply cleanly to its main at 77199c9, and its tests
pass afterwards (37: 26 before, plus 11 for the plugin system). End to end
against a running watchtower-fms, both loaded by short name with no
`PYTHONPATH`, either pip-installed or through `plugin_paths:` only:
- `tbavid-colour` showed `colour 123` on `/control`;
- `tbavid-combo` on a CPU-only machine showed `colour 29 (model off)` with
  an amber warning;
- `ball_aera` was flagged.

## 0005: adding a plugin without editing anything

- **One file:** copy `vision/plugins/_example.py` (a working, commented
  template) to `vision/plugins/my_counter.py` and change its `NAME`; then
  `counter: <that NAME>`. Files there are read with `ast`, never imported,
  so a broken one cannot stop vision from starting.
- **A whole repo:** `python run_vision.py --add-plugin ~/dev/TBACroppedOutVid`
  prints `added: counter: tbavid-colour / tbavid-combo now works`. It writes
  a one-line `vision/plugins/TBACroppedOutVid.path` file holding the folder,
  not a symlink, so it works on Windows. `--remove-plugin TBACroppedOutVid`
  undoes it. If the folder later moves, it is listed as missing rather than
  crashing.
- `vision/plugins/` is gitignored except the template and its README, so
  people's own counters stay local. Built-in names cannot be taken over.

Checked: 0003-0005 apply to its main at 77199c9 and its tests pass (42:
5 new). With this repository, `--add-plugin` found both counters, both
loaded, and the copied template counted.

## Apply

```bash
cd ~/dev/watchtower-fms
git checkout main
git pull
git checkout -b plugin-system
git am ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0003-*.patch ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0004-*.patch ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0005-*.patch
source .venv/bin/activate
python -m pytest -q
git push -u origin plugin-system
```

The tests should say `42 passed`. If 0003 and 0004 are already applied,
apply only 0005. Then open a pull request on GitHub. If
`git am` stops on a conflict (its main has moved), run `git am --abort` and
ask for the patches to be regenerated.

## Release Watchtower v0.1.0 (0003-0005 applied)

Watchtower has no tags or release workflow yet, so this is its first
release. The notes are `watchtower-v0.1.0.md` beside this file. Checked:
0003-0005 apply to its main at 77199c9 and its tests pass (42).

The changes go in as a pull request, not straight onto Arnan's main. With
the GitHub CLI (`brew install gh`, then `gh auth login`) it is two blocks.

1. Branch, apply, test, open the pull request:

```bash
cd ~/dev/TBACroppedOutVid
git pull
cd ~/dev/watchtower-fms
git checkout main
git pull
git checkout -b release-v0.1.0
git am ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0003-*.patch ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0004-*.patch ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0005-*.patch
source .venv/bin/activate
python -m pytest -q
git push -u origin release-v0.1.0
gh pr create --repo arnan-bajaj/watchtower-fms --base main --head release-v0.1.0 --title "Counter plugins: drop-in folder, --add-plugin, /control status (v0.1.0)" --body-file ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/watchtower-v0.1.0.md
```

2. After the pull request is merged, tag and publish:

```bash
cd ~/dev/watchtower-fms
git checkout main
git pull
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --repo arnan-bajaj/watchtower-fms --title "v0.1.0" --notes-file ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/watchtower-v0.1.0.md
```

Without the GitHub CLI: after `git push -u origin release-v0.1.0`, open
https://github.com/arnan-bajaj/watchtower-fms/compare/main...release-v0.1.0
to make the pull request; after the tag, Releases -> Draft a new release ->
tag `v0.1.0`, paste `watchtower-v0.1.0.md`, Publish.

## 0006: the Watchtower app in Watchtower's own releases

Watchtower's own releases then carry `Watchtower-mac.zip`, `-windows.zip`
(0006 predates this repository's .dmg and Windows installer, from v0.4.4),
`-linux-x64.tar.gz` and `-linux-arm64.tar.gz`. These are the same app as on
YOLOv26-FRC-Model's releases, built with Watchtower's own code. The
launcher, spec and smoke test stay in this repository; its workflow checks
out this repository at the tag in `.github/hubcounter-release`. So **this
repository's v0.4.3 must be tagged first**, or the workflow has nothing to
check out.

```bash
cd ~/dev/watchtower-fms
git checkout main
git pull
git checkout -b app-builds
git am ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0006-*.patch
git push -u origin app-builds
gh pr create --repo arnan-bajaj/watchtower-fms --base main --head app-builds --title "The Watchtower app: build it on each tag" --body "Builds Watchtower as a double-click app for Mac, Windows, Linux and Pi on each tag (.github/workflows/app.yml)."
```

After that pull request is merged, release Watchtower v0.1.1 with the apps:

```bash
cd ~/dev/watchtower-fms
git checkout main
git pull
git tag -a v0.1.1 -m "v0.1.1"
git push origin v0.1.1
gh release create v0.1.1 --repo arnan-bajaj/watchtower-fms --title "v0.1.1" --notes "The Watchtower app: double-click Watchtower for Mac, Windows, Linux and Raspberry Pi, with the hub counter built in. Downloads below (attached by the app workflow in about 20 minutes)."
```

