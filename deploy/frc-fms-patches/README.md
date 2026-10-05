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

## Apply

```bash
cd ~/dev/watchtower-fms
git checkout main
git pull
git checkout -b plugin-system
git am ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0003-*.patch ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/0004-*.patch
source .venv/bin/activate
python -m pytest -q
git push -u origin plugin-system
```

The tests should say `37 passed`. Then open a pull request on GitHub. If
`git am` stops on a conflict (its main has moved), run `git am --abort` and
ask for the patches to be regenerated.
