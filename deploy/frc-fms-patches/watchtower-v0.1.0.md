Watchtower's first tagged release: hub-counter plugins you can add without
editing Watchtower, their status on `/control`, and the YOLOv26-FRC-Model
colour and colour + model counters as plugins.

## New

### Counter plugins (`vision/`)

- **Add a counter without editing Watchtower:**
  - **One file:** copy `vision/plugins/_example.py` (a working, commented
    template) to `vision/plugins/my_counter.py` and change its `NAME`. Then
    set `counter: <that NAME>` in `config/vision.yaml`.
  - **A whole repo:** `python run_vision.py --add-plugin ~/dev/TBACroppedOutVid`
    prints the counters it found (`tbavid-colour / tbavid-combo`).
    `--remove-plugin <repo>` undoes it. It works the same on macOS, Windows
    and Linux.
  - **Installed packages:** anything that registers the `watchtower.counters`
    entry point is found by short name. So is a folder listed under
    `plugin_paths:` in `config/vision.yaml`.
- **`python run_vision.py --list-counters`** lists every counter with where it
  came from, what it needs and its options.
- **Mistakes are caught at startup:**
  - A misspelt option is flagged (`ball_aera` → "did you mean `ball_area`?").
  - A plugin written for a newer plugin API is refused with a message.
  - A broken plugin file cannot stop vision from starting.
- **Built-in names** (`zone`, `linecross`, `mock`) cannot be taken over by a
  plugin.
- **Older plugins work unchanged.** Every new plugin attribute is optional.

### `/control`

- **Status lines:** each hub's counter shows its own status line under Setup →
  Vision (e.g. `colour 123 · model 118`), with warnings in amber.
- **Load failures:** a plugin that fails to load shows its error there.
  Before, the hub just read "never connected".

### Counting

- **`zone` counts only the `fuel` class.** A model that also detects robots
  used to count robots as fuel.

## The YOLOv26-FRC-Model counters

They need [YOLOv26-FRC-Model](https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model)
v0.4.2 or later, cloned next to this repository and added with
`--add-plugin`.

- **`tbavid-colour`** counts yellow balls falling into a hub outline. It needs
  no model or GPU. Error against the official counts:
  - 10.1% mean on four 2026 Einstein broadcasts;
  - 6.4% on the Central Valley broadcast, with the outline's top edge 3–4
    ball widths above the hood.
- **`tbavid-combo`** blends that count with a fuel model.
  - **Error:** 6.8% on the Einstein matches (expect 7–9% on a new match), but
    34.5% on Central Valley. Check it on your own camera before choosing it.
  - **Hardware:** it needs Apple silicon or an NVIDIA GPU. When the model
    cannot keep up, it turns itself off and the hub counts by colour.
  - **The model file:** `model: built-in` uses `fuel_relabel.pt` placed in that
    repository's `models/` folder. It is attached to every YOLOv26-FRC-Model
    release.
- **Scoring:** both send each ball stamped with its camera capture time, so
  Watchtower places it in the right shift when it applies the inactive-hub
  rule (3 s grace).

## Checked

- **Tests:** 42 pass (`python -m pytest -q`).
- **End to end:** against a running Watchtower, both counters loaded by short
  name and showed their status on `/control`.

## Not checked yet

- Counting on a real practice hub: the 20-ball hand-count test.
- Settings for the 2026 rules: shift timing and grace are this repository's
  defaults (25 s shifts × 4, 3 s grace). Check them against the game manual.
