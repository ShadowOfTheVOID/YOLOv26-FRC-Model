# Patches for frc-fms

Two commits for [frc-fms](https://github.com/arnan-bajaj/frc-fms), made
against its `main` at 20524ce. They are kept here because this session could
not push to that repository. They apply cleanly there, and its tests pass
afterwards (26).

1. **`zone` counts only the fuel class.** `zone` passed every box the model
   returned to its tracker, so a model that also detects robots (`fuel,
   robot_blue, robot_red`) counted robots in the hub region as fuel. On one
   frame of Einstein 1 a three-class model gave 6 robot boxes in a wide
   region. It now uses the class named `fuel` (a `classes:` key overrides).
   A one-class model keeps working as before.
2. **Docs for this repo's counter plugin.** The README's counters table and
   a short "External hub counter" section: what `ColourCounter` /
   `ComboCounter` do, what they measured, a `vision.yaml` example and the
   `PYTHONPATH` they need. Plus a commented pointer in
   `config/vision.example.yaml`, a note in its CLAUDE.md, and the README's
   expected test count (16, already 22 before this) corrected to 26.

Apply, in a checkout of frc-fms:

```bash
cd ~/dev/frc-fms
git checkout -b tbavid-counter
git am ~/dev/TBACroppedOutVid/deploy/frc-fms-patches/*.patch
python -m pytest -q          # expect: 26 passed
git push -u origin tbavid-counter    # then open a pull request on GitHub
```

If its `main` has moved and `git am` stops on a conflict: `git am --abort`,
then ask for the patches to be regenerated.
