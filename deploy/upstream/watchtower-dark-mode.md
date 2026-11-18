# Courtesy PR for arnan-bajaj/watchtower-fms: dark mode

`watchtower-dark-mode.patch` (this folder) against watchtower-fms `090371f`
(main, 2026-11-18). It applies cleanly there and his 143 tests pass with it.

## Open it (about 2 minutes)

1. On GitHub, open https://github.com/arnan-bajaj/watchtower-fms and press **Fork**.
2. In a terminal:

   ```bash
   git clone https://github.com/<your-username>/watchtower-fms && cd watchtower-fms
   git checkout -b dark-mode
   git apply /path/to/YOLOv26-FRC-Model/deploy/upstream/watchtower-dark-mode.patch
   git commit -am "Pages: dark mode that follows the device setting"
   git push -u origin dark-mode
   ```

   His `authors` check fails any commit that credits Claude or Anthropic, so
   commit it as yourself and add no trailers.
3. GitHub shows **Compare & pull request**. Use the title and text below.

## Title

Pages: dark mode that follows the device setting

## Description

Refs, scorers and the emcee use their phones in dim gyms, where a white page is
the brightest thing in the stands. This adds a dark mode to style.css that
follows the phone's or laptop's own setting (`prefers-color-scheme`). Light
mode is unchanged.

- `style.css`: a dark token set under `@media (prefers-color-scheme: dark)`.
  Two new tokens, `--on-ink` (text on an ink background) and `--bar` (the
  status bar), keep today's light values, so places that put white text on
  `--ink` still read once ink turns light in dark mode. Inputs use `--panel`
  instead of `#fff` (the same colour in light mode).
- `display.html` and `overlay.html` opt out with `class="fixed-theme"` on
  `<html>`: one is a TV across the gym, the other is drawn over the stream,
  and both already use dark backgrounds.
- `control.html` (the field-control select button) and `index.html` (the
  Control station card) use the new tokens in place of `#fff` / `--ink`.

Checked in Chromium with an 18-team event and a played match:

- Light mode: every page is pixel-identical before and after (`/`, `/ref`,
  `/emcee`, `/control`, `/display`, `/overlay`). The only differences found
  were the Flip button's 3-second mouse timer and a 3-pixel antialiasing edge,
  which also differ between two runs of the unchanged code.
- Dark mode: `/`, `/ref`, `/emcee` and `/control` read correctly, and
  `/display` and `/overlay` look as they did.
- `python -m pytest -q`: 143 passed.
