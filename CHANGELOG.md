# Changelog

Notable changes per release. Dates are the release date, not the merge date.

## v0.2.0 — 2026-09-16

Harvesting is now restricted to official competition footage, and an existing
harvest can be checked against that same standard.

### Added

- **Competitive-only harvesting.** Only official competition events are walked:
  regional, district, district championship, championship division,
  championship final and Festival of Champions. Offseason, preseason and
  unlabeled events are never picked. Within an event only real match play
  counts (`qm`, `ef`, `qf`, `sf`, `f`), so practice matches are skipped even
  when TBA has a video for them.
- **`run.py audit`** reports which videos in an existing harvest are not
  competition footage, and how many frames each one contributes. It reports
  only and deletes nothing. A season TBA does not answer for is reported as
  unknown rather than condemned.
- **`DATA.md` and `docs/provenance/`** record what is in the current dataset,
  where the frame bundle lives, and the data-quality problems found in it.

### Changed

- Event listing uses `/events/{year}/simple` instead of `/events/{year}/keys`,
  because a bare event key does not carry `event_type`. Still one request per
  season, still cached and revalidated with `If-None-Match`.

### Fixed

- `deploy/make_code_archive.sh` built its file list from a hardcoded string and
  aborted once `QUICKSTART_DEBIAN.txt` was deleted from the repo — leaving a
  half-written archive behind that had never reached the keyless check. The
  list is now built from what is on disk, and a failed run removes its own
  partial output.
- `deploy/make_release.sh` no longer stages the deleted `QUICKSTART_DEBIAN.txt`,
  and now includes `DATA.md`, `CHANGELOG.md`, `LICENSE` and `docs/`.
- Packaging works on Windows and macOS, not just Linux. Both `make_*.sh`
  scripts are now thin wrappers around `deploy/package.py`, which uses
  `zipfile`, `tarfile`, `lzma` and `hashlib` from the standard library instead
  of `zip`, `tar`, `xz` and `sha256sum` — macOS has no `sha256sum` and Git Bash
  usually has no `zip`. `release` now also writes `dist/SHA256SUMS`.

### Escape hatch

- `--include-noncompetitive` on `pull`/`fetch`, or `"competitive_only": false`
  in `config.json`, restores the previous behaviour. The flag wins over config.

### Known issues

- The frame bundle harvested before this release contains 926 frames (12.0%)
  from `2026kylou`, an offseason event. `run.py audit` finds them; nothing
  deletes them for you.
- `2026gal_qm62` carries a bad scoreboard read (blue final 726 against a 19–285
  range elsewhere) and is not flagged as an error.
- A one-alliance scoreboard read is still recorded as a good read: the
  `scoreboard_ok` test in `tbavid/db.py` is satisfied by a series containing
  only one alliance. `2026mnmi2_qm11` and `2026mnmi_qm39` are affected.

## v0.1.0-beta — 2026-09-15

Released as tag `Beta`. First working end-to-end pipeline: pick unseen match
videos from TBA, strip them to main-camera footage, crop the scoreboard off,
export sampled frames, and read the fuel counters before discarding the banner.
Includes the scouting database, the read-only JSON API, and deployment notes
for Debian, Windows, Colab and Kaggle.
