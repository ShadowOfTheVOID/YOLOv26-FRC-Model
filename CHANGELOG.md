# Changelog

Notable changes per release. Dates are the release date, not the merge date.

Each `## vX.Y.Z` section below is the release body: `.github/workflows/release.yml`
reads it with `python3 deploy/package.py notes vX.Y.Z` when the tag is pushed,
so the notes and this file cannot drift apart. A tag with no section here fails
the build rather than publishing an empty release.

## Unreleased

Three things: a whole event day can be harvested from one stream, the crop
knows which broadcast it is looking at, and the scouting app can read this
harvest's output as a source of its own.

### Added

- **`run.py live` — scout a live feed and keep no video.** Everything else
  here builds a training set, which is no use to somebody who wants to know how
  an alliance is scoring this afternoon. This reads the burned-in fuel counter
  off a live stream and deletes every frame the moment it has been read: peak
  disk is one chunk, nothing enters the manifest or the dataset, and the clip
  is unlinked on the failure paths too. What it gives is the per-alliance
  scoring timeline for the match on the field — when fuel went in, to the
  second — which nothing else in either repo produces live. What it cannot give
  is which robot: the counter says an alliance scored and never which of its
  three did, and being live does not move that ceiling.
  It refuses to guess the match: `--match` names it or `--hub` asks a running
  scouting hub what is on the field, and one is required, because a timeline
  filed against the wrong key credits an alliance's scoring to six robots that
  were not on it. Rows are marked `status='live'`, since a reading that cannot
  be re-read is not the same evidence as one that can.
- **`run.py stream` — one broadcast in, one clip per match out.** Events that
  publish a single multi-hour stream per day rather than per-match uploads were
  simply unreachable: the picker looks for `match.videos[]` and finds nothing,
  and `download.py` refuses the stream it is handed. Now the matches are found
  inside it and cut out, and the clips go through the same pipeline as anything
  else. `--file` reads one already on disk; `--listen-only` reports what the
  audio contains without writing anything.
- **Matches are found by listening, not watching** (`tbavid/audio.py`). Between
  matches the camera is looking at the same field from the same place, so there
  is no cut for `shots.py` to find. The field's start sound and buzzer are the
  only events in a broadcast that are loud, tonal, repeated dozens of times and
  separated by a fixed interval — and the fixed interval is what identifies
  them. There is no frequency in the file: a band-pass at the horn's pitch
  would need that pitch from somewhere, and a wrong guess finds nothing while
  looking exactly like a stream with no matches in it. Every loud tonal burst
  is collected and the spacing that repeats most consistently is the match, so
  it works on the Webcast Unit's YouTube feed, a Twitch re-encode or a phone in
  the stands alike.
- **`deploy/frc-harvest.service` and `run.py db export`, so the API is not
  something somebody has to start.** `python3 serve.py` on a laptop dies with
  the lid, and the scouting app then shows an empty column that looks exactly
  like "no footage of these robots" rather than "nothing is listening" -- two
  states it goes out of its way to distinguish. Under systemd it survives a
  crash, a reboot and a host rebuild, and it carries no secrets because a
  GET-only API over a rebuildable database has nothing to authenticate to.
  `db export` exists because `cp` is not good enough and fails confusingly: the
  working database is WAL, a WAL database must create its `-shm` companion
  before even a read-only connection can read it, and the unit mounts its data
  directory read-only on purpose -- so a copied database starts fine and then
  answers every request with "attempt to write a readonly database". Confirmed
  as an unprivileged user against a 0555 directory. The export goes through
  SQLite's backup API, which also avoids catching the file mid-write, and
  leaves one file with no sidecars. See [deploy/HOSTING.md](deploy/HOSTING.md).
- **Identity comes from TBA or not at all.** Audio says where a match is, never
  which one it is, and `db.py` joins a roster onto the match key — so one
  mislabelled clip credits an alliance's fuel to six robots that were not on
  the field. Keys are assigned when the confirmed cue count equals TBA's match
  count, when recovered cues close the gap to it exactly, or when
  `--from-match` says where the day starts. Otherwise none are, and that is not
  a failure: training frames do not need a match key, scouting rows do, so an
  unidentified clip keeps its frames and enters the database under its own
  video id where no roster can join onto it.

- **Broadcast layout profiles** (`tbavid/formats.py`). The crop has always
  measured the overlay from the pixels, which needs no list of events kept up
  to date. What it could not do is know when it was wrong, and it failed
  silently both ways: a divider found on a single-camera feed crops away the
  bottom of the field, and frames that lost half a field still look plausible.
  A profile says what a layout is supposed to be, so a measurement can be
  checked against it. The measurement still wins — a profile only tunes the
  thresholds, supplies the fallback when detection is inconclusive, and vetoes
  a band its layout does not have.
- **`ca_district`, for FIRST California's weekend district events** —
  `2026caclv`, `2026casnf`, `2026calas`, `2026caven`, `2026caoec` and
  Aerospace Valley. One field camera, a banner across the top, no permanent
  side panel, so its main job is that veto. Marked `declared` rather than
  `measured`: the veto and thresholds are in force, but the banner range
  describes the layout as specified, because no California district footage has
  been through this pipeline yet — and those events are not one production
  (most are FIRST Webcast Unit on YouTube; Central Valley is on Twitch), so
  the range is wide and per-event calibration is worth doing.
- **A profile can be gated to a TBA `event_type`,** and `ca_district` is gated
  to a weekend district event. FIRST California's state championships
  (`2026cancmp` and its southern counterpart) carry district `ca` and are a
  different, larger production: selecting them into a profile whose whole
  contribution is a no-split-screen veto would keep an entire side-camera
  panel in the training set if they do run one, which is the failure the
  profiles exist to prevent with the sign flipped. They fall through to
  `generic`, which has no opinion. An event whose type could not be read fails
  the gate rather than having it waived, and config still outranks it — that
  gate stops the pipeline guessing, not a person who has measured one.
- **`run.py formats`** — list the profiles, ask which one an event would get
  and why, or measure a real download with `--calibrate`. Calibration prints
  the profile its measurements imply and writes nothing; promoting a profile
  from `declared` to `measured` stays a person's edit, after they have looked
  at the frames.
- **The district comes from TBA.** `/events/{year}/simple` — the one request
  per season the video picker already makes — carries each event's `district`,
  so the layout is known before the download and costs no extra request. A
  regional reads as no district, which is also how a regional is recognised.
  `crop.format` and `crop.formats` override it per run or per event.
- Every manifest entry records which profile it got and why, and the crop notes
  carry the reconciliation, so a refused divider is visible afterwards rather
  than being a number that quietly differs.

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

### Added (release process)

- **Releases are cut by pushing a tag.** `.github/workflows/release.yml` runs
  the suite, scans the tree for a committed key, builds all three archives with
  `deploy/package.py`, extracts the notes from this file, proves the built
  archive runs by extracting it and running the suite from inside it, then
  creates the GitHub release with the archives and `SHA256SUMS` attached.
  Nothing is built or uploaded by hand, and the checksums in the notes are
  always the ones from the artifacts actually attached.
- `workflow_dispatch` builds and verifies a version without publishing, so a
  release can be rehearsed before its tag exists.
- `python3 deploy/package.py notes vX.Y.Z` prints a release's section from this
  file.

### Fixed

- **`run.py pull` crashed on every video it successfully downloaded.**
  `_process()` read `shard` and `shards`, which are parameters of `fetch()`,
  not of it — `NameError: name 'shards' is not defined`. The conditional
  tested `shards > 1` first, so it raised on single-worker runs too, and it
  raised at the very end: after the download, shot analysis, crop, render and
  scoreboard OCR had all completed for that video. Present since the initial
  commit and in the `Beta` release. The shard is now passed in, and
  `reprocess` preserves the one already recorded instead of erasing it.
- A test walks the symbol table of every function in `tbavid/`, `run.py` and
  `serve.py` and fails on any global that no module global defines. The crash
  above needed a real video to reach, which the suite deliberately has none of.
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
