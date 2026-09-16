# The dataset

Frames and videos never live in this repository. A harvest is ~330 MB per
match; the `harvest_good` bundle below is 1.97 GB, and a single file over
100 MB is rejected by GitHub outright. What is versioned here instead is the
**provenance**: which YouTube videos were pulled, what came out of them, and
how good each one was. That is a few hundred KB, and it is the part you cannot
regenerate — the frames themselves you can, from a TBA key and the ledger.

| | |
| --- | --- |
| `docs/provenance/seen.json` | every video ever considered, and why it was kept or skipped |
| `docs/provenance/manifest.json` | per-video analysis, crop box, shot ranges and scoreboard reads |
| the frames themselves | outside the repo — see **Getting the bundle** |

## What is in the current harvest

A snapshot as of 2026-09-14. Numbers are from `docs/provenance/`.

- **37 videos considered** across 19 events: 30 `ok`, 6 `unavailable`, 1 `failed`.
- **17 processed into frames**, from 10 events, **7,733 exported frames**.
- All 17 are qualification matches. The ledger holds six playoff pulls
  (`sf8m1`, `sf11m1`, `sf13m1`, `sf9m1`, `sf12m1`) but none of them reached
  the frame export, so this bundle has **no playoff footage**.

### Known problems

This harvest was taken **before** the competitive filter existed, and it shows.
Run `python3 run.py audit` against any harvest to get this report for yourself;
it reads the manifest, asks TBA which events of the season were real
competition, and prints what does not belong. It deletes nothing.

- **926 frames (12.0%) are not competition footage.** They come from
  `2026kylou`, the RiverBOaT Rumble — an offseason event. Two more offseason
  events, `2026miwrc` (Wolverine Robotics Competition) and `2026xxmel`
  (FRC MRT), are in the ledger but produced no frames. Pulls made after the
  filter landed cannot pick any of these; this bundle predates it.
- **`2026gal_qm62` has a bad scoreboard read.** Its blue final of 726 sits
  against a 19–285 range across every other match, with 395 score events where
  a typical match has 100–250. It carries no error flag, so all 533 of its
  frames are labelled from it.
- **Two matches were only half-read.** `2026mnmi2_qm11` and `2026mnmi_qm39`
  have a red counter and no blue one. A one-alliance read still satisfies the
  `scoreboard_ok` test in `tbavid/db.py`, so they are stored as good reads with
  `blue_fuel` null.

`python3 run.py verify` checks OCR'd totals against TBA's official score
breakdown and will catch all three.

The provenance files are committed **as harvested**, offseason entries
included. A provenance record that has been cleaned up is not a record of
anything. Absolute paths from the harvesting machine were reduced to
basenames; nothing else was changed.

## Getting the bundle

The frame bundle is distributed out of band — currently as `harvest_good.tgz`
on Google Drive, shared directly. Ask a maintainer for access; the link is
deliberately not committed here, since it is a private share and a URL in a
repository outlives whoever pasted it.

If this dataset gets a wider audience, publish it as a **GitHub Release asset**
(2 GB per file, free, and it never touches clone size) or on **Hugging Face
Datasets** (versioned and streamable), and record the URL and a `sha256` here
so the dataset is pinned without being stored.

To fold a bundle into a local harvest:

```bash
tar -xzf harvest_good.tgz
python3 run.py merge harvest_good/   # union keyed on video id; safe to re-run
python3 run.py db sync
python3 run.py audit                 # what in it is not competition footage
```

`merge` reports overlaps rather than resolving them. A non-zero
`videos_already_had` means two harvesters used the same shard.
