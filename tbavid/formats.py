"""Which broadcast layout this is, and what its overlay is shaped like.

`crop.py` finds the overlay bands from the pixels alone, and that is the right
default: it needs no list of events to be kept up to date, and it was already
correct on feeds nobody had looked at. What it cannot do is know when it is
wrong. A measurement has no opinion about whether the answer is plausible, so
the two ways it fails are both silent:

  * **A band that isn't there.** `detect_bottom` hunts for a split-screen
    divider in the lower half of every frame. On a single-camera feed the
    strongest horizontal edge down there belongs to the field -- the guardrail,
    the scoring table, the front of the bleachers -- and if it clears the
    threshold, half the field is cropped away and the frames still look
    plausible.
  * **A band that is there and reads shallow.** The banner walk stops at the
    first row that moves. A layout that leaves a gap between the score banner
    and the field stops the walk early and keeps a strip of scoreboard in the
    training set, which is exactly the thing a detector will learn to key off.

A format profile is the missing opinion. It says what this feed's layout is
*supposed* to look like, so a measurement can be checked against it instead of
being taken on faith.

**The measurement still wins.** A profile is not a set of coordinates to apply
-- that is what `crop.overrides` is for, and it needs a human to read numbers
off a frame for every event. A profile only ever does three things:

  1. tunes the detector's thresholds for this layout before it runs;
  2. supplies a fallback when detection comes back inconclusive, in place of
     the one global `crop.top` that has to serve every feed at once;
  3. vetoes a band its layout does not have -- the one case above where the
     pixels are genuinely misleading and a measurement should not be trusted.

Which profile a video gets is decided by TBA, not by a filename. The event list
we already fetch to pick videos (`/events/{year}/simple`) carries each event's
`district` object, so the district an event belongs to costs no extra request
and is known before the video is downloaded.

## Provenance

Every profile carries `provenance`, and it is load-bearing rather than
documentation:

  * `measured` -- the numbers come from running this pipeline over footage of
    this feed, and the values in the profile are what it read.
  * `declared` -- the numbers describe the layout as specified, and no footage
    has been through `run.py formats calibrate` yet. A declared profile still
    tunes and still vetoes, but its fallback fractions are a starting point.

`run.py formats calibrate --event <key>` measures a real download and prints
the profile those measurements imply, which is how a `declared` profile is
meant to become a `measured` one. Nothing here silently promotes itself.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# How much a measurement may miss the profile's expected band by before it is
# worth saying so out loud. Well inside the expansion margins in crop.py, so a
# note here means a real disagreement and not a rounding difference.
BAND_TOLERANCE = 0.03

# TBA's event_type codes, in the same order `tba.COMPETITIVE_EVENT_TYPES`
# lists them. Only needed here for the labels in `describe`, and to name the
# distinction CA_DISTRICT's gate rests on: a district's weekend events (1) and
# that district's own state championship (2) are the same district and not the
# same broadcast.
EVENT_TYPE_NAMES = {
    0: "regional",
    1: "district event",
    2: "district championship",
    3: "championship division",
    4: "championship final",
    5: "district championship division",
    6: "Festival of Champions",
}
DISTRICT_EVENT = 1


class Format:
    """One broadcast layout.

    `banner` and `split` are each (low, high, fallback) in fractions of frame
    height: the range a measurement is expected to land in, and the number to
    use when there is no measurement at all. A `split` of None means the layout
    has no second camera panel and any divider found in one is a false positive.
    """

    def __init__(self, name: str, label: str, provenance: str, notes: str,
                 banner: Tuple[float, float, float],
                 split: Optional[Tuple[float, float, float]] = None,
                 districts: Tuple[str, ...] = (),
                 event_types: Tuple[int, ...] = (),
                 event_re: str = "", title_re: str = "",
                 tune: Optional[Dict[str, float]] = None):
        self.name = name
        self.label = label
        self.provenance = provenance
        self.notes = notes
        self.banner = banner
        self.split = split
        self.districts = tuple(d.lower() for d in districts)
        # A gate, not a selector: a profile with this set can only ever match
        # an event of one of these TBA event_types. A district's own state
        # championship belongs to the same district as its weekend events and
        # is not the same broadcast -- see CA_DISTRICT.
        self.event_types = tuple(event_types)
        self.event_re = re.compile(event_re, re.I) if event_re else None
        self.title_re = re.compile(title_re, re.I) if title_re else None
        self.tune = dict(tune or {})

    @property
    def splits(self) -> bool:
        """Does this layout carry a permanent second-camera panel?"""
        return self.split is not None

    def matches(self, district: str = "", event_key: str = "", title: str = "",
                event_type: Optional[int] = None) -> str:
        """Why this profile fits, or "" if it does not.

        The string is the reason, so a note in the manifest says what decided
        the profile rather than only which one won.

        An `event_types` gate that cannot be checked -- `event_type` unknown,
        as it is for a manifest entry harvested before it was recorded -- fails
        the gate rather than being waived. A profile's whole purpose is to
        veto, and a veto applied to a broadcast nobody has established it
        describes is the same mistake in the other direction.
        """
        if self.event_types and event_type not in self.event_types:
            return ""
        if district and district.lower() in self.districts:
            return f"TBA district {district.lower()}"
        if self.event_re and event_key and self.event_re.search(event_key):
            return f"event key matches /{self.event_re.pattern}/"
        if self.title_re and title and self.title_re.search(title):
            return f"video title matches /{self.title_re.pattern}/"
        return ""

    def tuning(self) -> Dict[str, float]:
        """Threshold overrides for `crop`, including the split-mode decision."""
        out = dict(self.tune)
        out.setdefault("split_mode", "expected" if self.splits else "unlikely")
        return out

    def describe(self) -> str:
        blo, bhi, bfb = self.banner
        line = (f"{self.name:<14} {self.label}\n"
                f"{'':<14} provenance: {self.provenance}\n"
                f"{'':<14} banner {blo:.2f}-{bhi:.2f} (fallback {bfb:.2f})\n")
        if self.splits:
            slo, shi, sfb = self.split
            line += f"{'':<14} split  {slo:.2f}-{shi:.2f} (fallback {sfb:.2f})\n"
        else:
            line += f"{'':<14} split  none expected\n"
        if self.districts:
            line += f"{'':<14} districts: {', '.join(self.districts)}\n"
        if self.event_types:
            line += (f"{'':<14} event types: "
                     f"{', '.join(str(t) for t in self.event_types)}"
                     f" ({EVENT_TYPE_NAMES.get(self.event_types[0], '?')}"
                     f"{', ...' if len(self.event_types) > 1 else ''})\n")
        return line


# -- the profiles ----------------------------------------------------------
#
# Ordered: the first profile whose selectors match wins, and `generic` matches
# nothing on purpose so it can only ever be reached as the fallback.

CA_DISTRICT = Format(
    name="ca_district",
    label="California district weekend event feed",
    provenance="declared",
    notes=(
        "FIRST California's district weekend events -- Central Valley, San "
        "Francisco, Los Angeles, Ventura County, Orange County, Aerospace "
        "Valley. One elevated field camera, a score banner across the top, "
        "and no permanent second-camera panel: the director cuts away to a "
        "replay or a pit shot rather than showing both at once, which the shot "
        "classifier already handles by dropping those shots. So the thing this "
        "profile is really for is the veto: without it, the strongest static "
        "horizontal edge in the lower half of a single-camera frame is the "
        "guardrail or the front row of the bleachers, and a feed with a clean "
        "one loses the bottom of its own field to a divider that was never "
        "there. "
        "NOT one production, though: most of these events are YouTube webcasts "
        "carried by the FIRST Webcast Unit, and at least one (Central Valley) "
        "goes out on Twitch instead. Those are different rigs, so the banner "
        "range here is wide and calibrating per event key is worth doing -- "
        "`crop.formats` takes one event at a time for exactly this."
    ),
    # Declared, not measured. Wide on purpose: a range this size still catches
    # a banner walk that ran away or stopped in the gap above the digits, which
    # is what the range is for, without pretending to a precision no footage
    # has been read for yet -- and it has to cover two different productions.
    banner=(0.10, 0.22, 0.16),
    split=None,
    districts=("ca",),
    # Weekend district events only. A district's own state championship
    # (2026cancmp, and its southern counterpart) carries district "ca" and is a
    # different and larger production, on which this profile's no-split veto
    # would be an assertion nobody has checked -- and vetoing a divider that IS
    # there keeps a whole side-camera panel in the training set, which is the
    # failure this file exists to prevent, pointed the other way. Those events
    # fall through to `generic`, which has no opinion, until somebody measures
    # one.
    event_types=(DISTRICT_EVENT,),
    tune={
        # A divider this layout does not have needs more than one strong row to
        # be believed -- see `split_mode: unlikely` in crop.detect_bottom.
        "edge_min_abs": 90.0,
        "edge_min_ratio": 6.0,
    },
)

CHAMPS_SPLIT = Format(
    name="champs_split",
    label="2026 Championship split-screen feed",
    provenance="measured",
    notes=(
        "The elevated field camera on top and a low side camera in the bottom "
        "third for the entire match, separated by a static textured divider. "
        "The side camera is a region of every frame rather than a shot to cut "
        "away from, so the crop is the only thing that can remove it. The "
        "divider's edge strength measured 74 against a lower half typical of "
        "well under 20."
    ),
    banner=(0.10, 0.22, 0.16),
    split=(0.28, 0.40, 0.33),
    event_re=r"^\d{4}(cmptx|cmpmi|gal|hop|new|arc|cars|cur|dal|dar|tes|joh|mil|tur|carv)$",
    title_re=r"championship|einstein|festival of champions",
)

GENERIC = Format(
    name="generic",
    label="unknown layout, detection only",
    provenance="measured",
    notes=(
        "What every feed got before profiles existed, and still the right "
        "answer for one nobody has characterised: detect both bands from the "
        "pixels, refuse a result past the sanity ceilings, and fall back to "
        "config.json. It has no opinion, so it cannot veto -- a split found "
        "here is taken at face value."
    ),
    banner=(0.0, 0.35, 0.0),
    split=(0.0, 0.50, 0.0),
)

FORMATS: Tuple[Format, ...] = (CA_DISTRICT, CHAMPS_SPLIT, GENERIC)
BY_NAME: Dict[str, Format] = {f.name: f for f in FORMATS}


def select(district: str = "", event_key: str = "", title: str = "",
           cfg: Optional[dict] = None,
           event_type: Optional[int] = None) -> Tuple[Format, str]:
    """Pick a profile for one video. Returns (format, why).

    Config beats TBA beats nothing, because the two config routes are a human
    saying they have looked at the footage and this is the layout:

        "crop": {
          "format": "ca_district",              # every video in this run
          "formats": {"2026casj": "generic"}    # or one event at a time
        }
    """
    conf = (cfg or {}).get("crop") or {}

    per_event = (conf.get("formats") or {}).get(event_key)
    if per_event:
        fmt = BY_NAME.get(per_event)
        if not fmt:
            raise SystemExit(
                f"config.json crop.formats[{event_key}] = {per_event!r} is not a "
                f"known format. Known: {', '.join(sorted(BY_NAME))}")
        return fmt, f"crop.formats[{event_key}]"

    forced = (conf.get("format") or "auto").strip()
    if forced and forced != "auto":
        fmt = BY_NAME.get(forced)
        if not fmt:
            raise SystemExit(
                f"config.json crop.format = {forced!r} is not a known format. "
                f"Known: auto, {', '.join(sorted(BY_NAME))}")
        return fmt, "crop.format"

    for fmt in FORMATS:
        why = fmt.matches(district=district, event_key=event_key, title=title,
                          event_type=event_type)
        if why:
            return fmt, why
    return GENERIC, "no profile matched"


def district_of(event: dict) -> str:
    """A TBA event's district abbreviation, or "" for a regional.

    `/events/{year}/simple` carries `district` as an object for a district
    event and `null` for everything else, so this is also the regional test.
    """
    district = (event or {}).get("district")
    if not isinstance(district, dict):
        return ""
    return (district.get("abbreviation") or "").strip().lower()


def reconcile(fmt: Format, top_frac: Optional[float], bottom_frac: Optional[float]
              ) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Check two measurements against a profile. Returns (top, bottom, notes).

    Three outcomes per band, and only one of them changes the number:

      * inside the expected range -- taken as measured, and said so, because a
        profile agreeing with the pixels is worth having in the manifest;
      * outside it -- still taken as measured, with the disagreement noted. The
        pixels are the thing being cropped and the profile is a description of
        a layout that may have been re-cut between events;
      * a split on a layout that has none -- dropped. This is the one case
        where the measurement is answering a different question than the one
        asked: there is no divider to find, so whatever cleared the threshold
        is field content.

    `None` in means detection was inconclusive; the profile's fallback goes in
    its place, which is the second thing profiles are for.
    """
    notes: List[str] = []

    blo, bhi, bfb = fmt.banner
    if top_frac is None:
        if bfb > 0:
            top_frac = bfb
            notes.append(f"format {fmt.name}: banner fallback top={bfb:.3f}")
    elif not (blo - BAND_TOLERANCE <= top_frac <= bhi + BAND_TOLERANCE):
        notes.append(f"format {fmt.name}: measured top={top_frac:.3f} is outside "
                     f"the expected {blo:.2f}-{bhi:.2f}; keeping the measurement")
    else:
        notes.append(f"format {fmt.name}: measured top={top_frac:.3f} matches the "
                     f"expected {blo:.2f}-{bhi:.2f}")

    if not fmt.splits:
        if bottom_frac:
            notes.append(f"format {fmt.name}: a split divider was found at "
                         f"bottom={bottom_frac:.3f}, but this layout has no "
                         f"second-camera panel -- ignoring it as field content")
        bottom_frac = 0.0
        return top_frac, bottom_frac, notes

    slo, shi, sfb = fmt.split
    if bottom_frac is None:
        if sfb > 0:
            bottom_frac = sfb
            notes.append(f"format {fmt.name}: split fallback bottom={sfb:.3f}")
    elif bottom_frac > 0 and not (slo - BAND_TOLERANCE <= bottom_frac <= shi + BAND_TOLERANCE):
        notes.append(f"format {fmt.name}: measured bottom={bottom_frac:.3f} is "
                     f"outside the expected {slo:.2f}-{shi:.2f}; keeping the "
                     f"measurement")
    return top_frac, bottom_frac, notes
