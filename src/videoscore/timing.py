"""Fit a beat grid to video cue points.

Every cue (scene transition or hit) becomes a segment boundary that lands
*exactly* on a downbeat. Each segment gets its own tempo, chosen so that its
length is a whole number of beats, and neighbouring segments are kept within a
few percent of each other. Tempo changes that small, made at the same moment
the arrangement changes, can't be heard. The cut lands on the beat to the
sample.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

BEATS_PER_BAR = 4
MAX_DEVIATION = 0.08  # tempo changes larger than this between segments get a warning
HALF_BAR_PENALTY = 8.0  # cost of ending a segment on a 2-beat bar (= a 4% tempo change)
LAST_HALF_BAR_PENALTY = 30.0  # the final bar rings out, so it should be a full bar


@dataclass
class Cue:
    time: float
    kind: str  # "section" | "hit"
    label: str = ""


@dataclass
class Segment:
    start: float
    end: float
    beats: int
    beat_offset: int  # global beat number at which this segment starts
    kind: str  # kind of cue that opens the segment: "start" | "section" | "hit"
    label: str = ""

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def beat_len(self) -> float:
        return self.length / self.beats

    @property
    def tempo(self) -> float:
        return 60.0 / self.beat_len


@dataclass
class Bar:
    index: int
    beat: int  # global beat number of the downbeat
    beats: int  # 4, or 2 for a half bar at the end of a segment
    segment: int


@dataclass
class Grid:
    base_tempo: float
    segments: list[Segment]
    dropped: list[Cue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_beats(self) -> int:
        last = self.segments[-1]
        return last.beat_offset + last.beats

    @property
    def end(self) -> float:
        return self.segments[-1].end

    def _segment_for_beat(self, beat: float) -> Segment:
        offsets = [s.beat_offset for s in self.segments]
        i = bisect.bisect_right(offsets, beat) - 1
        return self.segments[max(0, min(i, len(self.segments) - 1))]

    def time_of(self, beat: float) -> float:
        """Seconds at a (fractional) global beat position. Extrapolates past the end."""
        seg = self._segment_for_beat(beat)
        return seg.start + (beat - seg.beat_offset) * seg.beat_len

    def times_of(self, beats: np.ndarray) -> np.ndarray:
        beats = np.asarray(beats, dtype=float)
        offsets = np.array([s.beat_offset for s in self.segments])
        idx = np.clip(np.searchsorted(offsets, beats, side="right") - 1, 0, len(self.segments) - 1)
        starts = np.array([s.start for s in self.segments])[idx]
        lens = np.array([s.beat_len for s in self.segments])[idx]
        return starts + (beats - offsets[idx]) * lens

    def bars(self) -> list[Bar]:
        bars: list[Bar] = []
        for si, seg in enumerate(self.segments):
            b = 0
            while b < seg.beats:
                n = min(BEATS_PER_BAR, seg.beats - b)
                bars.append(Bar(index=len(bars), beat=seg.beat_offset + b, beats=n, segment=si))
                b += n
        return bars

    def beat_times(self) -> list[float]:
        return [float(t) for t in self.times_of(np.arange(self.total_beats))]


def _drop_close_cues(cues: list[Cue], end: float, min_len: float) -> tuple[list[Cue], list[Cue]]:
    """Remove cues that would make a segment shorter than one bar."""
    kept = sorted(cues, key=lambda c: c.time)
    dropped: list[Cue] = []
    while True:
        points = [0.0] + [c.time for c in kept] + [end]
        lengths = [b - a for a, b in zip(points, points[1:])]
        j = min(range(len(lengths)), key=lambda k: lengths[k])
        if lengths[j] >= min_len or not kept:
            return kept, dropped
        # Segment j runs from points[j] to points[j+1]; cue indices are shifted by one.
        candidates = [k for k in (j, j - 1) if 0 <= k < len(kept)]  # later cue first
        hit_first = [k for k in candidates if kept[k].kind == "hit"]
        k = (hit_first or candidates)[0]
        dropped.append(kept.pop(k))


def _beat_options(length: float, lo: float, hi: float) -> np.ndarray:
    """Even beat counts (>= one bar) whose tempo is near the allowed range."""
    b_lo = max(BEATS_PER_BAR, int(np.floor(length * lo * (1 - MAX_DEVIATION) / 60 / 2)) * 2)
    b_hi = max(b_lo, int(np.ceil(length * hi * (1 + MAX_DEVIATION) / 60 / 2)) * 2)
    return np.arange(b_lo, b_hi + 1, 2)


def fit_grid(
    cues: list[Cue],
    end: float,
    tempo_range: tuple[float, float],
    preferred: float,
) -> Grid:
    """Choose a whole number of beats for every segment so each cue is on a downbeat.

    Dynamic programming over the beat-count options of each segment minimises
    tempo *jumps* between neighbouring segments (what the ear notices), keeps
    every tempo inside the style's range, prefers full bars, and pulls gently
    toward the preferred tempo.
    """
    lo, hi = tempo_range
    min_len = 60.0 * BEATS_PER_BAR / (hi * (1 + MAX_DEVIATION))
    kept, dropped = _drop_close_cues(cues, end, min_len)

    points = [0.0] + [c.time for c in kept] + [end]
    lengths = np.diff(points)
    kinds = ["start"] + [c.kind for c in kept]

    def node_cost(k: int, beats: np.ndarray) -> np.ndarray:
        tempo = 60.0 * beats / lengths[k]
        half = LAST_HALF_BAR_PENALTY if k == len(lengths) - 1 else HALF_BAR_PENALTY
        out_of_range = np.maximum(0, np.log(lo / tempo)) + np.maximum(0, np.log(tempo / hi))
        return (np.where(beats % 4 != 0, half, 0.0)
                + 2.0 * (100 * out_of_range) ** 2
                + 0.02 * (100 * np.log(tempo / preferred)) ** 2)

    options = [_beat_options(L, lo, hi) for L in lengths]
    cost = node_cost(0, options[0])
    back: list[np.ndarray] = []
    for k in range(1, len(lengths)):
        prev_t = 60.0 * options[k - 1] / lengths[k - 1]
        cur_t = 60.0 * options[k] / lengths[k]
        # a jump hidden under a scene change is less noticeable than one under a hit
        weight = 0.5 if kinds[k] == "section" else 1.0
        jump = weight * (100 * np.log(cur_t[:, None] / prev_t[None, :])) ** 2
        total = jump + cost[None, :]
        back.append(np.argmin(total, axis=1))
        cost = total[np.arange(len(cur_t)), back[-1]] + node_cost(k, options[k])
    idx = [int(np.argmin(cost))]
    for k in range(len(lengths) - 1, 0, -1):
        idx.append(int(back[k - 1][idx[-1]]))
    idx.reverse()

    segments: list[Segment] = []
    offset = 0
    for k, L in enumerate(lengths):
        beats = int(options[k][idx[k]])
        label = "" if k == 0 else kept[k - 1].label
        segments.append(Segment(start=points[k], end=points[k + 1], beats=beats, beat_offset=offset,
                                kind=kinds[k], label=label))
        offset += beats

    total_beats = sum(s.beats for s in segments)
    base_tempo = 60.0 * total_beats / end
    grid = Grid(base_tempo=base_tempo, segments=segments, dropped=dropped)
    for c in dropped:
        grid.warnings.append(
            f"Cue at {c.time:.3f}s ({c.kind}) was not synced: it is less than one bar "
            f"({min_len:.2f}s) from a neighbouring cue."
        )
    for a, b in zip(segments, segments[1:]):
        change = abs(math.log(b.tempo / a.tempo))
        if change > MAX_DEVIATION:
            grid.warnings.append(
                f"Tempo changes {100 * change:.0f}% at {b.start:.2f}s ({a.tempo:.1f} -> {b.tempo:.1f} BPM); "
                "this may be noticeable. Moving that cue a little would avoid it."
            )
    return grid
