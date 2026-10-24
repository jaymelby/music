import math
import random

import pytest

from videoscore.timing import BEATS_PER_BAR, Cue, fit_grid


def _check_grid(grid, cues, end):
    kept = [c for c in cues if c not in grid.dropped]
    starts = [s.start for s in grid.segments[1:]]
    assert starts == pytest.approx([c.time for c in sorted(kept, key=lambda c: c.time)], abs=1e-12)
    for seg in grid.segments:
        assert seg.beats >= BEATS_PER_BAR and seg.beats % 2 == 0
        # the cue lands exactly on the first beat of its segment
        assert grid.time_of(seg.beat_offset) == pytest.approx(seg.start, abs=1e-9)
    assert grid.time_of(grid.total_beats) == pytest.approx(end, abs=1e-9)


@pytest.mark.parametrize("seed", range(25))
def test_random_cues_land_exactly_on_downbeats(seed):
    rng = random.Random(seed)
    end = rng.uniform(30, 300)
    times = sorted(rng.uniform(3, end - 3) for _ in range(rng.randint(0, 12)))
    cues = [Cue(t, rng.choice(["section", "section", "hit"])) for t in times]
    grid = fit_grid(cues, end, (100, 124), 112)
    _check_grid(grid, cues, end)
    downbeats = {round(grid.time_of(b.beat), 9) for b in grid.bars()}
    for c in cues:
        if c not in grid.dropped:
            assert round(c.time, 9) in downbeats


def test_well_spaced_cues_keep_tempo_steady():
    cues = [Cue(t, "section") for t in (8.0, 19.5, 33.0, 47.25, 60.0)]
    grid = fit_grid(cues, 75.0, (100, 124), 112)
    tempos = [s.tempo for s in grid.segments]
    for a, b in zip(tempos, tempos[1:]):
        assert abs(math.log(b / a)) < 0.06
    assert all(96 <= t <= 130 for t in tempos)
    assert not grid.warnings


def test_single_segment_uses_whole_bars_near_preferred_tempo():
    grid = fit_grid([], 60.0, (100, 124), 112)
    (seg,) = grid.segments
    assert seg.beats % BEATS_PER_BAR == 0
    assert seg.tempo == pytest.approx(112, abs=2)


def test_cues_closer_than_a_bar_are_dropped_with_warning():
    cues = [Cue(10.0, "section"), Cue(10.5, "hit"), Cue(30.0, "section")]
    grid = fit_grid(cues, 45.0, (100, 124), 112)
    assert [c.time for c in grid.dropped] == [10.5]  # the hit goes first
    assert any("not synced" in w for w in grid.warnings)
    _check_grid(grid, cues, 45.0)


def test_times_of_matches_time_of():
    grid = fit_grid([Cue(7.3, "section"), Cue(21.9, "section")], 40.0, (100, 124), 112)
    import numpy as np

    beats = np.linspace(0, grid.total_beats + 3, 97)
    assert grid.times_of(beats) == pytest.approx([grid.time_of(b) for b in beats])
