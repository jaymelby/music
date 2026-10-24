import pytest

from videoscore.compose import compose
from videoscore.spec import parse_spec
from videoscore.styles import STYLES

CUE = {"duration": 70, "scenes": [{"start": 0}, {"start": 9.4, "energy": 0.45}, {"start": 25.0, "energy": "high"},
                                  {"start": 48.6, "energy": 0.7}, {"start": 61.0, "energy": 0.35}], "hits": [36.0]}


@pytest.mark.parametrize("style", sorted(STYLES))
def test_every_style_composes_in_key_and_in_range(style):
    score = compose(parse_spec({**CUE, "style": style, "seed": 3}))
    layers = {ly.name: ly for ly in score.style.layers}
    assert score.notes and score.chords
    out_of_key = 0
    for n in score.notes:
        ly = layers[n.layer]
        lo, hi = ly.range
        assert lo - 12 <= n.pitch <= hi + 12, (n.layer, n.pitch)
        assert n.length > 0 and 0 < n.vel <= 1
        chord = next(c for b, nb, c in score.chords if b <= n.beat < b + nb + 1e-9) if n.beat < score.grid.total_beats else None
        if chord and n.pitch % 12 not in chord.pcs and n.pitch % 12 not in score.key.scale:
            out_of_key += 1
    assert out_of_key / len(score.notes) < 0.02


@pytest.mark.parametrize("style", sorted(STYLES))
def test_melody_strong_beats_are_chord_tones(style):
    score = compose(parse_spec({**CUE, "style": style, "seed": 5}))
    melodic = {ly.name for ly in score.style.layers if ly.role == "melody"}
    bars = score.grid.bars()
    checked = 0
    for n in score.notes:
        if n.layer not in melodic:
            continue
        bar = next(b for b in bars if b.beat <= n.beat < b.beat + b.beats)
        pos = n.beat - bar.beat  # beats 1 and 3 of the bar are the strong beats
        if abs(pos - round(pos)) < 1e-6 and round(pos) % 2 == 0:
            chord = next(c for b, nb, c in score.chords if b <= n.beat < b + nb)
            assert n.pitch % 12 in chord.pcs
            checked += 1
    assert checked > 0


def test_same_seed_same_music_different_seed_different_music():
    a = compose(parse_spec({**CUE, "seed": 11}))
    b = compose(parse_spec({**CUE, "seed": 11}))
    c = compose(parse_spec({**CUE, "seed": 12}))
    sig = lambda s: [(n.layer, n.beat, n.pitch) for n in s.notes]
    assert sig(a) == sig(b)
    assert sig(a) != sig(c)


def test_transitions_are_marked_and_ending_resolves():
    score = compose(parse_spec({**CUE, "style": "corporate", "seed": 2}))
    boundary_beats = {s.beat for s in score.sections[1:]}
    fx_beats = {f.beat for f in score.fx}
    assert boundary_beats <= fx_beats  # every scene change gets a crash, swell or riser
    final = score.chords[-1][2]
    assert final.root == score.key.tonic
    assert score.final_beat == score.chords[-1][0]


@pytest.mark.parametrize("style, key", [("lofi", "A lydian"), ("corporate", "D dorian"), ("ambient", "E mixolydian")])
def test_unsupported_mode_falls_back_to_diatonic_harmony(style, key):
    score = compose(parse_spec({"duration": 40, "style": style, "key": key}))
    assert score.key.mode in ("major", "minor")
    assert any("no " in w and "progressions" in w for w in score.warnings)
    allowed = set(score.key.scale)
    if score.key.mode == "minor":
        allowed.add((score.key.tonic + 11) % 12)  # harmonic-minor leading tone in V
    for _beat, _n, chord in score.chords:
        assert set(chord.pcs) <= allowed, chord.symbol


def test_long_single_scene_alternates_verse_and_chorus():
    score = compose(parse_spec({"duration": 150, "style": "electronic", "seed": 1}))
    assert {p.part for p in score.phrases} == {"A", "B"}


def test_one_bar_end_card_is_not_doubled():
    score = compose(parse_spec({"duration": 30, "transitions": [27.6]}))
    fb = score.final_beat
    final = [n for n in score.notes if n.beat >= fb]
    assert len(final) == len({(n.layer, n.pitch) for n in final})  # one ring-out chord, no repeats
    assert all(h.beat == fb for h in score.hits if h.beat >= fb)  # only the final hit, no pattern


@pytest.mark.parametrize("seed", range(6))
def test_melody_snaps_after_short_bars(seed):
    """Short (3-beat) bars must not shift which melody notes count as strong beats."""
    import random

    rng = random.Random(seed)
    hits = sorted(round(rng.uniform(8, 32), 2) for _ in range(4))
    score = compose(parse_spec({"duration": 40, "style": "lofi", "seed": 21, "hits": hits,
                                "scenes": [{"start": 0, "energy": 0.8}]}))
    melodic = {ly.name for ly in score.style.layers if ly.role == "melody"}
    bars = score.grid.bars()
    for n in score.notes:
        if n.layer not in melodic:
            continue
        bar = next(b for b in bars if b.beat <= n.beat < b.beat + b.beats)
        pos = n.beat - bar.beat
        if abs(pos - round(pos)) < 1e-6 and round(pos) % 2 == 0:
            chord = next(c for b, nb, c in score.chords if b <= n.beat < b + nb)
            assert n.pitch % 12 in chord.pcs, (n.beat, n.pitch, chord.symbol)
