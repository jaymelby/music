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
