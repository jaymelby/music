import pytest

from videoscore.spec import SpecError, parse_spec


def test_minimal_spec_defaults():
    spec = parse_spec({"duration": 30})
    assert spec.style == "corporate"
    assert [s.start for s in spec.scenes] == [0.0]
    assert spec.ending == "resolve"


def test_transitions_shorthand_and_energy_words():
    spec = parse_spec({"duration": 40, "transitions": [10, 20]})
    assert [s.start for s in spec.scenes] == [0, 10, 20]
    spec = parse_spec({"duration": 40, "scenes": [{"start": 0, "energy": "low"}, {"start": 5, "energy": "high"}]})
    assert [s.energy for s in spec.scenes] == [0.25, 0.8]


def test_scene_durations_build_the_timeline():
    spec = parse_spec({"scenes": [{"duration": 5}, {"duration": 7.5}, {"duration": 10}]})
    assert [s.start for s in spec.scenes] == [0, 5, 12.5]
    assert spec.duration == 22.5


def test_frames_unit():
    spec = parse_spec({"unit": "frames", "fps": 30, "duration": 900, "transitions": [150, 451]})
    assert spec.duration == 30
    assert [s.start for s in spec.scenes] == pytest.approx([0, 5, 451 / 30])


def test_missing_first_scene_is_added():
    spec = parse_spec({"duration": 30, "scenes": [{"start": 4, "label": "b"}]})
    assert [s.start for s in spec.scenes] == [0, 4]


@pytest.mark.parametrize("data, message", [
    ({"duration": 30, "transitons": [3]}, "Did you mean 'transitions'"),
    ({"duration": 2}, "between"),
    ({"duration": 30, "style": "corprate"}, "Did you mean 'corporate'"),
    ({"duration": 30, "transitions": [31]}, "outside the video"),
    ({"duration": 30, "scenes": [{"start": 0, "energy": 3}]}, "between 0 and 1"),
    ({"duration": 30, "key": "H major"}, "Can't parse key"),
    ({"transitions": [3]}, "'duration'"),
    ({"duration": 30, "duck": [[5, 2]]}, "ends before it starts"),
])
def test_helpful_errors(data, message):
    with pytest.raises(SpecError, match=message):
        parse_spec(data)


@pytest.mark.parametrize("data, message", [
    ({"duration": 30, "seed": -1}, "non-negative"),
    ({"duration": 30, "hits": 24}, "must be a list"),
    ({"duration": 30, "duck": 5}, "must be a list"),
    ({"duration": 30, "voiceover": ""}, "path to an audio file"),
    ({"duration": 30, "loudness": -8}, "between -40 and -12"),
    ({"duration": float("nan")}, "finite"),
    ({"duration": 30, "transitions": [float("nan")]}, "finite"),
])
def test_bad_values_are_spec_errors(data, message):
    with pytest.raises(SpecError, match=message):
        parse_spec(data)


def test_duck_regions_outside_the_video_are_ignored():
    spec = parse_spec({"duration": 30, "duck": [[-5, -1], [40, 50], [28, 35]]})
    assert spec.duck == [(28, 30)]
    assert sum("outside the video" in w for w in spec.warnings) == 2


def test_hit_at_zero_is_accepted():
    assert parse_spec({"duration": 30, "hits": [0, 12]}).hits == [12]


@pytest.mark.parametrize("text, name", [("A Minor", "A minor"), ("C Major", "C major"), ("Am", "A minor"),
                                        ("AM", "A major"), ("Bb", "Bb major"), ("f# DORIAN", "F# dorian")])
def test_key_names(text, name):
    from videoscore.theory import parse_key

    assert parse_key(text).name == name
