import hashlib
import json
import shutil
import wave

import numpy as np
import pytest

from videoscore import render
from videoscore.cli import main
from videoscore.dsp import lufs, true_peak_db


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
    return x.reshape(-1, ch).T, sr


CUE = {"duration": 31.5, "style": "corporate", "seed": 4,
       "scenes": [{"start": 0}, {"start": 6.2}, {"start": 17.8, "energy": "high"}]}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = tmp_path_factory.mktemp("r") / "m.wav"
    rep = render(CUE, out)
    return out, rep


def test_length_loudness_and_peak(rendered):
    out, rep = rendered
    x, sr = read_wav(out)
    assert sr == 48000
    assert x.shape == (2, round(31.5 * 48000))
    assert abs(lufs(x, sr) - (-18)) < 0.5
    assert true_peak_db(x) <= -0.9
    assert np.all(np.isfinite(x))
    assert np.abs(x[:, -200:]).max() < 1e-3  # ends in silence, no click
    assert rep["audio"]["samples"] == x.shape[1]


def test_report_lists_synced_cues(rendered):
    out, rep = rendered
    on_disk = json.loads(open(str(out) + ".json").read())
    assert on_disk["cues"] == rep["cues"]
    assert [c["time"] for c in rep["cues"]] == [6.2, 17.8]
    assert all(abs(c["error_ms"]) < 0.001 for c in rep["cues"])
    assert set(c["time"] for c in rep["cues"]) <= set(rep["downbeats"])


def test_transition_is_audible_at_the_cut(rendered):
    out, rep = rendered
    x, sr = read_wav(out)
    mono = x.mean(axis=0)
    t = 17.8
    before = np.sqrt(np.mean(mono[int((t - 2.0) * sr):int((t - 0.2) * sr)] ** 2))
    after = np.sqrt(np.mean(mono[int(t * sr):int((t + 1.8) * sr)] ** 2))
    assert after > before * 1.2  # the high-energy section starts on the cut


def test_deterministic(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    spec = {"duration": 12, "style": "lofi", "seed": 9}
    render(spec, a)
    render(spec, b)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_ducking_lowers_music_under_narration(tmp_path):
    base = {"duration": 20, "style": "corporate", "seed": 1, "scenes": [{"start": 0, "energy": 0.6}]}
    render(base, tmp_path / "plain.wav")
    render({**base, "duck": [[8, 14]], "duck_db": -10}, tmp_path / "duck.wav")
    p, sr = read_wav(tmp_path / "plain.wav")
    d, _ = read_wav(tmp_path / "duck.wav")
    seg = slice(int(9 * sr), int(13 * sr))
    ratio = np.sqrt(np.mean(d[:, seg] ** 2) / np.mean(p[:, seg] ** 2))
    assert 20 * np.log10(ratio) == pytest.approx(-10, abs=1.0)
    outside = slice(int(1 * sr), int(6 * sr))
    assert np.sqrt(np.mean(d[:, outside] ** 2) / np.mean(p[:, outside] ** 2)) == pytest.approx(1, abs=0.02)


def test_voiceover_file_is_detected_and_ducked(tmp_path):
    sr = 16000
    t = np.arange(10 * sr) / sr
    speech = np.where((t > 3) & (t < 6), 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 4 * t)), 0)
    vo = tmp_path / "vo.wav"
    with wave.open(str(vo), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((speech * 32767).astype("<i2").tobytes())
    rep = render({"duration": 10, "seed": 2, "voiceover": str(vo)}, tmp_path / "m.wav")
    (a, b), = rep["audio"]["ducked_regions"]
    assert a == pytest.approx(3, abs=0.1) and b == pytest.approx(6, abs=0.1)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_mp3_output(tmp_path):
    out = tmp_path / "m.mp3"
    render({"duration": 8, "style": "ambient"}, out)
    assert out.stat().st_size > 10_000


def test_cli_plan_and_errors(tmp_path, capsys):
    cue = tmp_path / "cue.json"
    cue.write_text(json.dumps({"duration": 20, "transitions": [5, 12]}))
    assert main(["plan", str(cue)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [c["time"] for c in plan["cues"]] == [5, 12]
    assert main(["plan", "--duration", "20", "--transitions", "5,12", "--style", "lofi"]) == 0
    capsys.readouterr()
    cue.write_text(json.dumps({"duration": 20, "transitons": [5]}))
    assert main(["plan", str(cue)]) == 2
    assert "Did you mean 'transitions'" in capsys.readouterr().err
