"""Public entry points: ``plan`` (timing only) and ``render`` (audio + report)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .audio_io import save, speech_regions
from .compose import Score, compose
from .spec import Spec, load_spec, parse_spec


def _as_spec(cue) -> Spec:
    if isinstance(cue, Spec):
        return cue
    if isinstance(cue, (str, Path)):
        return load_spec(cue)
    return parse_spec(cue)


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def report(score: Score, spec: Spec) -> dict:
    grid = score.grid
    bars = grid.bars()
    bar_of_beat = {b.beat: b.index for b in bars}
    cues = []
    for seg in grid.segments[1:]:
        t = grid.time_of(seg.beat_offset)
        cues.append({"time": _r(seg.start), "aligned_time": _r(t), "error_ms": _r((t - seg.start) * 1000, 3),
                     "kind": seg.kind, "label": seg.label, "bar": bar_of_beat.get(seg.beat_offset)})
    for c in grid.dropped:
        cues.append({"time": _r(c.time), "aligned_time": None, "error_ms": None, "kind": c.kind,
                     "label": c.label, "bar": None})
    cues.sort(key=lambda c: c["time"])

    sections = []
    for s in score.sections:
        start, end = grid.time_of(s.beat), grid.time_of(s.end_beat)
        layers: list[str] = []
        for ph in score.phrases:
            if ph.section is s:
                layers += [ly.name for ly in ph.layers if ly.name not in layers]
        if score.style.drum_min_energy <= s.energy and "drums" not in layers:
            layers.append("drums")
        sections.append({"label": s.label, "start": _r(start), "end": _r(end), "energy": _r(s.energy, 2),
                         "part": s.part, "bars": len(s.bars), "bpm": _r(60 * (s.end_beat - s.beat) / (end - start), 2),
                         "layers": layers})

    beats = grid.beat_times()
    return {
        "duration": _r(spec.duration),
        "style": score.style.name,
        "key": score.key.name,
        "seed": spec.seed,
        "base_bpm": _r(grid.base_tempo, 2),
        "ending": spec.ending,
        "final_downbeat": None if score.final_beat is None else _r(grid.time_of(score.final_beat)),
        "cues": cues,
        "sections": sections,
        "tempo_map": [{"start": _r(s.start), "end": _r(s.end), "bpm": _r(s.tempo, 3), "beats": s.beats}
                      for s in grid.segments],
        "chords": [{"time": _r(grid.time_of(b)), "chord": c.symbol} for b, _n, c in score.chords],
        "downbeats": [_r(grid.time_of(b.beat)) for b in bars],
        "beats": [_r(t) for t in beats],
        "warnings": spec.warnings + score.warnings,
    }


def plan(cue) -> dict:
    """Work out tempo, sections and the beat grid without rendering audio."""
    spec = _as_spec(cue)
    return report(compose(spec), spec)


def render(cue, out: str | Path, report_path: str | Path | None = None) -> dict:
    """Compose, synthesize and master background music for a cue sheet.

    Writes the audio to ``out`` (.wav, or .mp3/.m4a/.ogg/.flac/.opus via ffmpeg)
    and a JSON timing report next to it (``<out>.json`` unless ``report_path``
    is given). Returns the report.
    """
    from .mix import render_mix

    t0 = time.time()
    spec = _as_spec(cue)
    score = compose(spec)
    voice = speech_regions(spec.voiceover) if spec.voiceover else []
    audio, info = render_mix(score, spec, voice)
    out = save(out, audio, spec.sample_rate, seed=spec.seed)
    rep = report(score, spec)
    rep["audio"] = {"path": str(out), "sample_rate": spec.sample_rate, "samples": int(audio.shape[-1]),
                    "loudness_lufs": _r(info.loudness_lufs, 2), "true_peak_dbtp": _r(info.true_peak_db, 2),
                    "ducked_regions": [[_r(a, 3), _r(b, 3)] for a, b in info.duck_regions],
                    "render_seconds": _r(time.time() - t0, 2)}
    rp = Path(report_path) if report_path else Path(str(out) + ".json")
    rp.write_text(json.dumps(rep, indent=2))
    rep["report_path"] = str(rp)
    return rep
