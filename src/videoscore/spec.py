"""Cue-sheet parsing and validation.

A cue sheet describes the video the music is for: its total length, where the
scene transitions are, and (optionally) how intense each scene should feel.
Errors are raised as ``SpecError`` with messages written for a coding agent to
read and act on.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENERGY_WORDS = {
    "minimal": 0.15,
    "low": 0.25,
    "calm": 0.25,
    "soft": 0.3,
    "medium": 0.5,
    "mid": 0.5,
    "moderate": 0.5,
    "high": 0.8,
    "intense": 0.85,
    "peak": 0.95,
    "max": 0.95,
}

ENDINGS = ("resolve", "fade")
MIN_DURATION = 5.0
MAX_DURATION = 900.0

TOP_LEVEL_KEYS = {
    "duration", "style", "seed", "scenes", "transitions", "hits", "key", "tempo",
    "ending", "loudness", "duck", "duck_db", "voiceover", "sample_rate", "fps",
    "unit", "title", "$schema",
}
SCENE_KEYS = {"start", "time", "duration", "energy", "label", "name", "id"}


class SpecError(ValueError):
    """The cue sheet is invalid. The message says what to fix."""


@dataclass
class Scene:
    start: float
    energy: float | None = None
    label: str = ""


@dataclass
class Spec:
    duration: float
    scenes: list[Scene]
    style: str = "corporate"
    seed: int = 1
    hits: list[float] = field(default_factory=list)
    key: str | None = None
    tempo: float | None = None
    ending: str = "resolve"
    loudness: float = -18.0
    duck: list[tuple[float, float]] = field(default_factory=list)
    duck_db: float = -10.0
    voiceover: str | None = None
    sample_rate: int = 48000
    warnings: list[str] = field(default_factory=list)


def _check_keys(obj: dict, allowed: set[str], where: str) -> None:
    for k in obj:
        if k not in allowed:
            hint = difflib.get_close_matches(k, sorted(allowed), n=1)
            extra = f" Did you mean '{hint[0]}'?" if hint else ""
            raise SpecError(f"Unknown key '{k}' in {where}.{extra} Allowed: {', '.join(sorted(allowed))}.")


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"'{name}' must be a number, got {value!r}.")
    return float(value)


def parse_energy(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        word = value.strip().lower()
        if word not in ENERGY_WORDS:
            raise SpecError(f"{where}: energy '{value}' is not recognised. Use 0..1 or one of: {', '.join(ENERGY_WORDS)}.")
        return ENERGY_WORDS[word]
    e = _num(value, f"{where}.energy")
    if not 0.0 <= e <= 1.0:
        raise SpecError(f"{where}: energy must be between 0 and 1, got {e}.")
    return e


def parse_spec(data: dict) -> Spec:
    """Validate a cue-sheet dict and normalise it to seconds."""
    if not isinstance(data, dict):
        raise SpecError("The cue sheet must be a JSON object.")
    _check_keys(data, TOP_LEVEL_KEYS, "the cue sheet")

    unit = data.get("unit", "seconds")
    if unit not in ("seconds", "frames"):
        raise SpecError("'unit' must be 'seconds' or 'frames'.")
    scale = 1.0
    if unit == "frames":
        if "fps" not in data:
            raise SpecError("'unit' is 'frames' but no 'fps' was given.")
        fps = _num(data["fps"], "fps")
        if fps <= 0:
            raise SpecError("'fps' must be positive.")
        scale = 1.0 / fps

    warnings: list[str] = []
    scenes_raw = data.get("scenes")
    transitions = data.get("transitions")
    if scenes_raw is not None and transitions is not None:
        raise SpecError("Give either 'scenes' or 'transitions', not both.")

    scenes: list[Scene] = []
    if scenes_raw is not None:
        if not isinstance(scenes_raw, list) or not scenes_raw:
            raise SpecError("'scenes' must be a non-empty list.")
        by_duration = all(isinstance(s, dict) and "duration" in s and "start" not in s and "time" not in s
                          for s in scenes_raw)
        cursor = 0.0
        for i, s in enumerate(scenes_raw):
            where = f"scenes[{i}]"
            if isinstance(s, (int, float)) and not isinstance(s, bool):
                s = {"start": s}
            if not isinstance(s, dict):
                raise SpecError(f"{where} must be an object like {{\"start\": 12.5, \"energy\": 0.6}}.")
            _check_keys(s, SCENE_KEYS, where)
            if by_duration:
                d = _num(s["duration"], f"{where}.duration") * scale
                if d <= 0:
                    raise SpecError(f"{where}.duration must be positive.")
                start = cursor
                cursor += d
            else:
                if "start" not in s and "time" not in s:
                    raise SpecError(f"{where} needs a 'start' time (or give every scene a 'duration' instead).")
                start = _num(s.get("start", s.get("time")), f"{where}.start") * scale
            label = str(s.get("label", s.get("name", s.get("id", ""))))
            scenes.append(Scene(start=start, energy=parse_energy(s.get("energy"), where), label=label))
        if by_duration and "duration" not in data:
            data = {**data, "duration": cursor / scale}
    elif transitions is not None:
        if not isinstance(transitions, list):
            raise SpecError("'transitions' must be a list of times.")
        scenes = [Scene(start=0.0)] + [Scene(start=_num(t, f"transitions[{i}]") * scale)
                                       for i, t in enumerate(transitions)]

    if "duration" not in data:
        raise SpecError("'duration' (total video length) is required.")
    duration = _num(data["duration"], "duration") * scale
    if not MIN_DURATION <= duration <= MAX_DURATION:
        raise SpecError(f"'duration' must be between {MIN_DURATION:g} and {MAX_DURATION:g} seconds, got {duration:g}.")

    if not scenes:
        scenes = [Scene(start=0.0)]
    for i, sc in enumerate(scenes):
        if sc.start < 0 or sc.start >= duration:
            raise SpecError(f"Scene {i} starts at {sc.start:g}s, outside the video (0..{duration:g}s).")
    scenes.sort(key=lambda sc: sc.start)
    deduped: list[Scene] = []
    for sc in scenes:
        if deduped and abs(sc.start - deduped[-1].start) < 1e-6:
            warnings.append(f"Duplicate scene start at {sc.start:g}s ignored.")
            continue
        deduped.append(sc)
    scenes = deduped
    if scenes[0].start > 0:
        scenes.insert(0, Scene(start=0.0, energy=scenes[0].energy, label="opening"))

    hits = []
    for i, h in enumerate(data.get("hits", [])):
        if isinstance(h, dict):
            h = h.get("time", h.get("start"))
        t = _num(h, f"hits[{i}]") * scale
        if not 0 < t < duration:
            raise SpecError(f"hits[{i}] at {t:g}s is outside the video.")
        hits.append(t)

    style = str(data.get("style", "corporate")).strip().lower()
    from .styles import STYLES  # local import keeps spec importable on its own

    if style not in STYLES:
        hint = difflib.get_close_matches(style, list(STYLES), n=1)
        extra = f" Did you mean '{hint[0]}'?" if hint else ""
        raise SpecError(f"Unknown style '{style}'.{extra} Available: {', '.join(STYLES)}.")

    seed = data.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SpecError("'seed' must be an integer.")

    tempo = data.get("tempo")
    if tempo is not None:
        tempo = _num(tempo, "tempo")
        if not 50 <= tempo <= 180:
            raise SpecError("'tempo' must be between 50 and 180 BPM.")

    key = data.get("key")
    if key is not None:
        from .theory import parse_key

        try:
            parse_key(str(key))
        except ValueError as exc:
            raise SpecError(str(exc)) from None
        key = str(key)

    ending = data.get("ending", "resolve")
    if ending not in ENDINGS:
        raise SpecError(f"'ending' must be one of {ENDINGS}.")

    loudness = _num(data.get("loudness", -18.0), "loudness")
    if not -40 <= loudness <= -8:
        raise SpecError("'loudness' (integrated LUFS) must be between -40 and -8.")

    duck = []
    for i, region in enumerate(data.get("duck", [])):
        if isinstance(region, dict):
            region = [region.get("start"), region.get("end")]
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            raise SpecError(f"duck[{i}] must be [start, end].")
        a, b = _num(region[0], f"duck[{i}][0]") * scale, _num(region[1], f"duck[{i}][1]") * scale
        if b <= a:
            raise SpecError(f"duck[{i}] ends before it starts.")
        duck.append((max(0.0, a), min(duration, b)))

    duck_db = _num(data.get("duck_db", -10.0), "duck_db")
    if not -30 <= duck_db <= 0:
        raise SpecError("'duck_db' must be between -30 and 0.")

    sample_rate = data.get("sample_rate", 48000)
    if sample_rate not in (44100, 48000):
        raise SpecError("'sample_rate' must be 44100 or 48000.")

    voiceover = data.get("voiceover")
    if voiceover is not None and not isinstance(voiceover, str):
        raise SpecError("'voiceover' must be a path to an audio file.")

    return Spec(
        duration=duration, scenes=scenes, style=style, seed=seed, hits=sorted(hits), key=key,
        tempo=tempo, ending=ending, loudness=loudness, duck=duck, duck_db=duck_db,
        voiceover=voiceover, sample_rate=int(sample_rate), warnings=warnings,
    )


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SpecError(f"Cue sheet not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path} is not valid JSON: {exc}") from None
    spec = parse_spec(data)
    if spec.voiceover and not Path(spec.voiceover).is_absolute():
        spec.voiceover = str((path.parent / spec.voiceover).resolve())
    return spec
