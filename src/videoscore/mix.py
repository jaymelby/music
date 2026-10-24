"""Render a symbolic score to a mastered stereo buffer (channel-first, float32)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
from .compose import Score
from .drums import crash, drum_kit, impact, riser, swell, vinyl
from .instruments import DECAYING, INSTRUMENTS
from .spec import Spec
from .theory import midi_to_hz

TAIL = 6.0  # seconds of headroom past the end for releases and reverb
HALF_RATE = {"warm_pad", "air_pad", "strings", "supersaw"}  # nothing near Nyquist: synthesize at sr/2
CACHE_BYTES = 600 * 2**20
F32 = np.float32

FX_LEVELS = {  # gain dB, reverb send
    "crash": (-15.0, 0.15),
    "swell": (-17.0, 0.3),
    "riser": (-21.0, 0.3),
    "impact": (-9.0, 0.35),
}


@dataclass
class MixInfo:
    loudness_lufs: float
    true_peak_db: float
    duck_regions: list[tuple[float, float]]


class NoteCache:
    """Rendered notes keyed by their parameters, bounded in memory."""

    def __init__(self, limit: int = CACHE_BYTES):
        self.data: dict = {}
        self.size = 0
        self.limit = limit

    def get(self, key, make):
        buf = self.data.get(key)
        if buf is None:
            buf = make().astype(F32)
            if self.size + buf.nbytes > self.limit:
                self.data.clear()
                self.size = 0
            self.data[key] = buf
            self.size += buf.nbytes
        return buf


def _place(dst: np.ndarray, buf: np.ndarray, start: int, gain: float = 1.0) -> None:
    """Add ``buf`` into ``dst`` at sample ``start`` (both mono or both stereo)."""
    if start < 0:
        buf = buf[..., -start:]
        start = 0
    m = min(buf.shape[-1], dst.shape[-1] - start)
    if m > 0:
        if gain == 1.0:
            dst[..., start:start + m] += buf[..., :m]
        else:
            dst[..., start:start + m] += buf[..., :m] * F32(gain)


_RELEASE: dict[int, np.ndarray] = {}


def _damp(full: np.ndarray, off: int, rel: int) -> np.ndarray:
    """Note-off for a naturally decaying note: fade out over ``rel`` samples from ``off``."""
    if off >= full.shape[-1]:
        return full
    end = min(full.shape[-1], off + rel)
    curve = _RELEASE.get(end - off)
    if curve is None:
        curve = _RELEASE[end - off] = np.exp(-6.9 * np.arange(end - off) / max(1, end - off)).astype(F32)
    out = full[:, :end].copy()
    out[:, off:end] *= curve
    return out


def _render_layers(score: Score, spec: Spec, master, reverb_bus, delay_bus, kick_times, rng) -> None:
    sr, st, grid = spec.sample_rate, score.style, score.grid
    N = master.shape[-1]
    cache = NoteCache()
    by_layer: dict[str, list] = {}
    for note in score.notes:
        by_layer.setdefault(note.layer, []).append(note)
    for layer in st.layers:
        notes = by_layer.get(layer.name)
        if not notes:
            continue
        fn = INSTRUMENTS[layer.instrument]
        half = layer.instrument in HALF_RATE
        lsr = sr // 2 if half else sr
        stem = np.zeros((2, N // 2 + 1 if half else N), dtype=F32)
        human = st.humanize_ms / 1000 * (0.5 if layer.role == "bass" else 1.0)
        decaying = layer.instrument in DECAYING
        release = DECAYING.get(layer.instrument)
        params = tuple(sorted(layer.params.items()))  # layers sharing an instrument can differ
        for note in notes:
            t0 = grid.time_of(note.beat)
            t1 = grid.time_of(note.beat + note.length)
            if human:
                t0 += float(np.clip(rng.normal(0, human / 2), -human, human))
            dur = max(0.04, t1 - t0)
            vel = round(note.vel * 20) / 20  # 0.05 steps: inaudible, and lets notes be cached
            freq = midi_to_hz(note.pitch)
            if decaying:
                full = cache.get((layer.instrument, params, note.pitch, vel), lambda: fn(freq, vel, lsr, **layer.params))
                buf = full if release is None else _damp(full, int(dur * lsr), int(release * lsr))
            else:
                dur = round(dur / 0.02) * 0.02
                buf = cache.get((layer.instrument, params, note.pitch, dur, vel), lambda: fn(freq, dur, vel, lsr, **layer.params))
            _place(stem, buf, int(round(t0 * lsr)))
        if half:
            stem = dsp.resample(stem, 2, 1)[:, :N]
        if layer.highpass:
            stem = dsp.filt(stem, "highpass", layer.highpass, sr)
        if layer.lowpass:
            stem = dsp.filt(stem, "lowpass", layer.lowpass, sr)
        stem *= F32(dsp.db(layer.gain_db))
        dsp.balance(stem, layer.pan)
        if layer.sidechain and kick_times:
            stem *= dsp.sidechain(N, sr, kick_times, layer.sidechain)
        master += stem
        if layer.reverb:
            reverb_bus += stem * F32(layer.reverb)
        if layer.delay:
            delay_bus += stem * F32(layer.delay)


def _render_drums(score: Score, spec: Spec, master, reverb_bus, rng) -> None:
    sr, st, grid = spec.sample_rate, score.style, score.grid
    N = master.shape[-1]
    names = sorted({h.instrument for h in score.hits})  # fixed order: float sums must not depend on hashing
    if not names:
        return
    parts = {p.instrument: p for p in st.drums}
    kit = drum_kit(sr, spec.seed, names)
    stems = {name: np.zeros(N, dtype=F32) for name in names}
    counters = dict.fromkeys(names, 0)
    jitter = st.humanize_ms / 1000 * 0.3
    for h in score.hits:
        t = grid.time_of(h.beat)
        if jitter:
            t += float(rng.uniform(-jitter, jitter))
        takes = kit[h.instrument]
        sample = takes[counters[h.instrument] % len(takes)]
        counters[h.instrument] += 1
        _place(stems[h.instrument], sample, int(round(t * sr)), max(0.0, h.vel) ** 1.4)
    for name, mono in stems.items():
        part = parts.get(name)
        gain, pan, rev = (part.gain_db, part.pan, part.reverb) if part else (-6.0, 0.0, 0.1)
        a = (pan + 1) * np.pi / 4
        for ch, g in enumerate((np.cos(a), np.sin(a))):
            sig = mono * F32(g * np.sqrt(2) * dsp.db(gain - 6))
            master[ch] += sig
            reverb_bus[ch] += sig * F32(rev)


def _render_fx(score: Score, spec: Spec, master, reverb_bus) -> None:
    sr, grid = spec.sample_rate, score.grid
    crash_buf = impact_buf = None
    for k, fx in enumerate(score.fx):
        gain, rev = FX_LEVELS[fx.kind]
        t = grid.time_of(fx.beat)
        if fx.kind == "crash":
            crash_buf = crash(sr, spec.seed + 5) if crash_buf is None else crash_buf
            buf, start = crash_buf, t
        elif fx.kind == "impact":
            impact_buf = impact(sr, spec.seed + 6) if impact_buf is None else impact_buf
            buf, start = impact_buf, t
        else:
            length = max(0.3, t - grid.time_of(fx.beat - fx.length))
            maker = swell if fx.kind == "swell" else riser
            buf, start = maker(sr, spec.seed + 7 + k, length), t - length
        g = dsp.db(gain) * fx.vel
        _place(master, buf, int(round(start * sr)), g)
        _place(reverb_bus, buf, int(round(start * sr)), g * rev)


def render_mix(score: Score, spec: Spec, voice_regions: list[tuple[float, float]] | None = None) -> tuple[np.ndarray, MixInfo]:
    """Return the mastered (2, n) float32 buffer for ``score``, exactly ``spec.duration`` long."""
    sr, st, grid = spec.sample_rate, score.style, score.grid
    n = int(round(spec.duration * sr))
    N = n + int(TAIL * sr)
    rng = np.random.default_rng(spec.seed + 101)
    master = np.zeros((2, N), dtype=F32)
    reverb_bus = np.zeros((2, N), dtype=F32)
    delay_bus = np.zeros((2, N), dtype=F32)
    kick_times = [grid.time_of(h.beat) for h in score.hits if h.instrument.startswith("kick")]

    _render_layers(score, spec, master, reverb_bus, delay_bus, kick_times, rng)
    _render_drums(score, spec, master, reverb_bus, rng)
    _render_fx(score, spec, master, reverb_bus)
    if st.texture == "vinyl":
        master += vinyl(N, sr, spec.seed + 9) * F32(dsp.db(-36))

    rv = st.reverb
    master += dsp.reverb(reverb_bus, sr, rv.get("rt60", 2.0), rv.get("predelay", 0.02), rv.get("damping", 0.5),
                         seed=spec.seed + 11) * F32(dsp.db(-9))
    del reverb_bus
    if np.any(delay_bus):
        delay_s = 60.0 / grid.base_tempo * st.delay_beats
        master += dsp.ping_pong_delay(delay_bus, delay_s, st.delay_feedback, sr)[:, :N] * F32(dsp.db(-7))
    del delay_bus

    master = dsp.filt(master, "highpass", 35, sr)  # keep sub rumble out from under narration
    if st.master_lowpass:
        master = dsp.filt(master, "lowpass", st.master_lowpass, sr)

    if spec.ending == "resolve" and score.final_beat is not None:
        final_t = grid.time_of(score.final_beat)
        ring = spec.duration - final_t
        fade_start, fade_len = final_t + 0.35 * ring, 0.65 * ring
    else:
        fade_len = min(5.0, max(2.0, 0.1 * spec.duration))
        fade_start = spec.duration - fade_len
    master = master[:, :n] * dsp.fade(n, sr, 0.01, fade_start, fade_len)

    # loudness is set on the un-ducked music so ducking really lowers it under speech
    master *= F32(dsp.db(spec.loudness - dsp.lufs(master, sr)))
    regions = list(spec.duck) + list(voice_regions or [])
    if regions:
        master *= dsp.region_gain(n, sr, regions, spec.duck_db)
    master = dsp.limit(master, sr, -1.5)
    overshoot = dsp.true_peak_db(master) + 1.0  # inter-sample peaks can poke above the sample peak
    if overshoot > 0:
        master = dsp.limit(master, sr, -1.5 - overshoot)
    master[:, -1] = 0.0
    info = MixInfo(loudness_lufs=dsp.lufs(master, sr), true_peak_db=dsp.true_peak_db(master), duck_regions=regions)
    return master, info
