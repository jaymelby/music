"""Synthesized drums, cymbals and transition effects."""

from __future__ import annotations

import numpy as np

from . import dsp


def _t(sr: int, seconds: float) -> np.ndarray:
    return np.arange(int(seconds * sr)) / sr


def _sweep_sine(sr: int, seconds: float, f_start: float, f_end: float, tau: float) -> np.ndarray:
    t = _t(sr, seconds)
    f = f_end + (f_start - f_end) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / sr)


def _attack(x: np.ndarray, sr: int, seconds: float = 0.001) -> np.ndarray:
    a = max(1, int(seconds * sr))
    x[:a] *= np.linspace(0, 1, a)
    return x


def _metal(sr: int, seconds: float, rng, scale: float = 2.0) -> np.ndarray:
    t = _t(sr, seconds)
    freqs = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * scale
    sig = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in freqs)
    return sig / len(freqs)


def kick(sr, rng, f0=160.0, f1=52.0, decay=0.3, click=0.25):
    t = _t(sr, decay * 3)
    body = _sweep_sine(sr, decay * 3, f0, f1, 0.032) * np.exp(-t / decay)
    noise = dsp.filt(rng.standard_normal(len(t)), "highpass", 2500, sr)
    sig = body + click * noise * np.exp(-t / 0.004)
    return _attack(np.tanh(1.8 * sig) / np.tanh(1.8), sr, 0.0008)


def kick_soft(sr, rng):
    sig = kick(sr, rng, f0=115.0, f1=54.0, decay=0.26, click=0.08)
    return dsp.filt(sig, "lowpass", 2800, sr)


def snare(sr, rng, tone_decay=0.09, noise_decay=0.17, low=1400.0):
    t = _t(sr, 0.6)
    tone = (np.sin(2 * np.pi * 185 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / tone_decay)
    noise = dsp.bandpass(rng.standard_normal(len(t)), low, 9500, sr) * np.exp(-t / noise_decay)
    return _attack(0.55 * tone + 1.3 * noise, sr)


def snare_big(sr, rng):
    t = _t(sr, 1.0)
    tone = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 0.14)
    noise = dsp.bandpass(rng.standard_normal(len(t)), 900, 8000, sr) * np.exp(-t / 0.32)
    return _attack(0.7 * tone + 1.1 * noise, sr)


def clap(sr, rng):
    t = _t(sr, 0.5)
    noise = dsp.bandpass(rng.standard_normal(len(t)), 900, 6500, sr)
    env = np.zeros(len(t))
    for d in (0.0, 0.009, 0.019, 0.03):
        k = int(d * sr)
        env[k:] = np.maximum(env[k:], np.exp(-(t[: len(t) - k]) / 0.006))
    k = int(0.03 * sr)
    env[k:] = np.maximum(env[k:], 0.7 * np.exp(-(t[: len(t) - k]) / 0.13))
    return noise * env * 1.4


def rim(sr, rng):
    t = _t(sr, 0.15)
    tone = (np.sin(2 * np.pi * 1700 * t) + 0.6 * np.sin(2 * np.pi * 820 * t)) * np.exp(-t / 0.018)
    noise = dsp.bandpass(rng.standard_normal(len(t)), 2000, 7000, sr) * np.exp(-t / 0.008)
    return _attack(0.5 * tone + 0.6 * noise, sr, 0.0005)


def hat(sr, rng, decay=0.045):
    t = _t(sr, decay * 6)
    metal = dsp.filt(_metal(sr, decay * 6, rng), "highpass", 7000, sr, order=4)
    noise = dsp.filt(rng.standard_normal(len(t)), "highpass", 8000, sr, order=2)
    return _attack((0.6 * metal + 0.5 * noise) * np.exp(-t / decay), sr, 0.0005)


def ohat(sr, rng):
    return hat(sr, rng, decay=0.26)


def shaker(sr, rng):
    t = _t(sr, 0.18)
    noise = dsp.bandpass(rng.standard_normal(len(t)), 4500, 12000, sr)
    env = np.minimum(t / 0.012, 1) * np.exp(-np.maximum(t - 0.012, 0) / 0.05)
    return noise * env * 0.8


def tom_low(sr, rng):
    t = _t(sr, 1.2)
    body = _sweep_sine(sr, 1.2, 105, 62, 0.08) * np.exp(-t / 0.5)
    over = _sweep_sine(sr, 1.2, 170, 100, 0.06) * np.exp(-t / 0.18)
    thump = dsp.filt(rng.standard_normal(len(t)), "lowpass", 900, sr) * np.exp(-t / 0.02)
    return _attack(np.tanh(1.5 * (body + 0.35 * over + 0.4 * thump)), sr)


DRUMS = {
    "kick": kick,
    "kick_soft": kick_soft,
    "snare": snare,
    "snare_big": snare_big,
    "clap": clap,
    "rim": rim,
    "hat": hat,
    "ohat": ohat,
    "shaker": shaker,
    "tom_low": tom_low,
}


def drum_kit(sr: int, seed: int, names, variants: int = 3) -> dict[str, list[np.ndarray]]:
    """Render a few slightly different takes of each drum to avoid the machine-gun effect."""
    kit = {}
    for i, name in enumerate(sorted(set(names))):
        takes = []
        for v in range(variants):
            rng = np.random.default_rng(seed * 1000 + i * 10 + v)
            x = DRUMS[name](sr, rng)
            takes.append(x / max(np.max(np.abs(x)), 1e-9))
        kit[name] = takes
    return kit


# ---------------------------------------------------------------- effects
def crash(sr: int, seed: int, seconds: float = 3.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(sr, seconds)
    out = np.zeros((2, len(t)))
    for ch in range(2):
        noise = rng.standard_normal(len(t))
        metal = _metal(sr, seconds, rng, scale=3.1)
        raw = dsp.filt(0.7 * noise + 0.5 * metal, "highpass", 3200, sr, order=2)
        dark = dsp.filt(raw, "lowpass", 5500, sr)
        mix = np.exp(-t / 0.5)
        out[ch] = (raw * mix + dark * (1 - mix)) * np.exp(-t / 1.25)
    out[:, : int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))
    return out / np.max(np.abs(out))


def swell(sr: int, seed: int, seconds: float) -> np.ndarray:
    """Reverse cymbal that peaks exactly at its end."""
    c = crash(sr, seed, max(seconds, 0.5) + 0.2)[:, : int(seconds * sr)]
    rev = c[:, ::-1] * np.linspace(0, 1, c.shape[1]) ** 1.5
    k = int(0.01 * sr)
    rev[:, -k:] *= np.linspace(1, 0.3, k)
    return rev


def riser(sr: int, seed: int, seconds: float) -> np.ndarray:
    """Filtered noise sweep rising into the next section."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    frame, hop = 2048, 512
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    window = np.hanning(frame)
    out = np.zeros((2, n + frame))
    noise = rng.standard_normal((2, n + frame))
    for start in range(0, n, hop):
        x = (start + frame / 2) / n
        fc = 250 * (9000 / 250) ** min(x, 1.0)
        mask = np.exp(-0.5 * (np.log2(np.maximum(freqs, 1) / fc) / 0.7) ** 2)
        spec = np.fft.rfft(noise[:, start:start + frame] * window, axis=-1) * mask
        out[:, start:start + frame] += np.fft.irfft(spec, frame, axis=-1) * window
    out = out[:, :n] * np.linspace(0, 1, n) ** 2.2
    k = int(0.02 * sr)
    out[:, -k:] *= np.linspace(1, 0, k)
    return out / max(np.max(np.abs(out)), 1e-9)


def impact(sr: int, seed: int) -> np.ndarray:
    """Deep cinematic boom."""
    rng = np.random.default_rng(seed)
    t = _t(sr, 3.0)
    boom = _sweep_sine(sr, 3.0, 90, 42, 0.25) * np.exp(-t / 0.8)
    body = dsp.filt(rng.standard_normal(len(t)), "lowpass", 700, sr) * np.exp(-t / 0.12)
    crack = dsp.bandpass(rng.standard_normal(len(t)), 1500, 7000, sr) * np.exp(-t / 0.03)
    sig = np.tanh(1.6 * (boom + 0.5 * body)) + 0.25 * crack
    sig = _attack(sig, sr, 0.001)
    return np.stack([sig, sig]) / np.max(np.abs(sig))


def vinyl(n: int, sr: int, seed: int) -> np.ndarray:
    """Dusty record texture: soft hiss plus sparse crackles."""
    rng = np.random.default_rng(seed)
    hiss = dsp.filt(dsp.filt(rng.standard_normal((2, n)).astype(np.float32), "lowpass", 5000, sr),
                    "highpass", 700, sr) * 0.05
    clicks = np.zeros((2, n), dtype=np.float32)
    count = int(n / sr * 9)
    clicks[rng.integers(0, 2, count), rng.integers(0, n, count)] = (
        rng.uniform(0.2, 1.0, count) * rng.choice([-1, 1], count))
    return hiss + dsp.bandpass(clicks, 1500, 9000, sr) * 1.6
