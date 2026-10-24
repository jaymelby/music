"""Pitched instruments, synthesized from scratch (no samples or soundfonts).

Every oscillator is band-limited (wavetables built from harmonics below
Nyquist, or additive partials), which avoids the harsh aliasing that makes
naive digital synths sound cheap. Each function returns a stereo buffer that
includes the release tail. Output depends only on the arguments, so notes
can be cached.
"""

from __future__ import annotations

import zlib

import numpy as np

from . import dsp

TABLE = 4096


def _rng(*key) -> np.random.Generator:
    """Deterministic generator per note (``hash()`` is salted per process, crc32 is not)."""
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _table(amps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    spec = np.zeros(TABLE // 2 + 1, dtype=complex)
    k = np.arange(1, len(amps) + 1)
    spec[k] = amps * np.exp(1j * phases)
    t = np.fft.irfft(spec, TABLE) * TABLE / 2
    return t / max(np.max(np.abs(t)), 1e-9)


def _saw_table(freq: float, sr: int, cutoff: float, slope: float = 1.0, odd_only: bool = False, seed: int = 0):
    nyq = sr / 2
    kmax = int(min(nyq * 0.92, max(cutoff * 6, freq * 2)) // freq)
    kmax = max(1, min(kmax, TABLE // 2 - 1))
    k = np.arange(1, kmax + 1)
    amps = 1.0 / k**slope
    amps *= 1.0 / np.sqrt(1 + (k * freq / cutoff) ** 4)  # 2nd-order low-pass shape
    if odd_only:
        amps[k % 2 == 0] = 0
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, kmax)
    return _table(amps, phases)


_TABLE_X = np.arange(TABLE + 1, dtype=float)


def _read(table: np.ndarray, freq: float, n: int, sr: int, phase0: float = 0.0,
          fm: np.ndarray | None = None) -> np.ndarray:
    """Linear-interpolated wavetable oscillator (``fm``: per-sample frequency multiplier)."""
    step = freq / sr * TABLE
    if fm is None:
        ph = np.arange(n) * step
    else:
        ph = np.cumsum(fm * step)
        ph -= ph[0]
    ph += phase0 * TABLE
    np.mod(ph, TABLE, out=ph)
    return np.interp(ph, _TABLE_X, np.append(table, table[0]))


def _adsr(n_on: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """Envelope for a note held ``n_on`` samples, followed by its release."""
    na, nd, nr = max(1, int(a * sr)), max(1, int(d * sr)), max(1, int(r * sr))
    env = np.full(n_on, s, dtype=float)
    k = min(na, n_on)
    env[:k] = np.sin(np.linspace(0, np.pi / 2, na))[:k] ** 2
    if n_on > na:
        k2 = min(nd, n_on - na)
        env[na:na + k2] = s + (1 - s) * np.exp(-5 * np.arange(k2) / nd)
    last = env[-1] if n_on else 0.0
    rel = last * np.exp(-6.9 * np.arange(nr) / nr)
    return np.concatenate([env, rel])


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right])


# ---------------------------------------------------------------- sustained
def warm_pad(freq, dur, vel, sr, attack=0.5, release=1.2, cutoff=2000.0, voices=3, detune=7.0, **_):
    n_on = int(dur * sr)
    env = _adsr(n_on, sr, attack, 0.6, 0.85, release)
    n = len(env)
    t = np.arange(n) / sr
    fc = cutoff * (0.6 + 0.6 * vel)
    tab = _saw_table(freq, sr, fc, slope=1.2, seed=int(freq))
    rng = _rng("pad", round(freq, 3))
    outs = []
    spread = np.linspace(-detune, detune, voices)
    for c in spread:
        lfo = 1 + 0.0012 * np.sin(2 * np.pi * rng.uniform(0.15, 0.4) * t + rng.uniform(0, 6.28))
        outs.append(_read(tab, freq * 2 ** (c / 1200), n, sr, rng.uniform(), lfo))
    left = sum(o * w for o, w in zip(outs, np.linspace(1.0, 0.3, voices)))
    right = sum(o * w for o, w in zip(outs, np.linspace(0.3, 1.0, voices)))
    return _stereo(left, right) * env * (0.6 / voices) * (0.5 + 0.5 * vel)


def air_pad(freq, dur, vel, sr, attack=2.0, release=3.0, **_):
    n_on = int(dur * sr)
    env = _adsr(n_on, sr, attack, 1.0, 0.9, release)
    n = len(env)
    tab = _saw_table(freq, sr, freq * 5, slope=1.6, odd_only=True, seed=int(freq) + 1)
    t = np.arange(n) / sr
    l = _read(tab, freq * 2 ** (-5 / 1200), n, sr, 0.1, 1 + 0.002 * np.sin(2 * np.pi * 0.2 * t))
    r = _read(tab, freq * 2 ** (5 / 1200), n, sr, 0.6, 1 + 0.002 * np.sin(2 * np.pi * 0.27 * t + 1))
    return _stereo(l, r) * env * 0.35 * (0.5 + 0.5 * vel)


def strings(freq, dur, vel, sr, attack=0.6, release=1.0, **_):
    n_on = int(dur * sr)
    env = _adsr(n_on, sr, attack, 0.8, 0.9, release)
    n = len(env)
    t = np.arange(n) / sr
    fc = 1800 + 2200 * vel
    tab = _saw_table(freq, sr, fc, slope=1.0, seed=int(freq) + 2)
    rng = _rng("strings", round(freq, 3))
    vib_amt = np.clip((t - 0.35) / 0.6, 0, 1) * 0.0035
    outs = []
    spread = [-11, -5, 0, 5, 11]
    for c in spread:
        vib = 1 + vib_amt * np.sin(2 * np.pi * rng.uniform(4.6, 5.6) * t + rng.uniform(0, 6.28))
        outs.append(_read(tab, freq * 2 ** (c / 1200), n, sr, rng.uniform(), vib))
    w = np.linspace(1, 0, len(outs))
    left = sum(o * (0.4 + 0.6 * a) for o, a in zip(outs, w))
    right = sum(o * (0.4 + 0.6 * (1 - a)) for o, a in zip(outs, w))
    out = _stereo(left, right) * env * 0.16 * (0.4 + 0.6 * vel)
    return dsp.filt(out, "lowpass", fc * 1.2, sr)


def supersaw(freq, dur, vel, sr, attack=0.03, release=0.25, **_):
    n_on = int(dur * sr)
    env = _adsr(n_on, sr, attack, 0.3, 0.8, release)
    n = len(env)
    tab = _saw_table(freq, sr, 4200, slope=1.0, seed=int(freq) + 3)
    rng = _rng("ssaw", round(freq, 3))
    spread = [-18, -11, -5, 0, 5, 11, 18]
    outs = [_read(tab, freq * 2 ** (c / 1200), n, sr, rng.uniform()) for c in spread]
    w = np.linspace(1, 0, len(outs))
    left = sum(o * (0.3 + 0.7 * a) for o, a in zip(outs, w))
    right = sum(o * (0.3 + 0.7 * (1 - a)) for o, a in zip(outs, w))
    return _stereo(left, right) * env * 0.1 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- struck / plucked
# These are rendered once per (pitch, velocity) with their natural decay; the
# mixer applies the note-off (damping) per note, so repeated notes are cached.
MAX_DECAY = 8.0


def _cap(sig: np.ndarray, sr: int) -> np.ndarray:
    n = int(MAX_DECAY * sr)
    if len(sig) > n:
        sig = sig[:n]
        f = int(0.8 * sr)
        sig[-f:] *= np.linspace(1, 0, f) ** 2
    return sig


def ep(freq, vel, sr, **_):
    """FM electric piano (Rhodes-like): bell-ish attack that mellows as it decays."""
    tau = 1.8 * (220 / freq) ** 0.35
    n = int(min(tau * 5, MAX_DECAY) * sr)
    t = np.arange(n) / sr
    amp = np.exp(-t / tau)
    amp[:int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))
    index = (0.4 + 1.6 * vel) * np.exp(-t / 0.5) + 0.25
    ph = 2 * np.pi * freq * t
    sin_ph = np.sin(ph)
    body = np.sin(ph + index * sin_ph)
    tine_ratio = 14 if freq * 15 < sr * 0.45 else 7
    m = int(0.25 * sr)
    tine = np.sin(ph[:m] + 1.2 * vel * np.exp(-t[:m] / 0.03) * np.sin(tine_ratio * ph[:m])) - sin_ph[:m]
    body[:m] += 0.5 * tine
    sig = _cap(body * amp * (0.35 + 0.65 * vel), sr)
    trem = 0.18 * np.sin(2 * np.pi * 4.2 * t[: len(sig)])
    return _stereo(sig * (1 + trem), sig * (1 - trem)) * 0.5


def _additive(partials, n: int, sr: int, seed) -> np.ndarray:
    """Sum of decaying sinusoids. ``partials``: iterable of (freq, amp, tau)."""
    out = np.zeros(n)
    rng = _rng(*seed)
    t = np.arange(n) / sr
    for f, a, tau in partials:
        if f >= sr * 0.45 or a < 1e-4:
            continue
        m = min(n, int(tau * 5.5 * sr) + 1)  # stop at -48 dB
        out[:m] += a * np.exp(-t[:m] / tau) * np.sin(2 * np.pi * f * t[:m] + rng.uniform(0, 6.28))
    return out


def _haas(sig: np.ndarray, sr: int, delay: float = 0.0004) -> np.ndarray:
    """Tiny inter-channel delay for width without phasey chorus."""
    d = int(delay * sr)
    return _stereo(sig, np.concatenate([np.zeros(d), sig[: len(sig) - d]]))


def pluck(freq, vel, sr, decay=0.45, **_):
    """Plucked synth: upper harmonics die away first, like a damped string."""
    kmax = int(min(24, (sr * 0.42) // freq))
    bright = 0.5 + 0.8 * vel
    partials = [(k * freq, (1 / k**1.1) * np.exp(-(k - 1) * (0.32 / bright)), decay / (1 + 0.55 * (k - 1)))
                for k in range(1, kmax + 1)]
    sig = _additive(partials, int(decay * 5 * sr), sr, ("pluck", round(freq, 3)))
    a = int(0.0015 * sr)
    sig[:a] *= np.linspace(0, 1, a)
    return _haas(sig, sr) * 0.32 * (0.4 + 0.6 * vel)


def bell(freq, vel, sr, **_):
    """Soft celesta / music-box tone."""
    tau = 1.3 * (523 / freq) ** 0.5
    partials = [(freq, 1.0, tau), (2 * freq, 0.32 * vel + 0.08, tau * 0.45), (3 * freq, 0.12, tau * 0.3),
                (4.07 * freq, 0.06 * vel, tau * 0.18), (5.4 * freq, 0.03, tau * 0.12)]
    sig = _additive(partials, int(min(tau * 6, MAX_DECAY) * sr), sr, ("bell", round(freq, 3)))
    a = int(0.001 * sr)
    sig[:a] *= np.linspace(0, 1, a)
    return _stereo(sig, sig) * 0.22 * (0.3 + 0.7 * vel)


def soft_piano(freq, vel, sr, **_):
    """Felt piano: stretched partials, two-stage decay and gently beating string pairs."""
    B = 0.00025 * (freq / 261.6) ** 0.6
    kmax = int(min(16, (sr * 0.4) // freq))
    cutoff = 900 + 2600 * vel
    long_tau = 3.5 * (261.6 / freq) ** 0.65
    partials = []
    for k in range(1, kmax + 1):
        fk = k * freq * np.sqrt(1 + B * k * k)
        a = (1 / k**1.25) / np.sqrt(1 + (fk / cutoff) ** 4)
        partials.append((fk * (1 + 0.00035), a * 0.55, 0.32 / (1 + 0.12 * k)))
        partials.append((fk * (1 - 0.00035), a * 0.45, long_tau / (1 + 0.18 * k)))
    sig = _additive(partials, int(min(long_tau * 6, MAX_DECAY) * sr), sr, ("piano", round(freq, 3)))
    a = int(0.003 * sr)
    sig[:a] *= np.linspace(0, 1, a)
    return _haas(_cap(sig, sr), sr, 0.0003) * 0.42 * (0.35 + 0.65 * vel)


def round_bass(freq, vel, sr, **_):
    """Warm, rounded electric/upright-style bass."""
    kmax = int(min(8, (sr * 0.4) // freq))
    partials = [(k * freq, 1 / k**1.6, 1.1 / (1 + 0.7 * (k - 1))) for k in range(1, kmax + 1)]
    sig = _additive(partials, int(5 * sr), sr, ("rbass", round(freq, 3)))
    a = int(0.004 * sr)
    sig[:a] *= np.linspace(0, 1, a)
    sig = np.tanh(1.2 * sig)
    return _stereo(sig, sig) * 0.55 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- bass
def sub_bass(freq, dur, vel, sr, **_):
    n_on = int(dur * sr)
    env = _adsr(n_on, sr, 0.006, 0.25, 0.75, 0.07)
    n = len(env)
    ph = 2 * np.pi * freq * np.arange(n) / sr
    sig = np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)
    sig = np.tanh(1.4 * sig) / np.tanh(1.4)
    out = sig * env * 0.5 * (0.5 + 0.5 * vel)
    return _stereo(out, out)


def synth_bass(freq, dur, vel, sr, **_):
    """Filtered saw bass with a short filter 'pluck' at the start of each note."""
    n_on = int(dur * sr)
    env = _adsr(n_on, sr, 0.004, 0.15, 0.8, 0.05)
    n = len(env)
    bright = _saw_table(freq, sr, 1400, seed=int(freq) + 9)
    dark = _saw_table(freq, sr, 300, seed=int(freq) + 9)
    t = np.arange(n) / sr
    mix = np.exp(-t / 0.09)
    sig = _read(bright, freq, n, sr) * mix + _read(dark, freq, n, sr) * (1 - mix)
    sub = np.sin(2 * np.pi * freq * t)
    sig = np.tanh(1.5 * (0.6 * sig + 0.6 * sub))
    out = sig * env * 0.42 * (0.5 + 0.5 * vel)
    return _stereo(out, out)


# note-off release time for decaying instruments (None: let it ring, like a bell)
DECAYING = {"ep": 0.18, "pluck": 0.08, "bell": None, "soft_piano": 0.3, "round_bass": 0.08}

INSTRUMENTS = {
    "warm_pad": warm_pad,
    "air_pad": air_pad,
    "strings": strings,
    "supersaw": supersaw,
    "ep": ep,
    "pluck": pluck,
    "bell": bell,
    "soft_piano": soft_piano,
    "sub_bass": sub_bass,
    "round_bass": round_bass,
    "synth_bass": synth_bass,
}
