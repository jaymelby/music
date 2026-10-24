"""Signal processing: filters, reverb, delay, envelopes, loudness and limiting.

Buffers are channel-first: ``(n,)`` for mono or ``(2, n)`` for stereo, so every
channel is contiguous in memory and filters run along the last axis. Long
buffers in the mixer are float32 to halve memory traffic.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def db(x: float) -> float:
    return 10 ** (x / 20)


def sos(kind: str, cutoff: float, sr: int, order: int = 2):
    nyq = sr / 2
    cutoff = min(max(cutoff, 10.0), nyq * 0.95)
    return signal.butter(order, cutoff / nyq, btype=kind, output="sos")


def _sosfilt(s, x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        s = s.astype(np.float32)  # keep single precision instead of upcasting
    return signal.sosfilt(s, x, axis=-1)


def filt(x: np.ndarray, kind: str, cutoff: float, sr: int, order: int = 2) -> np.ndarray:
    return _sosfilt(sos(kind, cutoff, sr, order), x)


def bandpass(x: np.ndarray, lo: float, hi: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = sr / 2
    s = signal.butter(order, [max(lo, 10) / nyq, min(hi, nyq * 0.95) / nyq], btype="band", output="sos")
    return _sosfilt(s, x)


def resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    return signal.resample_poly(x, up, down, axis=-1).astype(x.dtype, copy=False)


def balance(x: np.ndarray, pan: float) -> np.ndarray:
    """Pan a stereo signal by attenuating the opposite side (in place)."""
    if pan > 0:
        x[0] *= 1 - pan
    elif pan < 0:
        x[1] *= 1 + pan
    return x


def reverb_ir(sr: int, rt60: float, predelay: float, damping: float, seed: int = 7) -> np.ndarray:
    """Synthetic stereo impulse response: decorrelated noise with frequency-dependent decay."""
    rng = np.random.default_rng(seed)
    n = int(sr * min(rt60 * 1.3, 8.0))
    t = np.arange(n) / sr
    noise = rng.standard_normal((2, n))
    low = filt(noise, "lowpass", 2500, sr)
    high = noise - low
    env_low = np.exp(-6.91 * t / rt60)
    env_high = np.exp(-6.91 * t / (rt60 * (1 - 0.75 * damping)))
    ir = low * env_low + 0.6 * high * env_high
    ir = filt(filt(ir, "highpass", 180, sr), "lowpass", 10000, sr)
    ramp = min(n, int(0.012 * sr))
    ir[:, :ramp] *= np.linspace(0, 1, ramp) ** 2  # soft onset so the tail blooms instead of clicking
    rms = np.sqrt(np.mean(ir**2))
    for d, g, side in ((0.011, 0.35, 0), (0.017, 0.3, 1), (0.023, 0.22, 0), (0.031, 0.18, 1)):
        k = int(d * sr)
        if k < n:
            ir[side, k] += g * rms * 40  # a few early reflections
    ir /= np.sqrt(np.sum(ir**2) / 2)
    return np.concatenate([np.zeros((2, int(predelay * sr))), ir], axis=1)


def reverb(x: np.ndarray, sr: int, rt60: float, predelay: float, damping: float, seed: int) -> np.ndarray:
    """Convolution reverb, run at half rate (the tail is dark anyway) to save time."""
    half = resample(x, 1, 2)
    ir = reverb_ir(sr // 2, rt60, predelay, damping, seed).astype(x.dtype)
    wet = np.stack([signal.oaconvolve(half[ch], ir[ch]) for ch in range(2)])
    wet = resample(wet, 2, 1)
    n = x.shape[-1]
    return wet[:, :n] if wet.shape[-1] >= n else np.pad(wet, ((0, 0), (0, n - wet.shape[-1])))


def ping_pong_delay(x: np.ndarray, delay_s: float, feedback: float, sr: int, repeats: int = 6) -> np.ndarray:
    """Tempo-synced stereo echo bouncing left/right, filtered so it stays behind the dry sound."""
    d = int(delay_s * sr)
    n = x.shape[-1]
    out = np.zeros((2, n + d * repeats), dtype=x.dtype)
    mono = x.mean(axis=0)
    for k in range(1, repeats + 1):
        out[(k + 1) % 2, k * d:k * d + n] += mono * feedback ** (k - 1)
    return filt(filt(out, "lowpass", 4500, sr), "highpass", 300, sr)


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness (ITU-R BS.1770-4)."""
    if x.ndim == 1:
        x = np.stack([x, x])
    # K-weighting: high shelf + high pass (coefficients as in pyloudnorm)
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (gain / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    b_shelf = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a_shelf = [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    a_hp = [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    kw = np.concatenate([signal.tf2sos(b_shelf, a_shelf), signal.tf2sos([1, -2, 1], a_hp)])
    y = signal.sosfilt(kw, x.astype(np.float64), axis=-1)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if y.shape[-1] < block:
        return -70.0
    cs = np.concatenate([np.zeros(1), np.cumsum((y**2).sum(axis=0))])
    starts = np.arange(0, y.shape[-1] - block + 1, hop)
    blocks = (cs[starts + block] - cs[starts]) / block
    loud = -0.691 + 10 * np.log10(np.maximum(blocks, 1e-12))
    gated = blocks[loud > -70]
    if not len(gated):
        return -70.0
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10
    gated = blocks[(loud > -70) & (loud > rel)]
    return float(-0.691 + 10 * np.log10(gated.mean()))


def true_peak_db(x: np.ndarray, block: int = 4096) -> float:
    """4x-oversampled peak level, evaluated around the loudest blocks only."""
    x = np.atleast_2d(x)
    n = x.shape[-1] // block * block
    peaks = np.abs(x[:, :n]).reshape(x.shape[0], -1, block).max(axis=(0, 2)) if n else np.zeros(0)
    top = float(np.max(np.abs(x))) if x.size else 0.0
    best = top
    candidates = np.flatnonzero(peaks > top * 0.89)  # inter-sample overs are well under 1 dB
    candidates = candidates[np.argsort(peaks[candidates])[::-1][:64]]
    if n == 0:
        candidates = [0]
    for i in candidates:
        a, b = max(0, i * block - 64), min(x.shape[-1], (i + 1) * block + 64)
        up = signal.resample_poly(x[:, a:b].astype(np.float64), 4, 1, axis=-1)
        best = max(best, float(np.max(np.abs(up[:, 256:-256] if up.shape[-1] > 512 else up))))
    return float(20 * np.log10(max(best, 1e-9)))


def _windowed(fn, x: np.ndarray, before: int, after: int) -> np.ndarray:
    """Apply a scipy 1-D window filter over [t - before, t + after] for every t."""
    size = before + after + 1
    shift = size // 2 - before
    pad = abs(shift) + 1
    xp = np.pad(x, (pad, pad), mode="edge")
    y = fn(xp, size=size, mode="nearest")
    return y[pad + shift: pad + shift + len(x)]


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.5, attack: float = 0.008, hold: float = 0.06) -> np.ndarray:
    """Look-ahead brickwall limiter.

    ``need`` is the gain each sample requires. Taking the minimum over
    [t - hold, t + attack] and then averaging over [t - attack, t] gives a smooth
    curve that is never above ``need`` anywhere, so no sample exceeds the ceiling.
    """
    ceiling = db(ceiling_db)
    peak = np.max(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if need.min() >= 1.0:
        return x
    w, h = max(1, int(attack * sr)), int(hold * sr)
    g = _windowed(minimum_filter1d, need, before=h, after=w)
    g = _windowed(uniform_filter1d, g, before=w, after=0)
    g = np.minimum(g, need)  # guard against float rounding
    return (x * g).astype(x.dtype, copy=False)


def sidechain(n: int, sr: int, times: list[float], depth: float, release: float = 0.22) -> np.ndarray:
    """Gain curve that dips after each kick and recovers smoothly (classic pumping)."""
    g = np.ones(n, dtype=np.float32)
    if depth <= 0 or not times:
        return g
    attack = int(0.004 * sr)
    rel = int(release * sr)
    shape = np.concatenate([np.linspace(1, 1 - depth, attack, endpoint=False),
                            1 - depth * (1 - (np.linspace(0, 1, rel) ** 1.6))])
    for t in times:
        i = int(t * sr)
        if i >= n:
            continue
        j = min(n, i + len(shape))
        g[i:j] = np.minimum(g[i:j], shape[: j - i])
    return g


def region_gain(n: int, sr: int, regions: list[tuple[float, float]], gain_db: float, attack: float = 0.25,
                release: float = 0.6) -> np.ndarray:
    """Smooth gain curve that is ``gain_db`` inside regions (narration ducking).

    The dip starts slightly before each region so speech never fights the music.
    """
    if not regions:
        return np.ones(n, dtype=np.float32)
    low = db(gain_db)
    target = np.zeros(n, dtype=np.float32)
    for a, b in regions:
        i, j = max(0, int((a - attack) * sr)), min(n, int((b + 0.1) * sr))
        target[i:j] = 1.0
    smooth = uniform_filter1d(target, size=max(1, int(release * sr)), mode="nearest")
    smooth = np.maximum(smooth, uniform_filter1d(target, size=max(1, int(attack * sr)), mode="nearest"))
    smooth = np.clip(smooth * 1.5, 0, 1)
    return (1 - (1 - low) * smooth).astype(np.float32)


def fade(n: int, sr: int, fade_in: float, fade_out_start: float, fade_out_len: float) -> np.ndarray:
    g = np.ones(n, dtype=np.float32)
    a = int(fade_in * sr)
    if a > 0:
        g[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    s = int(fade_out_start * sr)
    m = min(n, s + int(fade_out_len * sr))
    if s < n:
        g[s:m] *= np.cos(np.linspace(0, np.pi / 2, m - s)) ** 2
        g[m:] = 0
    return g
