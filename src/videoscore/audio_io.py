"""Reading and writing audio files (WAV natively, everything else through ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ENCODED = {".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"], ".m4a": ["-c:a", "aac", "-b:a", "192k"],
           ".aac": ["-c:a", "aac", "-b:a", "192k"], ".ogg": ["-c:a", "libvorbis", "-q:a", "6"],
           ".flac": ["-c:a", "flac"], ".opus": ["-c:a", "libopus", "-b:a", "160k"]}


def write_wav(path: str | Path, audio: np.ndarray, sr: int, bits: int = 16, seed: int = 0) -> None:
    """Write channel-first (channels, n) float audio to WAV with TPDF dither."""
    if bits not in (16, 24):
        raise ValueError("bits must be 16 or 24")
    audio = np.atleast_2d(audio).T  # interleave: (n, channels)
    scale = 2 ** (bits - 1) - 1
    rng = np.random.default_rng(seed)
    dither = (rng.random(audio.shape, dtype=np.float32) - rng.random(audio.shape, dtype=np.float32)) / scale
    q = np.clip(np.round((audio + dither) * scale), -scale - 1, scale).astype(np.int32)
    if bits == 16:
        data = q.astype("<i2").tobytes()
    else:
        b = q.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        data = np.ascontiguousarray(b).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        w.writeframes(data)


def save(path: str | Path, audio: np.ndarray, sr: int, seed: int = 0) -> Path:
    """Save to .wav directly, or encode to .mp3/.m4a/.ogg/.flac/.opus with ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".wav":
        write_wav(path, audio, sr, seed=seed)
        return path
    if ext not in ENCODED:
        raise ValueError(f"Unsupported output format '{ext}'. Use .wav, " + ", ".join(ENCODED))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"ffmpeg is needed to write {ext} files. Install it or write a .wav instead.")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        write_wav(wav, audio, sr, bits=24, seed=seed)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), *ENCODED[ext], str(path)]
        subprocess.run(cmd, check=True)
    return path


def read_mono(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Decode any audio file to mono float at ``sr`` (WAV without ffmpeg)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if path.suffix.lower() != ".wav":
        raise RuntimeError("ffmpeg is needed to read non-WAV voice-over files.")
    with wave.open(str(path)) as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 2**31
    else:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float64) / 2**23
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != sr:
        from scipy.signal import resample_poly

        x = resample_poly(x, sr, rate)
    return x


def speech_regions(path: str | Path, min_gap: float = 0.35, min_len: float = 0.15) -> list[tuple[float, float]]:
    """Find where a voice-over is talking (simple energy gate)."""
    sr = 16000
    x = read_mono(path, sr)
    hop = int(0.03 * sr)
    if len(x) < hop:
        return []
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    thresh = max(-48.0, np.percentile(rms_db, 95) - 28)
    active = rms_db > thresh
    regions: list[list[float]] = []
    for i, on in enumerate(active):
        if not on:
            continue
        t = i * hop / sr
        if regions and t - regions[-1][1] <= min_gap:
            regions[-1][1] = t + hop / sr
        else:
            regions.append([t, t + hop / sr])
    return [(a, b) for a, b in regions if b - a >= min_len]
