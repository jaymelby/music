"""Print the spectral balance of WAV files (share of energy per band, dB) and crest factor."""

import sys
import wave

import numpy as np
from scipy import signal

EDGES = [20, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000]


def load(path):
    with wave.open(path) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return x.reshape(-1, ch).mean(axis=1), sr


print(f"{'file':32s}" + "".join(f"{a}-{b}".rjust(12) for a, b in zip(EDGES, EDGES[1:])) + "  crest")
for path in sys.argv[1:]:
    x, sr = load(path)
    f, p = signal.welch(x, sr, nperseg=8192)
    row = [10 * np.log10(p[(f >= a) & (f < b)].sum() / p.sum() + 1e-12) for a, b in zip(EDGES, EDGES[1:])]
    crest = 20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x**2)))
    print(f"{path.split('/')[-1][:32]:32s}" + "".join(f"{v:12.1f}" for v in row) + f"  {crest:5.1f}")
