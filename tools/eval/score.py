"""Score audio with Meta's audiobox-aesthetics model (an optional dev tool, not a runtime dependency).

Prints content enjoyment (CE), content usefulness (CU), production complexity (PC) and
production quality (PQ), roughly on a 1-10 scale. Long files are scored in 20 s chunks;
the mean and the worst chunk are shown.

    python -m venv .evalenv
    .evalenv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    .evalenv/bin/pip install audiobox_aesthetics requests
    .evalenv/bin/python tools/eval/score.py music.wav [more.wav ...]
"""

import os
import sys
import wave

import numpy as np
import torch
from audiobox_aesthetics.infer import initialize_predictor

CHUNK = 20.0


def load(path):
    with wave.open(path) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = (((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8) >> 8).astype(np.float32) / 2**23
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    return x.reshape(-1, ch).T, sr


def main(paths):
    predictor = initialize_predictor()
    for path in paths:
        x, sr = load(path)
        n = int(CHUNK * sr)
        items = [{"path": torch.from_numpy(np.ascontiguousarray(x[:, i:i + n])), "sample_rate": sr}
                 for i in range(0, max(1, x.shape[1] - n // 2), n)]
        res = []
        for k in range(0, len(items), 4):
            res += predictor.forward(items[k:k + 4])
        keys = list(res[0])
        mean = {k: np.mean([r[k] for r in res]) for k in keys}
        worst = {k: np.min([r[k] for r in res]) for k in keys}
        if os.environ.get("CHUNKS"):
            print("   CE per chunk:", " ".join(f"{r['CE']:.2f}" for r in res))
        print(f"{os.path.basename(path):34s} " + "  ".join(f"{k}={mean[k]:.2f} (min {worst[k]:.2f})" for k in keys),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
