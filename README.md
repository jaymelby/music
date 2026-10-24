# videoscore

Background music for videos, composed and synthesized from scratch and **synced to your scene
transitions**. It's built to be run by a coding agent (e.g. GitHub Copilot's coding agent) right
after it renders a video.

- **Exact timing**: the output is exactly as long as the video, and every scene transition lands
  on a musical downbeat, to the sample. Transitions are marked by arrangement changes and by
  crashes, swells, risers or impacts.
- **Sounds produced, not "MIDI"**: curated harmony, voice-led chords, motif-based melodies,
  band-limited synths, reverb and sidechain, mastered to a LUFS target. On Meta's
  audiobox-aesthetics quality model it scores in the same range as professionally produced
  royalty-free library music ([evaluation](docs/evaluation.md)).
- **No dependencies worth worrying about**: numpy + scipy. No samples, soundfonts, model
  weights, network or API keys. ffmpeg is optional.
- **Deterministic**: the same cue sheet and seed give a byte-identical file.
- **Fast**: about 10× realtime on a laptop CPU (a 5-minute track renders in about 25 s).

Why this approach rather than AI models, hosted APIs, SoundFonts or stock music:
[docs/research.md](docs/research.md).

## Install

```bash
pip install "git+https://github.com/jaymelby/music.git"
# or, from a checkout:
pip install -e .
```

## Quick start

```bash
videoscore example > cue.json          # a template cue sheet
videoscore render cue.json -o music.wav
```

```
corporate in G major, base 112.5 BPM, 48.0s, ending: resolve
  cue    4.500s section -> downbeat of bar 2 (error 0.000 ms) problem
  cue   15.200s section -> downbeat of bar 7 (error 0.000 ms) demo
  cue   24.000s hit     -> downbeat of bar 11 (error 0.000 ms)
  cue   33.800s section -> downbeat of bar 16 (error 0.000 ms) results
  cue   42.000s section -> downbeat of bar 20 (error 0.000 ms) call to action
  section    0.00-4.50    energy 0.25 part A 106.7 BPM: pad
  section    4.50-15.20   energy 0.40 part A 112.2 BPM: pad, keys, bass, drums
  ...
wrote music.wav (-18.0 LUFS, true peak -3.62 dBTP, 8.79s) and music.wav.json
```

You don't need a file for simple cases:

```bash
videoscore render --duration 95 --transitions 6.5,21,48.2,80 --style lofi -o music.mp3
```

Python:

```python
from videoscore import render, plan

report = render({"duration": 60, "transitions": [8, 30, 52], "style": "cinematic"}, "music.wav")
report["downbeats"]   # times you can snap animations to
```

## Cue sheet

```json
{
  "duration": 48.0,
  "style": "corporate",
  "seed": 7,
  "scenes": [
    {"start": 0.0,  "label": "logo",    "energy": "low"},
    {"start": 4.5,  "label": "problem", "energy": 0.4},
    {"start": 15.2, "label": "demo",    "energy": "high"},
    {"start": 33.8, "label": "results", "energy": 0.7},
    {"start": 42.0, "label": "outro",   "energy": 0.45}
  ],
  "hits": [24.0],
  "ending": "resolve",
  "loudness": -18
}
```

| key | meaning |
|---|---|
| `duration` | Video length in seconds. Required, 5–900 s; the target use is 30–300 s. It can be left out if every scene has a `duration`. |
| `scenes` | List of `{start, label?, energy?}`, or `{duration, ...}` to give scene lengths instead of start times. |
| `transitions` | Shorthand: a list of scene-change times. |
| `hits` | Accents inside scenes (logo reveal, stat pop). Each also lands on a downbeat. |
| `energy` | 0–1 or `minimal`/`low`/`medium`/`high`/`peak`. Controls which instruments play and how busy they are. Omitted energies follow an automatic arc: intro → build → peak around 70% → outro. |
| `style` | `corporate`, `ambient`, `lofi`, `cinematic`, `electronic` (`videoscore styles`). |
| `seed` | Picks key, progression and motifs. Change it for a different piece. |
| `key`, `tempo` | Optional overrides, e.g. `"A minor"`, `96`. |
| `ending` | `resolve` (final chord on the last downbeat, rings out) or `fade`. |
| `loudness` | Integrated LUFS target (default -18). |
| `voiceover` | Path to narration audio. The music ducks while it speaks. |
| `duck`, `duck_db` | Explicit `[[start, end], ...]` ducking regions and depth (default -10 dB). |
| `unit`, `fps` | `"unit": "frames"` with `fps` lets every time be given in frames (Remotion). |
| `sample_rate` | 48000 (default, the video standard) or 44100. |

Unknown keys are rejected with a "did you mean" hint, so typos can't silently change the result.

## Output

- Audio: `.wav` (16-bit, dithered), or `.mp3/.m4a/.aac/.ogg/.flac/.opus` through ffmpeg.
- Report `<out>.json`:
  - `cues`: each cue's requested and aligned time, the error in ms, and its bar
  - `sections`: start and end, energy, BPM and active layers
  - `tempo_map`
  - `chords`, `beats` and `downbeats`
  - `audio`: measured LUFS and true peak, plus ducked regions
  - `warnings`
- `videoscore plan cue.json` prints the same timing information without rendering audio.

## How it works

```
cue sheet ─► spec.py      validate, normalise to seconds
          ─► timing.py    fit a beat grid: every cue on a downbeat (per-segment tempo, DP-minimised jumps)
          ─► compose.py   sections → phrases → chords, pads, keys, arps, bass, melody, drums, transition FX
          ─► mix.py       synthesize (instruments.py, drums.py), mix, reverb/delay, sidechain, duck, master (dsp.py)
          ─► audio_io.py  WAV / ffmpeg encode + JSON report (engine.py)
```

Styles are plain data in `styles.py`: tempo range, progressions, layers with energy thresholds,
16-step patterns per energy tier, and transition FX. You can add one without touching the
engine.

## Using it from GitHub Copilot's coding agent

1. Copy `.github/skills/video-background-music/` into the repository where Copilot makes
   videos. The skill tells the agent how to read timings, write the cue sheet, render, check
   sync and mux.
2. Copy `integrations/copilot/copilot-setup-steps.yml` to `.github/workflows/` in that
   repository, so videoscore and ffmpeg are installed before the agent starts. If this repo is
   private, follow the comments in that file.
3. Optionally, paste `integrations/copilot/copilot-instructions-snippet.md` into
   `.github/copilot-instructions.md`.

Putting the music on the video:

```bash
ffmpeg -y -i video.mp4 -i music.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest final.mp4
```

In Remotion, put the file in `public/` and add `<Audio src={staticFile("music.wav")} />` at
the top level of the composition.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests check:
- every cue lands exactly on a downbeat (randomised);
- tempo continuity;
- notes stay in key and register, and melody strong beats are chord tones;
- determinism;
- exact length, LUFS, true peak and a silent tail;
- the transition is audible at the cut;
- ducking depth and voice-over detection;
- CLI errors.
