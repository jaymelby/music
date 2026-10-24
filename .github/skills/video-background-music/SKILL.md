---
name: video-background-music
description: Generate royalty-free background music for a video, timed so every scene transition lands exactly on a musical downbeat. Use after creating or editing a video (Remotion, MoviePy, Manim, ffmpeg, etc.) when it needs a soundtrack, or when asked to add, replace, re-time or sync background music.
---

# Background music for videos (videoscore)

`videoscore` composes and synthesizes original instrumental music offline: no network, no
samples, no API keys. You give it the video length and scene start times, and it writes a WAV
that is exactly that long. Each scene change falls on a downbeat, to the sample, and is
marked by an arrangement change plus a crash, swell, riser or impact.

## 1. Make sure it is installed

```bash
videoscore --help >/dev/null 2>&1 || pip install "git+https://github.com/jaymelby/music.git"
```

It only needs Python 3.9+, numpy and scipy. ffmpeg is optional (for .mp3/.m4a output and
voice-over input).

## 2. Collect exact timings from the video source

Read them from the code that builds the video. Don't guess from memory.

- **duration**: total video length in seconds. Check the final render with
  `ffprobe -v error -show_entries format=duration -of csv=p=0 out.mp4`.
- **scene starts**: the time of every cut or major transition, in seconds.
  - Remotion: each `<Sequence from={...}>` / `<Series.Sequence durationInFrames={...}>`.
    Pass frames directly with `"unit": "frames", "fps": 30`.
  - MoviePy / ffmpeg concat: cumulative clip durations. List scenes with `"duration"`
    instead of `"start"` and the engine adds them up.
- **hits** (optional): single moments inside a scene that deserve an accent, like a logo
  reveal or a number popping up. Each one also lands on a downbeat.
- **energy** per scene (optional, 0–1 or `low`/`medium`/`high`/`peak`). Leave it out to get
  an automatic arc: calm intro, build, peak around 70%, calm outro. Set it when the story
  needs something specific, such as a quiet problem statement or a big product reveal.

## 3. Pick a style

| style | feel | use for |
|---|---|---|
| `corporate` (default) | bright, plucky, light four-on-the-floor | product demos, explainers, launches |
| `ambient` | calm pads, soft piano and bells, no drums | tutorials with narration, calm explainers |
| `lofi` | jazzy chords, swung dusty drums, vinyl | dev logs, coding timelapses, casual walkthroughs |
| `cinematic` | strings, ostinato, impacts and risers | trailers, reveals, dramatic launches |
| `electronic` | sidechained synths, 16th plucks, punchy kick | promos, social clips, fast montages |

## 4. Write the cue sheet (keep it in the repo, e.g. `music/cue.json`)

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
  "ending": "resolve"
}
```

Other keys:
- `"ending"`: `"resolve"` (default) lands on a final chord that rings out; `"fade"` fades out.
- `"loudness"`: integrated LUFS (default `-18`). Use `-14` for music-only social videos (the loudest allowed is `-12`), and
  about `-22` under narration if you don't use ducking.
- `"voiceover": "narration.wav"`: the music ducks automatically while the voice is speaking.
  You can also give explicit `"duck": [[start, end], ...]` regions, with `"duck_db"`
  (default `-10`).
- `"key"` (e.g. `"D major"`), `"tempo"` (preferred BPM), `"seed"` (a different seed gives a
  different piece; the same seed gives byte-identical output).

Run `videoscore example` for a template and `videoscore styles` for the list of styles.

## 5. Render

```bash
videoscore render music/cue.json -o music/bgm.wav
```

Check the summary it prints:
- every cue should read `-> downbeat of bar N (error 0.000 ms)`;
- `NOT SYNCED` means two cues are less than one bar (about 2 s) apart. Merge them, or move
  the video cut onto a nearby time from `downbeats` in `music/bgm.wav.json`;
- a `Tempo changes N%` warning means a scene is much shorter or longer than its neighbours
  in musical terms. It is usually fine; nudging that cut by a fraction of a second removes it.

The JSON report (`<out>.json`) also contains `beats`, `downbeats`, `sections`, `chords` and
`tempo_map`. Use them if you want animations to hit the beat as well.

To preview timing without rendering, run `videoscore plan music/cue.json`.

## 6. Put it on the video

Video without audio:
```bash
ffmpeg -y -i video.mp4 -i music/bgm.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest final.mp4
```

Video that already has narration: duck the music first (`"voiceover"` in the cue sheet), then mix.
```bash
ffmpeg -y -i video.mp4 -vn -ac 1 -ar 48000 narration.wav   # if the narration only exists inside the video
ffmpeg -y -i video.mp4 -i music/bgm.wav -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k final.mp4
```

Remotion: copy the file to `public/bgm.wav` and add `<Audio src={staticFile("bgm.wav")} />`
at the top level of the composition, so it starts at frame 0.

## Rules

- Don't trim, time-stretch, loop or re-fade the output. It is already exactly the right length,
  with its own ending.
- If you change any scene timing, update the cue sheet and render again (it takes seconds).
- Don't put background music at the same level as speech. Use `voiceover`/`duck` or a lower
  `loudness`.
- Commit the cue sheet. Commit the audio only if the repo already stores media.
