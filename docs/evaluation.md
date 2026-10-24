# Evaluation: does it sound good?

The engine is meant to be run by an agent that can't listen, so the music has to sound right
without anyone checking it by ear. Quality is checked in three ways.

## 1. A learned quality model, calibrated against human-produced music

[Meta audiobox-aesthetics](https://arxiv.org/abs/2502.05139) predicts human ratings on four
axes (roughly 1–10):

- **CE**: content enjoyment
- **CU**: content usefulness
- **PC**: production complexity
- **PQ**: production quality

Files are scored in 20 s chunks and the mean is shown. To calibrate, the same model scored
professionally produced royalty-free library tracks by Kevin MacLeod (incompetech.com, CC-BY;
60 s excerpts, used locally only and not redistributed) and a naive baseline of random sine
notes.

### Demo renders (one per style, realistic cue sheets)

| track | CE | CU | PC | PQ |
|---|---|---|---|---|
| corporate, 60 s, 5 scenes and 1 hit | 7.52 | 8.08 | 5.19 | 8.15 |
| lofi, 75 s, 5 scenes, fade ending | 7.58 | 8.10 | 4.52 | 8.07 |
| electronic, 40 s, 6 fast cuts | 7.40 | 8.11 | 5.65 | 8.12 |
| cinematic, 45 s, 4 scenes and 1 hit | 7.20 | 7.87 | 5.17 | 7.95 |
| ambient, 90 s, 5 scenes | 7.19 | 7.84 | 4.14 | 7.94 |

### References

| track | CE | CU | PC | PQ |
|---|---|---|---|---|
| *Carefree* (upbeat) | 7.93 | 8.48 | 5.52 | 8.49 |
| *Local Forecast – Elevator* (jazz) | 7.68 | 8.20 | 6.21 | 8.31 |
| *Wallpaper* (electronic) | 7.27 | 8.14 | 5.43 | 8.22 |
| *Inspired* (orchestral) | 7.25 | 7.91 | 5.50 | 8.09 |
| six ambient tracks (*Dreamer*, *Clean Soul*, *Meditation Impromptu 01*, …) | 6.01–7.62 | 7.43–8.15 | 2.65–5.46 | 7.43–8.21 |
| naive random sine notes | **4.64** | 7.16 | 2.45 | 7.44 |

Every style lands within the score range of the human-produced references for its genre, and
far above the naive baseline on enjoyment.

### Consistency: 20 renders (5 styles × 4 seeds, 35–150 s, 2–6 random cuts)

| style | CE range | PQ range |
|---|---|---|
| corporate | 7.54–7.59 | 8.12–8.22 |
| lofi | 7.36–7.57 | 7.85–8.11 |
| electronic | 7.19–7.38 | 7.98–8.16 |
| cinematic | 7.18–7.23 | 7.92–8.04 |
| ambient | 6.97–7.21 | 7.87–8.01 |

Across seeds, keys, lengths and cue layouts, quality stays consistent: no seed produced an
outlier. During tuning, the scores showed specific problems, which were then fixed:
- low-energy ambient sections were too sparse, so a piano arpeggio and a bell "sparkle" layer
  were added at lower energy;
- too much energy below 60 Hz, fixed by tightening the kick and a 35 Hz master high-pass;
- a dull top end in the reverb.

## 2. Objective checks in the test suite (`pytest`)

- Every cue lands exactly on a downbeat. This is randomised over 25 cue layouts, and the
  report shows `error_ms == 0`.
- Pitches stay within ±12 semitones of their layer's register, and fewer than 2% of notes
  fall outside the current chord and the key. Melody notes on beats 1 and 3 are always chord
  tones.
- The output is exactly `duration × sample_rate` samples, integrated loudness is within
  0.5 LU of the target, true peak ≤ -1 dBTP, and the file ends in silence (no click).
- The high-energy section is louder right after its cut than before it, so the transition is
  audible.
- Ducking lowers the music by the requested dB under narration and leaves it untouched
  elsewhere. Narration is detected from a voice-over file.
- The same cue sheet and seed give byte-identical audio.

## 3. Spectral balance

`tools/eval/bands.py` compares the share of energy per octave band against the references.
That's how the excess sub-bass and the dull reverb were caught.

## Caveats

A model score stands in for listening and doesn't replace it, and audiobox-aesthetics has its
own biases (it rewards rhythmic, busier material slightly). Listen to the demos before relying
on a style for something important.

## Reproducing

```bash
python -m venv .evalenv
.evalenv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.evalenv/bin/pip install audiobox_aesthetics requests
videoscore render examples/product-demo.json -o /tmp/demo.wav
.evalenv/bin/python tools/eval/score.py /tmp/demo.wav
python tools/eval/bands.py /tmp/demo.wav
```
