# Research: generating background music for videos programmatically

**Goal.** A GitHub Copilot coding agent finishes a video (30–300 s) and needs a soundtrack that
(1) is exactly as long as the video, (2) changes on the scene transitions, (3) doesn't sound
cheap, and (4) can be produced unattended inside the agent's sandbox.

**Constraints that come from running inside Copilot's coding agent**

- It runs on an ephemeral Ubuntu GitHub Actions runner, with no GPU by default.
- The agent's own shell commands go through a firewall with an allowlist. Package registries
  such as PyPI and npm are on the default list. Model hubs and arbitrary asset hosts usually
  aren't. Commands in `copilot-setup-steps.yml` run *before* the firewall applies.
  ([firewall](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/customize-the-agent-firewall),
  [setup steps](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/customize-the-agent-environment))
- The agent can't listen to the result, so quality has to come from the method, not from
  trial and error.
- It can load repo-provided **skills** from `.github/skills/<name>/SKILL.md`. That's the
  natural way to teach it a tool.
  ([skills](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-skills))

## Options surveyed

### A. Symbolic composition libraries (they produce notes, not sound)

| | Python | Node |
|---|---|---|
| Theory / analysis | music21, musicaiz, MusPy | tonal |
| Write MIDI | mido, pretty_midi, MIDIUtil, miditoolkit | midi-writer-js, @tonejs/midi, scribbletune |

These are handy for the note-level work, but none of them make sound. Each needs a renderer
(B), and none of them deal with fitting music to cue points.

### B. Turning notes into audio

| Approach | Quality | Headless CI | Notes |
|---|---|---|---|
| SoundFont via FluidSynth (`pyfluidsynth`, `midi2audio`, `fluidsynth` CLI) | General MIDI sounds dated ("MIDI file" sound); a good `.sf2` helps | needs an apt package and a 30–200 MB soundfont download (firewall) | easiest path from MIDI to audio |
| DAW in Python: [DawDreamer](https://arxiv.org/abs/2111.09931) (VST/Faust), [Pedalboard](https://pypi.org/project/pedalboard/) (JUCE effects, hosts VST3) | as good as the plugins | Linux VST instruments are rare and need licences/downloads | Pedalboard is **GPL-3** |
| Synthesis languages: Csound, SuperCollider (supriya), pyo | high, with expertise | heavy installs, servers | steep learning curve |
| Node: Tone.js + a Web Audio implementation ([node-web-audio-api](https://cdn.jsdelivr.net/npm/node-web-audio-api@2.1.0/README.md), Rust-backed) | decent synths | workable | Tone.js is browser-first; offline rendering in Node is fragile, and JS has no scipy-grade DSP (filters, FFT convolution, resampling) |
| **numpy + scipy DSP written from scratch** | good *if* oscillators are band-limited and the mix is mastered | **2 pip wheels, no assets** | full control and deterministic, but you write the synths, mixer and mastering yourself |

### C. Local generative AI models

| Model | Licence of weights | Practical notes |
|---|---|---|
| Meta MusicGen (audiocraft) | CC-BY-NC 4.0 (non-commercial) | GPU recommended, about 30 s per generation |
| Stable Audio Open 1.0 / [Stable Audio 3.0 Small](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models) (May 2026) | Stability AI Community Licence (free under $1M revenue) | Small is about 459M parameters and makes up to 2 min; instrumental |
| ACE-Step 1.5, YuE | open weights | full songs; GPU-class hardware |

They can sound great, but they're the wrong fit here. The weights are multi-GB downloads
that the firewall blocks unless they're allowlisted or baked into setup steps. CPU inference is
slow. You can't make a section boundary land on a given frame; you can only generate pieces
and crossfade them, which breaks the musical flow. Quality varies from run to run, and the
agent can't hear the difference. Some licences don't allow commercial use.

### D. Hosted music APIs

- [ElevenLabs Music API](https://elevenlabs.io/docs/overview/capabilities/music): `POST /v1/music`,
  3 s up to several minutes, *composition plans* with named sections and durations; commercial
  use on paid plans.
- [Google Lyria 2 on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/lyria/lyria-002):
  instrumental 48 kHz clips up to about 33 s each.
- Also Stable Audio API, Mubert, Beatoven.ai and Soundraw.

These give the best raw quality, but they need API keys stored in the agent environment,
cost per render, need the firewall opened, and are non-deterministic. Section lengths are
approximate seconds, not beats locked to frames. They're worth offering as an optional
backend later, not as the default.

### E. Stock music, edited to fit

Royalty-free tracks (YouTube Audio Library, Pixabay, incompetech/CC-BY) cut and looped to the
video. Raw quality is high. But hitting arbitrary cut points means beat-tracking, re-ordering
bars and time-stretching, and every edit risks an audible seam. On top of that there's
licence and attribution bookkeeping per track, and assets to store or download.

## Decision

The engine uses **rule-based composition plus band-limited synthesis from scratch, in Python
(numpy + scipy)**, driven by a JSON cue sheet through a CLI. It's packaged as a pip-installable
tool plus a Copilot skill.

| Requirement | How this approach meets it |
|---|---|
| Exact timing | The composer owns the beat grid, so transitions are placed *by construction*, to the sample |
| Runs inside the agent sandbox | 2 wheels from PyPI, no model weights, samples, soundfonts or API keys; about 10× realtime on CPU |
| Doesn't sound terrible | Curated harmony and voice leading, motif-based melodies, band-limited oscillators, a real mix and master (see below), and an objective score comparable to human-produced library music |
| Agent-friendly | One command; same seed gives identical output; JSON report of what happened; error messages that say how to fix the cue sheet |
| Licensing | All output is generated from scratch by the engine, so there are no third-party samples or model weights involved |

Python rather than Node: scipy supplies the DSP building blocks (IIR filters, FFT
convolution, polyphase resampling) that Node lacks. A CLI makes the language irrelevant to the
caller, so a Remotion (Node) project just runs `videoscore render`.

## How the transitions are synced

1. Cue times split the video into segments.
2. Each segment gets a whole number of beats, always even and at least one bar, so every cue is
   exactly a downbeat.
3. Dynamic programming over the beat-count options chooses the combination that keeps the
   tempo inside the style's range, prefers whole bars (a 2-beat bar is allowed before a cut),
   and minimises tempo *jumps* between neighbouring segments. Jumps under a scene change
   count half as much, because the arrangement change masks them. Typical changes are 1–5%,
   which you can't hear at a scene change.
4. Cues less than one bar apart (about 2 s) are reported, with the nearest downbeat, rather
   than forced.
5. The composer works in beats, and the grid maps beats to seconds, so drums, chords,
   crashes, risers and swells all stay locked to the grid.

## How "doesn't sound terrible" is engineered

**Composition**
- Progressions are taken from proven pop, film and lo-fi patterns (I–V–vi–IV, i–VI–III–VII,
  ii–V–I with sevenths, and so on). They aren't random.
- Chords are voiced with minimal voice movement, without muddy low intervals, and common tones
  are tied.
- Melodies are 2-bar motifs that repeat with variation. On strong beats and on any note
  longer than an eighth, the pitch snaps to a chord tone, so there are no sustained clashes.
- Phrases are 4/8 bars. Energy-driven layering adds or removes instruments per scene.
  Transitions get crashes, reverse swells, risers, drum fills and pre-drop breaks. The ending
  is a cadence (V–I) that rings out.

**Sound**
- Wavetable and additive oscillators are built only from harmonics below Nyquist. Naive
  digital synths sound cheap mostly because of aliasing, and this avoids it.
- Supporting techniques: detuned unison, plucks whose upper partials decay faster, FM
  electric piano, felt piano with stretched partials, synthesized drums with several variants
  per hit, and humanized timing and velocity.

**Mix and master**
- Per-layer EQ, panning, kick-driven sidechain, a synthetic stereo convolution reverb, and a
  ping-pong delay synced to the tempo.
- Master high-pass at 35 Hz, so there's no sub rumble under the voice.
- Loudness normalised to the requested integrated LUFS (ITU-R BS.1770), then a look-ahead
  limiter at -1 dBTP.
- Optional ducking under narration, either detected from the voice-over or given as explicit
  regions.

## How quality was measured

I can't listen to the output, so I scored it with
[Meta's audiobox-aesthetics](https://arxiv.org/abs/2502.05139) model, which predicts human
ratings for *content enjoyment* (CE), *content usefulness* (CU), *production complexity* (PC)
and *production quality* (PQ) on a 1–10 scale. I calibrated it against professionally produced
royalty-free tracks (Kevin MacLeod, incompetech, CC-BY; used locally only, not redistributed)
and against a naive random-note baseline. Results are in [evaluation.md](evaluation.md).

## Limitations and possible extensions

- Acoustic instruments (guitar, real piano, orchestra) can't be convincingly synthesized from
  scratch. The styles stick to timbres synthesis does well. A future optional backend could
  render the same score through a SoundFont or a sample library when one is available.
- No vocals.
- The model-based score is a proxy for listening, not a replacement. Spot-check by ear.
