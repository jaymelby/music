"""Turn a cue sheet and a style into a symbolic score.

The score is a list of note, drum-hit and effect events positioned in *beats*;
the beat grid (see ``timing``) maps beats to seconds so that every scene
transition falls on a downbeat.

Musical rules that keep the result pleasant:

* harmony comes from curated progressions, voiced with minimal voice movement;
* each scene is a section; long scenes are split into 8-bar phrases;
* energy decides which layers play and how busy drums/bass/arps are;
* melodies are short motifs that repeat with variation, snapping to chord
  tones on strong beats and on any note held a quarter note or longer;
* transitions are marked with crashes, risers, swells, fills and breaks.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from .spec import Spec
from .styles import STYLES, Layer, Style
from .theory import MINOR_LIKE, Chord, Key, nearest_pitch, parse_key, roman_chord, scale_step, voice_chord
from .timing import BEATS_PER_BAR, Bar, Cue, Grid, fit_grid

STEPS_PER_BEAT = 4
DRUM_VEL = {"X": 1.0, "x": 0.8, "o": 0.55, "g": 0.35}


@dataclass
class Note:
    layer: str
    beat: float
    length: float  # beats
    pitch: int
    vel: float


@dataclass
class Hit:
    instrument: str
    beat: float
    vel: float


@dataclass
class Fx:
    kind: str  # crash | riser | swell | impact
    beat: float  # the moment the effect lands (risers and swells end here)
    length: float  # beats (risers / swells)
    vel: float


@dataclass
class Section:
    index: int
    label: str
    energy: float
    bars: list[Bar]
    part: str = "A"
    auto_energy: bool = False

    @property
    def beat(self) -> int:
        return self.bars[0].beat

    @property
    def end_beat(self) -> int:
        return self.bars[-1].beat + self.bars[-1].beats


@dataclass
class Phrase:
    section: Section
    bars: list[Bar]
    energy: float
    index: int  # index within section
    count: int  # phrases in section
    tier: str
    layers: list[Layer]
    part: str = "A"  # "A" (verse-like) or "B" (chorus-like) progression


@dataclass
class Score:
    style: Style
    key: Key
    grid: Grid
    notes: list[Note] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    fx: list[Fx] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    chords: list[tuple[int, int, Chord]] = field(default_factory=list)  # (beat, beats, chord)
    final_beat: float | None = None  # downbeat of the ring-out bar (resolve ending)
    warnings: list[str] = field(default_factory=list)


def energy_curve(x: float) -> float:
    """Default energy arc: gentle intro, peak around 70% of the video, calm outro."""
    x = min(max(x, 0.0), 1.0)
    return 0.3 + 0.55 * math.sin(math.pi * x ** 1.943)


def parse_rhythm(pattern: str, steps: int) -> list[tuple[int, int, str]]:
    """Parse a step pattern into (step, length, token) onsets."""
    pattern = pattern[:steps]
    out = []
    for i, ch in enumerate(pattern):
        if ch in "-.":
            continue
        n = 1
        while i + n < len(pattern) and pattern[i + n] == "-":
            n += 1
        out.append((i, n, ch))
    return out


def _tier(style: Style, energy: float) -> str:
    lo, hi = style.tiers
    return "low" if energy < lo else "mid" if energy < hi else "high"


def _pick(options, index: int):
    return options[index % len(options)] if options else None


class Composer:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.style = STYLES[spec.style]
        self.rng = np.random.default_rng(spec.seed)
        st = self.style

        if spec.key:
            self.key = parse_key(spec.key)
        else:
            weights = np.array([0.6] + [0.4 / max(1, len(st.modes) - 1)] * (len(st.modes) - 1))
            mode = st.modes[int(self.rng.choice(len(st.modes), p=weights / weights.sum()))]
            tonic = st.keys[int(self.rng.integers(len(st.keys)))]
            self.key = parse_key(f"{tonic} {mode}")
        family = "minor" if self.key.mode in MINOR_LIKE else "major"
        self.warnings: list[str] = []
        bank = st.progressions.get(self.key.mode)
        if bank is None:
            # Progressions are written for a specific mode; fall back to plain major/minor
            # rather than reading major-key numerals against, say, a lydian scale.
            if self.key.mode != family:
                self.warnings.append(f"The {st.name} style has no {self.key.mode} progressions; "
                                     f"using {Key(self.key.tonic, family).name} instead.")
                self.key = Key(self.key.tonic, family)
            bank = st.progressions.get(family)
        if bank is None:
            from .styles import MINOR_BANK

            bank = MINOR_BANK if family == "minor" else STYLES["corporate"].progressions["major"]
        self.prog_pair = bank[int(self.rng.integers(len(bank)))]
        cadence = st.cadence.get(self.key.mode) or st.cadence.get(family) or ("V" if family == "major" else "VII")
        self.cadence_symbol = cadence
        self.tonic_symbol = "i" if family == "minor" else "I"
        if st.name == "lofi":
            self.tonic_symbol += "7" if family == "minor" else "maj7"
        elif st.name == "ambient":
            self.tonic_symbol += "add9"

        if spec.tempo:
            tempo_range, preferred = (spec.tempo * 0.96, spec.tempo * 1.04), spec.tempo
        else:
            tempo_range, preferred = st.tempo, st.preferred_tempo
        cues = [Cue(sc.start, "section", sc.label) for sc in spec.scenes[1:]]
        for h in spec.hits:
            if all(abs(h - c.time) > 0.05 for c in cues):
                cues.append(Cue(h, "hit"))
        self.grid = fit_grid(cues, spec.duration, tempo_range, preferred, full_last_bar=spec.ending == "resolve")
        self.score = Score(style=st, key=self.key, grid=self.grid, warnings=self.warnings + self.grid.warnings)
        self.motifs: dict = {}
        self.prev_voicing: dict[str, tuple[int, ...]] = {}
        self.prev_bass: int | None = None
        self.prev_melody: dict[str, int] = {}

    # ------------------------------------------------------------------ structure
    def build_sections(self) -> None:
        spec, grid = self.spec, self.grid
        bars = grid.bars()
        scene_by_start = {round(sc.start, 6): sc for sc in spec.scenes}
        sections: list[Section] = []
        for bar in bars:
            seg = grid.segments[bar.segment]
            opens = bar.beat == seg.beat_offset and seg.kind in ("start", "section")
            if opens or not sections:
                sc = scene_by_start.get(round(seg.start, 6))
                label = sc.label if sc else seg.label
                energy = sc.energy if sc else None
                sections.append(Section(index=len(sections), label=label or f"scene {len(sections) + 1}",
                                        energy=-1.0 if energy is None else energy, bars=[],
                                        auto_energy=energy is None))
            sections[-1].bars.append(bar)
        total = grid.end
        for s in sections:
            if s.auto_energy:
                mid = (grid.time_of(s.beat) + grid.time_of(s.end_beat)) / 2
                s.energy = energy_curve(mid / total)
        self.score.sections = sections

    def build_phrases(self) -> None:
        st, grid = self.style, self.grid
        total = grid.end
        for s in self.score.sections:
            bars = s.bars
            if self.spec.ending == "resolve" and s is self.score.sections[-1]:
                bars = bars[:-1]  # the final bar is the ring-out, handled by write_ending
                if not bars:
                    continue
            chunks = []
            full = len(bars)
            if full <= 12:
                chunks = [bars]
            else:
                i = 0
                while i < full:
                    take = 8 if full - i - 8 >= 4 else full - i
                    chunks.append(bars[i:i + take])
                    i += take
            for j, chunk in enumerate(chunks):
                if s.auto_energy:
                    mid = (grid.time_of(chunk[0].beat) + grid.time_of(chunk[-1].beat + chunk[-1].beats)) / 2
                    e = energy_curve(mid / total)
                else:
                    ramp = 0.0 if s is self.score.sections[-1] else min(0.1, 0.05 * j)
                    e = min(1.0, s.energy + ramp)
                layers = [ly for ly in st.layers if ly.min_energy <= e < ly.max_energy]
                if not layers:
                    layers = [st.layers[0]]
                self.score.phrases.append(Phrase(section=s, bars=chunk, energy=e, index=j, count=len(chunks),
                                                 tier=_tier(st, e), layers=layers))
        for s in self.score.sections:
            ph = [p for p in self.score.phrases if p.section is s]
            peak = max((p.energy for p in ph), default=s.energy)
            s.part = "B" if peak >= 0.6 else "A"
            for p in ph:
                # a scene keeps one progression; a long scene with an automatic energy arc
                # alternates verse/chorus material as its energy rises and falls
                p.part = ("B" if p.energy >= 0.6 else "A") if s.auto_energy else s.part

    # ------------------------------------------------------------------ harmony
    def build_chords(self) -> dict[int, Chord]:
        """Chord for every bar (keyed by downbeat)."""
        st, key = self.style, self.key
        by_bar: dict[int, Chord] = {}
        prev_part = None
        full_idx = 0
        for ph in self.score.phrases:
            prog = list(self.prog_pair[0] if ph.part == "A" else self.prog_pair[1])
            bpc = st.bars_per_chord[0 if ph.energy < 0.55 else 1]
            if ph.part == prev_part:
                full_idx = -(-full_idx // bpc) * bpc  # same part again: carry on from the next chord
            else:
                full_idx = 0
            prev_part = ph.part
            last: Chord | None = None
            for bar in ph.bars:
                if bar.beats < BEATS_PER_BAR and last is not None:
                    by_bar[bar.beat] = last
                    continue
                last = roman_chord(prog[(full_idx // bpc) % len(prog)], key)
                by_bar[bar.beat] = last
                full_idx += 1
        if self.spec.ending == "resolve":
            last_s = self.score.sections[-1]
            full_bars = [b for b in last_s.bars if b.beats == BEATS_PER_BAR]
            final = last_s.bars[-1]
            by_bar[final.beat] = roman_chord(self.tonic_symbol, key)
            self.score.final_beat = final.beat
            if len(full_bars) >= 3:
                pen = [b for b in last_s.bars if b.beat < final.beat][-1]
                by_bar[pen.beat] = roman_chord(self.cadence_symbol, key)
        self.chord_by_bar = by_bar
        bars = self.grid.bars()
        self.bar_starts = [b.beat for b in bars]
        for b in bars:
            self.score.chords.append((b.beat, b.beats, by_bar[b.beat]))
        return by_bar

    def chord_at(self, beat: float) -> Chord:
        i = max(0, bisect.bisect_right(self.bar_starts, beat) - 1)
        return self.chord_by_bar[self.bar_starts[i]]

    # ------------------------------------------------------------------ helpers
    def swing(self, step: int) -> float:
        """Beat offset for a 16th step (swing delays the odd 16ths)."""
        return (self.style.swing - 0.5) * 0.5 if step % 2 == 1 else 0.0

    def jitter(self, amount: float = 0.04) -> float:
        return float(self.rng.uniform(-amount, amount))

    def add(self, layer: Layer, beat: float, length: float, pitch: int, vel: float) -> None:
        self.score.notes.append(Note(layer.name, beat, length, int(pitch), float(min(1.0, max(0.05, vel)))))

    # ------------------------------------------------------------------ layers
    def write_pad(self, ly: Layer, ph: Phrase, open_notes: dict) -> None:
        vel = 0.45 + 0.4 * ph.energy
        spans: list[tuple[int, int, Chord]] = []
        for bar in ph.bars:
            ch = self.chord_by_bar[bar.beat]
            if spans and spans[-1][2] is ch and spans[-1][0] + spans[-1][1] == bar.beat:
                spans[-1] = (spans[-1][0], spans[-1][1] + bar.beats, ch)
            else:
                spans.append((bar.beat, bar.beats, ch))
        for beat, beats, ch in spans:
            v = voice_chord(ch, self.prev_voicing.get(ly.name), ly.range[0], ly.range[1], ly.voices)
            self.prev_voicing[ly.name] = v
            still_open = {}
            for p in v:
                prev = open_notes.get(p)
                if prev is not None and abs(prev.beat + prev.length - beat) < 1e-6:
                    prev.length += beats  # common tone: tie it over instead of re-attacking
                    still_open[p] = prev
                else:
                    n = Note(ly.name, beat, beats, p, vel + self.jitter(0.03))
                    self.score.notes.append(n)
                    still_open[p] = n
            open_notes.clear()
            open_notes.update(still_open)

    def write_keys(self, ly: Layer, ph: Phrase, variation: int) -> None:
        pattern = _pick(ly.patterns.get(ph.tier) or ly.patterns.get("mid"), variation)
        for bar in ph.bars:
            ch = self.chord_by_bar[bar.beat]
            v = voice_chord(ch, self.prev_voicing.get(ly.name), ly.range[0], ly.range[1], ly.voices)
            self.prev_voicing[ly.name] = v
            for k, (step, n, _tok) in enumerate(parse_rhythm(pattern, bar.beats * STEPS_PER_BEAT)):
                beat = bar.beat + step / STEPS_PER_BEAT + self.swing(step)
                vel = (0.72 if k == 0 else 0.58) * (0.8 + 0.3 * ph.energy)
                for p in v:
                    self.add(ly, beat, n / STEPS_PER_BEAT * 0.95, p, vel + self.jitter(0.03))

    def write_arp(self, ly: Layer, ph: Phrase, variation: int) -> None:
        pattern = _pick(ly.patterns.get(ph.tier) or ly.patterns.get("mid"), variation)
        rate = ly.arp_rate.get(ph.tier, 2)
        counter = 0
        lo, hi = ly.range
        for bar in ph.bars:
            ch = self.chord_by_bar[bar.beat]
            base = sorted(nearest_pitch(pc, lo + 6, lo, hi) for pc in ch.pcs)
            pool = list(base)
            octave = 12
            while len(pool) < 8:
                nxt = [p + octave for p in base if p + octave <= hi + 7]
                if not nxt:
                    break
                pool += nxt
                octave += 12
            pool = sorted(set(pool))
            for step in range(0, bar.beats * STEPS_PER_BEAT, rate):
                idx = pattern[counter % len(pattern)]
                counter += 1
                p = pool[idx % len(pool)]
                beat = bar.beat + step / STEPS_PER_BEAT + (self.swing(step) if rate == 1 else 0.0)
                accent = 0.82 if step % 4 == 0 else 0.62
                self.add(ly, beat, rate / STEPS_PER_BEAT, p, accent * (0.8 + 0.3 * ph.energy) + self.jitter(0.04))

    def write_bass(self, ly: Layer, ph: Phrase, variation: int, mute_from: float | None) -> None:
        pattern = _pick(ly.patterns.get(ph.tier) or ly.patterns.get("mid"), variation)
        lo, hi = ly.range
        for bar in ph.bars:
            ch = self.chord_by_bar[bar.beat]
            target = self.prev_bass if self.prev_bass is not None else lo + 5
            root = nearest_pitch(ch.root, target, lo, hi)
            if self.prev_bass is not None and abs(root - self.prev_bass) > 7:
                root = nearest_pitch(ch.root, (lo + hi) / 2, lo, hi)
            self.prev_bass = root
            next_beat = bar.beat + bar.beats
            next_root = self.chord_at(next_beat).root if next_beat < self.grid.total_beats else ch.root
            for k, (step, n, tok) in enumerate(parse_rhythm(pattern, bar.beats * STEPS_PER_BEAT)):
                beat = bar.beat + step / STEPS_PER_BEAT + self.swing(step)
                if mute_from is not None and beat >= mute_from - 1e-6:
                    continue
                if tok == "5":
                    p = root + 7 if root + 7 <= hi + 5 else root - 5
                elif tok == "8":
                    p = root + 12
                elif tok == "3":
                    p = root + (ch.intervals[1] if len(ch.intervals) > 1 else 4)
                elif tok == "A":
                    nr = nearest_pitch(next_root, root, lo, hi + 5)
                    p = scale_step(nr, -1, self.key) if nr >= root else scale_step(nr, 1, self.key)
                else:
                    p = root
                length = n / STEPS_PER_BEAT * 0.92
                if mute_from is not None:
                    length = min(length, mute_from - beat)
                vel = (0.85 if k == 0 else 0.72) * (0.85 + 0.2 * ph.energy)
                self.add(ly, beat, length, p, vel + self.jitter(0.03))

    def _motif(self, ly: Layer, ph: Phrase) -> tuple:
        dense = ph.energy >= 0.72
        key = (ly.name, ph.part, dense)
        if key not in self.motifs:
            pool = self.style.melody_rhythms.get("dense" if dense else "sparse") or ["x-------x-------x-------x-------"]
            rhythm = pool[int(self.rng.integers(len(pool)))]
            onsets = parse_rhythm(rhythm, 32)
            moves = [int(m) for m in self.rng.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3], size=len(onsets))]
            moves[0] = 0
            alt = list(moves)
            for i in range(max(1, len(alt) * 2 // 3), len(alt)):
                alt[i] = int(self.rng.choice([-2, -1, 1, 2]))
            self.motifs[key] = (onsets, moves, alt)
        return self.motifs[key]

    def write_melody(self, ly: Layer, ph: Phrase) -> None:
        onsets, moves, alt = self._motif(ly, ph)
        lo, hi = ly.range
        prev = self.prev_melody.get(ly.name)
        if prev is None:
            prev = nearest_pitch(self.key.tonic, (lo + hi) / 2 - 2, lo, hi)
        bars = ph.bars
        units = [bars[i:i + 2] for i in range(0, len(bars), 2)]
        for u, unit in enumerate(units):
            steps_avail = sum(b.beats for b in unit) * STEPS_PER_BEAT
            first_bar_steps = unit[0].beats * STEPS_PER_BEAT
            mv = moves if u % 2 == 0 else alt
            last_unit = u == len(units) - 1
            written: list[Note] = []
            for (step, n, _), m in zip(onsets, mv):
                if step >= steps_avail:
                    break
                n = min(n, steps_avail - step)
                beat = unit[0].beat + step / STEPS_PER_BEAT + self.swing(step)
                ch = self.chord_at(beat)
                p = scale_step(prev, m, self.key) if m else prev
                in_bar = step if step < first_bar_steps else step - first_bar_steps
                if in_bar % 8 == 0 or n >= 4:  # beats 1 and 3 of the bar, or a held note
                    p = nearest_pitch(min(ch.triad_pcs, key=lambda pc: abs(nearest_pitch(pc, p) - p)), p)
                while p > hi:
                    p -= 12
                while p < lo:
                    p += 12
                vel = (0.75 if step % 16 == 0 else 0.62) * (0.85 + 0.2 * ph.energy) + self.jitter(0.04)
                note = Note(ly.name, beat, n / STEPS_PER_BEAT * 0.9, int(p), vel)
                written.append(note)
                prev = p
            if last_unit and written:
                end = written[-1]
                ch = self.chord_at(end.beat)
                stable = [ch.root, (ch.root + ch.intervals[1]) % 12]
                end.pitch = min((nearest_pitch(pc, end.pitch, lo, hi) for pc in stable), key=lambda q: abs(q - end.pitch))
                unit_end = unit[-1].beat + unit[-1].beats
                end.length = max(end.length, (unit_end - end.beat) * 0.95)
                prev = end.pitch
            self.score.notes.extend(written)
        self.prev_melody[ly.name] = prev

    # ------------------------------------------------------------------ drums
    def write_drums(self, ph: Phrase, variation: int, fill: bool, mute_from: float | None) -> None:
        st = self.style
        if ph.energy < st.drum_min_energy:
            return
        level = 0.8 + 0.25 * ph.energy
        for bar in ph.bars:
            steps = bar.beats * STEPS_PER_BEAT
            is_last = bar is ph.bars[-1]
            fill_inst = st.fx.get("fill") if (fill and is_last) else None
            fill_steps = 8 if fill_inst and fill_inst.startswith("tom") else 4
            fill_from = steps - fill_steps
            for part in st.drums:
                pattern = _pick(part.patterns.get(ph.tier) or [""], variation)
                if not pattern:
                    continue
                for step, ch in enumerate(pattern[:steps]):
                    if ch not in DRUM_VEL:
                        continue
                    if fill_inst and step >= fill_from and part.instrument in (fill_inst, "hat", "ohat", "shaker"):
                        continue
                    beat = bar.beat + step / STEPS_PER_BEAT + self.swing(step)
                    if mute_from is not None and beat >= mute_from - 1e-6:
                        continue
                    self.score.hits.append(Hit(part.instrument, beat, DRUM_VEL[ch] * level + self.jitter(0.03)))
            if fill_inst:
                for i in range(fill_steps):
                    step = fill_from + i
                    beat = bar.beat + step / STEPS_PER_BEAT
                    if mute_from is not None and beat >= mute_from - 1e-6:
                        break
                    vel = (0.45 + 0.55 * (i + 1) / fill_steps) * level
                    self.score.hits.append(Hit(fill_inst, beat, vel))

    # ------------------------------------------------------------------ transitions
    def write_transitions(self) -> None:
        st, grid = self.style, self.grid
        sections = self.score.sections
        for i, s in enumerate(sections):
            if i == 0:
                continue
            prev = sections[i - 1]
            prev_e = max((p.energy for p in self.score.phrases if p.section is prev), default=prev.energy)
            jump = s.energy - prev_e
            beat = s.beat
            beat_len = grid.time_of(beat) - grid.time_of(beat - 1)
            if st.fx.get("crash") is not None and s.energy >= st.fx["crash"]:
                self.score.fx.append(Fx("crash", beat, 0, 0.55 + 0.4 * s.energy))
            if st.fx.get("impact") is not None and s.energy >= st.fx["impact"]:
                self.score.fx.append(Fx("impact", beat, 0, 0.6 + 0.4 * s.energy))
            prev_beats = prev.end_beat - prev.beat
            if st.fx.get("riser") and jump >= 0.18 and s.energy >= 0.55:
                length = min(8, prev_beats)
                self.score.fx.append(Fx("riser", beat, length, 0.5 + 0.5 * s.energy))
            elif st.fx.get("swell"):
                length = float(min(prev_beats, max(2, round(1.8 / beat_len / 2) * 2)))
                self.score.fx.append(Fx("swell", beat, length, 0.35 + 0.4 * max(s.energy, prev_e)))
        for seg in grid.segments:
            if seg.kind != "hit":
                continue
            beat = seg.beat_offset
            kind = st.fx.get("hit", "crash")
            if kind == "crash":
                self.score.fx.append(Fx("crash", beat, 0, 0.8))
                self.score.hits.append(Hit("kick", beat, 0.9))
            elif kind == "impact":
                self.score.fx.append(Fx("impact", beat, 0, 0.85))
            else:
                self.score.fx.append(Fx("swell", beat, 4, 0.6))

    def write_ending(self, last_layers: list[Layer], last_energy: float) -> None:
        if self.score.final_beat is None:
            return
        st = self.style
        fb = self.score.final_beat
        bar = [b for b in self.grid.bars() if b.beat == fb][0]
        ring = bar.beats
        tonic = self.chord_by_bar[fb]
        names = {ly.name for ly in last_layers}
        harmonic = [ly for ly in st.layers if ly.role in ("pad", "keys") and ly.name in names]
        if not harmonic:
            harmonic = [ly for ly in st.layers if ly.role in ("pad", "keys")][:1]
        for ly in harmonic:
            v = voice_chord(tonic, self.prev_voicing.get(ly.name), ly.range[0], ly.range[1], ly.voices)
            for p in v:
                self.add(ly, fb, ring, p, 0.55 + 0.3 * last_energy)
        for ly in st.layers:
            if ly.name not in names:
                continue
            if ly.role == "bass":
                self.add(ly, fb, ring, nearest_pitch(tonic.root, self.prev_bass or ly.range[0] + 5, *ly.range), 0.8)
            elif ly.role == "melody":
                prev = self.prev_melody.get(ly.name, (ly.range[0] + ly.range[1]) // 2)
                self.add(ly, fb, ring, nearest_pitch(tonic.root, prev, *ly.range), 0.65)
        if last_energy >= st.drum_min_energy:
            if any(d.instrument.startswith("kick") for d in st.drums):
                kick = next(d.instrument for d in st.drums if d.instrument.startswith("kick"))
                self.score.hits.append(Hit(kick, fb, 0.85))
            if st.fx.get("crash") is not None:
                self.score.fx.append(Fx("crash", fb, 0, 0.6))
        if st.fx.get("impact") is not None:
            self.score.fx.append(Fx("impact", fb, 0, 0.7))

    # ------------------------------------------------------------------ main
    def compose(self) -> Score:
        self.build_sections()
        self.build_phrases()
        self.build_chords()
        st = self.style
        sections = self.score.sections
        open_pads: dict[str, dict] = {}
        last_layers: list[Layer] = []
        last_energy = 0.3
        for ph in self.score.phrases:
            s = ph.section
            if ph.index == 0:
                open_pads = {}
            variation = s.index + ph.index
            is_section_end = ph.index == ph.count - 1
            nxt = sections[s.index + 1] if s.index + 1 < len(sections) else None
            fill = bool(st.fx.get("fill")) and nxt is not None and is_section_end
            mute_from = None
            if (st.fx.get("break") and nxt is not None and is_section_end and len(s.bars) >= 2
                    and nxt.energy - ph.energy >= 0.2 and nxt.energy >= 0.55):
                mute_from = nxt.beat - 2  # drop drums and bass for the last two beats
                fill = False
            for ly in ph.layers:
                if ly.role == "pad":
                    self.write_pad(ly, ph, open_pads.setdefault(ly.name, {}))
                elif ly.role == "keys":
                    self.write_keys(ly, ph, variation)
                elif ly.role == "arp":
                    self.write_arp(ly, ph, variation)
                elif ly.role == "bass":
                    self.write_bass(ly, ph, variation, mute_from)
                elif ly.role == "melody":
                    if ph.count >= 3 and ph.index == 0:
                        continue  # let long sections breathe before the tune comes in
                    self.write_melody(ly, ph)
            self.write_drums(ph, variation, fill, mute_from)
            last_layers, last_energy = ph.layers, ph.energy
        self.write_transitions()
        self.write_ending(last_layers, last_energy)
        self.score.notes.sort(key=lambda n: n.beat)
        self.score.hits.sort(key=lambda h: h.beat)
        return self.score


def compose(spec: Spec) -> Score:
    return Composer(spec).compose()
