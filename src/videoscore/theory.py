"""Music theory helpers: keys, scales, roman-numeral chords and voice leading."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4, "E#": 5,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

MODES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}
MINOR_LIKE = {"minor", "dorian"}


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0-11
    mode: str

    @property
    def scale(self) -> list[int]:
        return [(self.tonic + s) % 12 for s in MODES[self.mode]]

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {self.mode}"


def parse_key(text: str) -> Key:
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m|dorian|lydian|mixolydian)?\s*", text)
    if not m:
        raise ValueError(f"Can't parse key '{text}'. Use e.g. 'D major', 'F# minor', 'A dorian'.")
    name = (m.group(1) + m.group(2)).upper()
    mode = m.group(3) or "major"
    mode = {"maj": "major", "min": "minor", "m": "minor"}.get(mode, mode)
    return Key(tonic=_PC[name], mode=mode)


@dataclass(frozen=True)
class Chord:
    root: int  # pitch class
    intervals: tuple[int, ...]  # semitones above root, first is 0
    symbol: str

    @property
    def pcs(self) -> list[int]:
        return [(self.root + i) % 12 for i in self.intervals]

    @property
    def triad_pcs(self) -> list[int]:
        """Root, third (or sus tone) and fifth: the 'safe' tones for melodies."""
        return [(self.root + i) % 12 for i in self.intervals[:3]]


_NUMERALS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}
_ROMAN = re.compile(r"^(b|#)?(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)(°|o|\+)?(.*)$")


@lru_cache(maxsize=None)
def roman_chord(symbol: str, key: Key) -> Chord:
    """Build a chord from a roman numeral relative to ``key``'s mode.

    Uppercase = major, lowercase = minor, ``°`` diminished. Suffixes: 7, maj7, 9,
    add9, sus2, sus4, 6. A leading b/# alters the root by a semitone.
    """
    m = _ROMAN.match(symbol)
    if not m:
        raise ValueError(f"Bad roman numeral '{symbol}'")
    acc, numeral, qual, suffix = m.groups()
    degree = _NUMERALS[numeral.upper()]
    root = (key.tonic + MODES[key.mode][degree - 1] + {"b": -1, "#": 1, None: 0}[acc]) % 12
    minor = numeral.islower()
    if qual in ("°", "o"):
        tri = [0, 3, 6]
    elif qual == "+":
        tri = [0, 4, 8]
    else:
        tri = [0, 3, 7] if minor else [0, 4, 7]
    ext: list[int] = []
    for part in re.findall(r"maj7|maj9|add9|sus2|sus4|7|9|6", suffix):
        if part == "sus2":
            tri[1] = 2
        elif part == "sus4":
            tri[1] = 5
        elif part == "7":
            ext.append(10 if qual not in ("°", "o") else 9)
        elif part == "maj7":
            ext.append(11)
        elif part == "maj9":
            ext += [11, 14]
        elif part == "9":
            ext += [10, 14]
        elif part == "add9":
            ext.append(14)
        elif part == "6":
            ext.append(9)
    return Chord(root=root, intervals=tuple(tri + sorted(set(ext))), symbol=symbol)


def nearest_pitch(pc: int, target: float, lo: int = 0, hi: int = 127) -> int:
    """MIDI note with pitch class ``pc`` closest to ``target`` within [lo, hi]."""
    base = int(round(target)) - 6
    cands = [n for n in range(base, base + 13) if n % 12 == pc and lo <= n <= hi]
    if not cands:
        cands = [n for n in range(lo, hi + 1) if n % 12 == pc]
    return min(cands, key=lambda n: (abs(n - target), n))


def voice_chord(chord: Chord, prev: tuple[int, ...] | None, lo: int, hi: int, voices: int = 4,
                center: float | None = None) -> tuple[int, ...]:
    """Pick a voicing of ``chord`` that moves as little as possible from ``prev``.

    Avoids muddy close intervals in the low register and keeps the span compact.
    """
    pcs = chord.pcs
    if len(pcs) > voices:
        # Drop the fifth first: it adds the least colour.
        fifth = (chord.root + 7) % 12
        pcs = [p for p in pcs if p != fifth] if fifth in pcs else pcs
        pcs = pcs[:voices]
    notes = [n for n in range(lo, hi + 1) if n % 12 in pcs]
    center = (lo + hi) / 2 if center is None else center
    best, best_cost = None, float("inf")
    for combo in itertools.combinations(notes, voices):
        got = {n % 12 for n in combo}
        if not set(pcs) <= got:
            continue
        if combo[-1] - combo[0] > 19:
            continue
        cost = 0.0
        for a, b in zip(combo, combo[1:]):
            gap = b - a
            if gap == 0:
                cost += 100
            if a < 55 and gap < 5:
                cost += 6  # thirds and seconds get muddy down low
            if gap == 1:
                cost += 5  # minor seconds rub
            if gap == 2 and b < 66:
                cost += 1.5
        if combo.count(combo[0]) > 1:
            cost += 100
        if prev is not None:
            p = sorted(prev)
            if len(p) == voices:
                cost += sum(abs(x - y) for x, y in zip(combo, p))
            else:
                cost += sum(min(abs(x - y) for y in p) for x in combo)
            cost += 0.15 * abs(sum(combo) / voices - center)
        else:
            cost += abs(sum(combo) / voices - center)
        if cost < best_cost:
            best, best_cost = combo, cost
    if best is None:
        best = tuple(sorted(nearest_pitch(pc, center, lo, hi) for pc in pcs[:voices]))
    return best


def scale_step(pitch: int, steps: int, key: Key) -> int:
    """Move ``pitch`` by ``steps`` diatonic steps within ``key`` (snapping first)."""
    scale = key.scale
    p = pitch
    while p % 12 not in scale:
        p -= 1
    direction = 1 if steps > 0 else -1
    for _ in range(abs(steps)):
        p += direction
        while p % 12 not in scale:
            p += direction
    return p
