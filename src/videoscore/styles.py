"""Style presets.

A style is pure data: tempo range, harmony, which instrument layers come in at
which energy, drum patterns per energy tier, and how transitions are marked.

Pattern notation (one character per 16th note, 16 per bar):

* drums: ``X`` accent, ``x`` normal, ``o`` soft, ``g`` ghost, ``.`` rest
* comping/bass: ``x``/``R`` root (or chord), ``3``/``5``/``8`` chord third, fifth,
  octave, ``A`` approach note toward the next chord, ``-`` hold, ``.`` rest
* melody rhythms are two bars (32 steps) of ``x`` onset, ``-`` hold, ``.`` rest
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Layer:
    name: str
    role: str  # pad | keys | arp | bass | melody
    instrument: str
    min_energy: float = 0.0
    max_energy: float = 1.01
    gain_db: float = 0.0
    pan: float = 0.0
    highpass: float = 0.0
    lowpass: float = 0.0
    reverb: float = 0.2
    delay: float = 0.0
    sidechain: float = 0.0
    range: tuple[int, int] = (52, 76)
    voices: int = 4
    # keys/bass: rhythm patterns per tier ("low", "mid", "high"); arp: index patterns
    patterns: dict = field(default_factory=dict)
    arp_rate: dict = field(default_factory=dict)  # tier -> steps per note (1 = 16th)
    params: dict = field(default_factory=dict)


@dataclass
class DrumPart:
    instrument: str
    patterns: dict  # tier -> list of 16-step patterns (variations)
    gain_db: float = 0.0
    pan: float = 0.0
    reverb: float = 0.08


@dataclass
class Style:
    name: str
    description: str
    tempo: tuple[float, float]
    preferred_tempo: float
    modes: list[str]
    progressions: dict  # mode -> list of (A, B) roman-numeral progressions
    cadence: dict  # mode family -> chord before the final tonic
    layers: list[Layer]
    drums: list[DrumPart]
    drum_min_energy: float = 0.35
    bars_per_chord: tuple[int, int] = (1, 1)  # (low energy, high energy)
    swing: float = 0.5
    humanize_ms: float = 6.0
    tiers: tuple[float, float] = (0.5, 0.72)  # energy where drums go low->mid->high
    fx: dict = field(default_factory=dict)
    reverb: dict = field(default_factory=dict)
    delay_beats: float = 0.75
    delay_feedback: float = 0.35
    melody_rhythms: dict = field(default_factory=dict)  # "sparse"/"dense" -> list of 32-step strings
    master_lowpass: float = 0.0
    texture: str | None = None
    keys: list[str] = field(default_factory=lambda: ["C", "D", "Eb", "F", "G", "A", "Bb"])


MINOR_BANK = [
    (["i", "VI", "III", "VII"], ["VI", "VII", "i", "i"]),
    (["i", "iv", "VI", "V"], ["VI", "III", "VII", "i"]),
    (["i", "VII", "VI", "VII"], ["iv", "VI", "III", "VII"]),
]

SPARSE_MELODY = [
    "x-----..x---x---x-------........",
    "x---x---x-------....x-x-x-------",
    "....x---x---x---x-------........",
    "x-------x---..x-x---------......",
]
DENSE_MELODY = [
    "x-x-x---x-x-x---x---x---x-------",
    "x---x-x-x---x---x-x-x---x---....",
    "..x-x-x-x---x---..x-x-x-x-------",
    "x-x---x-x---x-x-x-------x-x-x---",
]

CORPORATE = Style(
    name="corporate",
    description="Bright, optimistic product/tech music: plucked arpeggios, electric piano, "
                "soft four-on-the-floor. Good default for demos, explainers and launches.",
    tempo=(104, 122), preferred_tempo=112,
    modes=["major"],
    progressions={
        "major": [
            (["Iadd9", "vi", "IVadd9", "V"], ["IV", "I", "V", "vi"]),
            (["vi", "IV", "I", "V"], ["I", "V", "vi", "IV"]),
            (["I", "iii", "vi", "IV"], ["IV", "V", "I", "vi"]),
            (["I", "V", "vi", "IV"], ["IVadd9", "V", "vi", "Iadd9"]),
        ],
        "minor": MINOR_BANK,
    },
    cadence={"major": "V", "minor": "VII"},
    layers=[
        Layer("pad", "pad", "warm_pad", gain_db=-7, highpass=180, reverb=0.35, sidechain=0.35,
              range=(52, 76), params={"attack": 0.5, "release": 1.2, "cutoff": 2200}),
        Layer("keys", "keys", "ep", min_energy=0.3, gain_db=-10, highpass=200, reverb=0.25,
              range=(55, 76), voices=4, pan=-0.15,
              patterns={"low": ["x---------------"], "mid": ["x-----x---x-----", "x-----x-----x---"],
                        "high": ["x-----x---x---x-", "x---x-----x-x---"]}),
        Layer("arp", "arp", "pluck", min_energy=0.42, gain_db=-11, highpass=300, reverb=0.2,
              delay=0.22, pan=0.2, range=(64, 88),
              patterns={"low": [[0, 1, 2, 3, 1, 2, 3, 4]], "mid": [[0, 1, 2, 3, 1, 2, 3, 4], [0, 2, 1, 3, 2, 4, 3, 1]],
                        "high": [[0, 1, 2, 3, 4, 3, 2, 1], [0, 2, 1, 3, 2, 4, 3, 1]]},
              arp_rate={"low": 2, "mid": 2, "high": 1}),
        Layer("bass", "bass", "sub_bass", min_energy=0.3, gain_db=-6, sidechain=0.4, range=(33, 47),
              patterns={"low": ["R---------------"], "mid": ["R-------R---R---", "R-----R-----R---"],
                        "high": ["R-.-R-.-R-.-R-.-", "R-.-R-.-R-.-8-.-"]}),
        Layer("melody", "melody", "bell", min_energy=0.58, gain_db=-12, highpass=400, reverb=0.35,
              delay=0.3, pan=0.1, range=(72, 88)),
    ],
    drums=[
        DrumPart("kick", {"low": ["x.......x......."], "mid": ["x...x...x...x..."],
                          "high": ["x...x...x...x...", "x...x...x...x..x"]}, gain_db=-3),
        DrumPart("clap", {"low": [""], "mid": ["....x.......x..."], "high": ["....x.......x..."]},
                 gain_db=-9, reverb=0.2),
        DrumPart("hat", {"low": [""], "mid": ["..x...x...x...x."], "high": ["gox.gox.gox.gox.", "g.x.g.x.g.x.goxo"]},
                 gain_db=-15, pan=0.25),
        DrumPart("shaker", {"low": ["o.go..go..go..go"], "mid": ["gogogogogogogogo"], "high": ["gxgogxgogxgogxgo"]},
                 gain_db=-19, pan=-0.3),
        DrumPart("ohat", {"low": [""], "mid": [""], "high": ["..o...o...o...o."]}, gain_db=-20, pan=0.2),
    ],
    drum_min_energy=0.36,
    fx={"crash": 0.5, "riser": True, "swell": True, "impact": None, "fill": "clap", "break": True,
        "hit": "crash"},
    reverb={"rt60": 2.2, "predelay": 0.02, "damping": 0.5},
    melody_rhythms={"sparse": SPARSE_MELODY, "dense": DENSE_MELODY},
)

AMBIENT = Style(
    name="ambient",
    description="Calm and spacious: slow evolving pads, soft piano and bell motifs, no drums until "
                "the very top of the energy range. For explainers, nature, product beauty shots.",
    tempo=(64, 84), preferred_tempo=72,
    modes=["major", "lydian"],
    progressions={
        "major": [
            (["Imaj7", "IVmaj7"], ["vi7", "IVmaj7", "Iadd9", "Vsus4"]),
            (["Iadd9", "iii7", "IVmaj7", "IVmaj7"], ["vi7", "V", "IVmaj7", "Iadd9"]),
            (["IVmaj7", "Iadd9"], ["vi7", "iii7", "IVmaj7", "Vsus4"]),
        ],
        "lydian": [
            (["Iadd9", "II"], ["vi7", "II", "Imaj7", "V"]),
            (["Imaj7", "II", "iii7", "II"], ["vi7", "V", "II", "Iadd9"]),
        ],
        "minor": [
            (["i7", "VImaj7"], ["III", "VII", "VImaj7", "i7"]),
            (["i", "VI", "III", "VII"], ["VI", "VII", "i", "i"]),
        ],
    },
    cadence={"major": "IVmaj7", "lydian": "II", "minor": "VII"},
    layers=[
        Layer("pad", "pad", "warm_pad", gain_db=-6, highpass=150, reverb=0.5, range=(50, 76),
              params={"attack": 1.4, "release": 2.5, "cutoff": 2200}),
        Layer("air", "pad", "air_pad", min_energy=0.35, gain_db=-17, highpass=500, reverb=0.65,
              range=(67, 86), voices=3, params={"attack": 2.0, "release": 3.0}),
        Layer("piano", "arp", "soft_piano", min_energy=0.25, gain_db=-11, highpass=150, reverb=0.45,
              delay=0.15, pan=-0.1, range=(57, 81),
              patterns={"low": [[0, 2, 1, 3]], "mid": [[0, 2, 1, 3, 2, 4, 3, 1]], "high": [[0, 2, 4, 3, 1, 3, 2, 4]]},
              arp_rate={"low": 4, "mid": 4, "high": 4}),
        Layer("sparkle", "arp", "bell", min_energy=0.28, gain_db=-19, highpass=900, reverb=0.6, delay=0.4,
              pan=0.25, range=(79, 96),
              patterns={"low": [[0, 2, 1, 3, 2, 4]], "mid": [[0, 2, 1, 3, 2, 4]], "high": [[0, 2, 4, 3, 1, 3]]},
              arp_rate={"low": 6, "mid": 6, "high": 3}),
        Layer("bass", "bass", "sub_bass", min_energy=0.45, gain_db=-11, range=(33, 47),
              patterns={"low": ["R---------------"], "mid": ["R---------------"], "high": ["R-------R-------"]}),
        Layer("melody", "melody", "bell", min_energy=0.45, gain_db=-14, highpass=500, reverb=0.6,
              delay=0.35, pan=0.15, range=(72, 88)),
    ],
    drums=[
        DrumPart("kick_soft", {"low": [""], "mid": [""], "high": ["x.......x......."]}, gain_db=-8),
        DrumPart("shaker", {"low": [""], "mid": [""], "high": ["..o...o...o...o."]}, gain_db=-22, pan=-0.3),
    ],
    drum_min_energy=0.78,
    bars_per_chord=(2, 2),
    humanize_ms=10,
    tiers=(0.5, 0.78),
    fx={"crash": None, "riser": False, "swell": True, "impact": None, "fill": None, "break": False,
        "hit": "swell"},
    reverb={"rt60": 4.5, "predelay": 0.04, "damping": 0.6},
    delay_beats=1.5, delay_feedback=0.4,
    melody_rhythms={"sparse": SPARSE_MELODY, "dense": SPARSE_MELODY},
)

LOFI = Style(
    name="lofi",
    description="Laid-back lo-fi hip-hop: jazzy seventh chords on electric piano, swung dusty "
                "drums, warm bass and vinyl texture. For dev logs, tutorials, coding timelapses.",
    tempo=(72, 90), preferred_tempo=82,
    modes=["dorian", "minor", "major"],
    progressions={
        "major": [
            (["IVmaj7", "iii7", "ii7", "Imaj7"], ["IVmaj7", "V7", "iii7", "vi7"]),
            (["ii7", "V7", "Imaj7", "vi7"], ["IVmaj7", "iii7", "vi7", "V9"]),
        ],
        "dorian": [
            (["i7", "IV9"], ["IIImaj7", "IV7", "v7", "i7"]),
            (["i7", "IV7", "i7", "VIImaj7"], ["IIImaj7", "IV7", "VIImaj7", "i7"]),
        ],
        "minor": [
            (["i7", "iv7", "VII7", "IIImaj7"], ["VImaj7", "iv7", "v7", "i7"]),
            (["i7", "VImaj7", "iv7", "v7"], ["IIImaj7", "VII7", "VImaj7", "v7"]),
        ],
    },
    cadence={"major": "V7", "dorian": "IV7", "minor": "v7"},
    layers=[
        Layer("keys", "keys", "ep", gain_db=-6, highpass=120, lowpass=4200, reverb=0.25, range=(53, 74),
              voices=4, pan=-0.1,
              patterns={"low": ["x---------------", "x-------------x-"], "mid": ["x---------x-----", "x-----x-------x-"],
                        "high": ["x-----x---x-----", "x---------x---x-"]}),
        Layer("pad", "pad", "warm_pad", min_energy=0.3, gain_db=-17, highpass=200, lowpass=1800, reverb=0.4,
              range=(55, 76), params={"attack": 0.8, "release": 1.5, "cutoff": 1200}),
        Layer("bass", "bass", "round_bass", min_energy=0.25, gain_db=-5, range=(33, 47),
              patterns={"low": ["R-------........"], "mid": ["R-------5-----A-", "R-----R---5-----"],
                        "high": ["R-----R---5---A-", "R---.-R-.-5-.-A-"]}),
        Layer("melody", "melody", "bell", min_energy=0.5, gain_db=-14, highpass=400, lowpass=5000,
              reverb=0.35, delay=0.25, pan=0.15, range=(70, 86)),
    ],
    drums=[
        DrumPart("kick_soft", {"low": ["x.........x....."], "mid": ["x.........x..x.."],
                               "high": ["x......x..x.....", "x.........x..x.."]}, gain_db=-3),
        DrumPart("rim", {"low": ["....o.......o..."], "mid": [""], "high": [""]}, gain_db=-12),
        DrumPart("snare", {"low": [""], "mid": ["....x.......x..."], "high": ["....x.......x..g"]}, gain_db=-7,
                 reverb=0.12),
        DrumPart("hat", {"low": ["o.g.o.g.o.g.o.g."], "mid": ["x.o.x.o.x.o.x.o."], "high": ["xgogxgogxgogxgog"]},
                 gain_db=-15, pan=0.2),
    ],
    drum_min_energy=0.2,
    swing=0.58,
    humanize_ms=9,
    fx={"crash": None, "riser": False, "swell": True, "impact": None, "fill": None, "break": True,
        "hit": "swell"},
    reverb={"rt60": 1.6, "predelay": 0.015, "damping": 0.7},
    delay_beats=0.75, delay_feedback=0.3,
    melody_rhythms={"sparse": SPARSE_MELODY, "dense": SPARSE_MELODY},
    master_lowpass=11000,
    texture="vinyl",
)

CINEMATIC = Style(
    name="cinematic",
    description="Dramatic and building: string ensemble, ostinato, felt piano, deep impacts and "
                "risers on scene changes. For launches, trailers, reveals and big moments.",
    tempo=(80, 100), preferred_tempo=90,
    modes=["minor", "major"],
    progressions={
        "minor": [
            (["i", "VI", "III", "VII"], ["i", "iv", "VI", "V"]),
            (["i", "VI", "iv", "V"], ["VI", "III", "VII", "i"]),
            (["i", "III", "VII", "VI"], ["iv", "VI", "III", "V"]),
        ],
        "major": [
            (["I", "V", "vi", "IV"], ["vi", "IV", "I", "V"]),
            (["vi", "IV", "I", "V"], ["IV", "I", "V", "vi"]),
        ],
    },
    cadence={"major": "V", "minor": "V"},
    layers=[
        Layer("strings", "pad", "strings", gain_db=-6, highpass=90, reverb=0.45, range=(48, 74),
              params={"attack": 0.6, "release": 1.0}),
        Layer("piano", "melody", "soft_piano", max_energy=0.6, gain_db=-9, highpass=150, reverb=0.45,
              delay=0.12, range=(64, 84)),
        Layer("ostinato", "arp", "pluck", min_energy=0.42, gain_db=-11, highpass=180, reverb=0.3, range=(50, 69),
              patterns={"low": [[0, 1, 2, 1]], "mid": [[0, 1, 2, 1], [0, 2, 1, 2]], "high": [[0, 1, 2, 3, 2, 1, 2, 1]]},
              arp_rate={"low": 2, "mid": 2, "high": 1}, params={"decay": 0.25}),
        Layer("bass", "bass", "sub_bass", min_energy=0.3, gain_db=-6, range=(31, 45),
              patterns={"low": ["R---------------"], "mid": ["R-------R-------"],
                        "high": ["R-.-R-.-R-.-R-.-"]}),
        Layer("high_strings", "pad", "strings", min_energy=0.62, gain_db=-14, highpass=400, reverb=0.5,
              range=(69, 86), voices=3, params={"attack": 0.8, "release": 1.0}),
        Layer("glock", "melody", "bell", min_energy=0.62, gain_db=-15, highpass=600, reverb=0.45, delay=0.2,
              pan=0.2, range=(76, 91)),
    ],
    drums=[
        DrumPart("tom_low", {"low": [""], "mid": ["x.....x.x......."], "high": ["x..x..x.x..x..x.", "x.....x.x..x.xx."]},
                 gain_db=-5, reverb=0.3),
        DrumPart("kick", {"low": [""], "mid": [""], "high": ["x.......x......."]}, gain_db=-4),
        DrumPart("snare_big", {"low": [""], "mid": [""], "high": ["........x......."]}, gain_db=-7, reverb=0.4),
        DrumPart("shaker", {"low": [""], "mid": [""], "high": ["gogxgogxgogxgogx"]}, gain_db=-21, pan=0.3),
    ],
    drum_min_energy=0.5,
    bars_per_chord=(2, 1),
    tiers=(0.5, 0.74),
    humanize_ms=7,
    fx={"crash": 0.7, "riser": True, "swell": True, "impact": 0.65, "fill": "tom_low", "break": True,
        "hit": "impact"},
    reverb={"rt60": 3.2, "predelay": 0.03, "damping": 0.55},
    delay_beats=0.75, delay_feedback=0.3,
    melody_rhythms={"sparse": SPARSE_MELODY, "dense": SPARSE_MELODY},
)

ELECTRONIC = Style(
    name="electronic",
    description="Energetic modern electronic/pop: side-chained supersaw chords, 16th-note plucks, "
                "off-beat bass, four-on-the-floor. For promos, social clips and fast montages.",
    tempo=(116, 128), preferred_tempo=122,
    modes=["minor", "major"],
    progressions={
        "minor": [
            (["i", "VI", "III", "VII"], ["VI", "VII", "i", "i"]),
            (["i", "VII", "VI", "VII"], ["VI", "III", "VII", "i"]),
            (["VI", "VII", "i", "i"], ["iv", "VI", "III", "VII"]),
        ],
        "major": [
            (["I", "V", "vi", "IV"], ["vi", "IV", "I", "V"]),
            (["vi", "IV", "I", "V"], ["IV", "V", "vi", "I"]),
        ],
    },
    cadence={"major": "V", "minor": "VII"},
    layers=[
        Layer("chords", "pad", "supersaw", gain_db=-9, highpass=200, reverb=0.25, sidechain=0.6,
              range=(55, 76), params={"attack": 0.03, "release": 0.25}),
        Layer("arp", "arp", "pluck", min_energy=0.4, gain_db=-11, highpass=350, reverb=0.2, delay=0.25,
              range=(67, 88), pan=0.15,
              patterns={"low": [[0, 1, 2, 3]], "mid": [[0, 1, 2, 3, 1, 2, 3, 4]], "high": [[0, 2, 1, 3, 2, 4, 3, 1]]},
              arp_rate={"low": 2, "mid": 1, "high": 1}, params={"decay": 0.18}),
        Layer("bass", "bass", "synth_bass", min_energy=0.3, gain_db=-6, sidechain=0.3, range=(33, 47),
              patterns={"low": ["R---------------"], "mid": ["..R-..R-..R-..R-"], "high": ["..R-..R-..R-..R-", "..R-..R-..R-..8-"]}),
        Layer("melody", "melody", "bell", min_energy=0.6, gain_db=-12, highpass=400, reverb=0.3, delay=0.3,
              range=(72, 88)),
    ],
    drums=[
        DrumPart("kick", {"low": ["x.......x......."], "mid": ["x...x...x...x..."], "high": ["x...x...x...x..."]},
                 gain_db=-1),
        DrumPart("clap", {"low": [""], "mid": ["....x.......x..."], "high": ["....x.......x..."]}, gain_db=-8,
                 reverb=0.25),
        DrumPart("hat", {"low": ["..o...o...o...o."], "mid": ["gox.gox.gox.gox."], "high": ["gxoxgxoxgxoxgxox"]},
                 gain_db=-15, pan=0.2),
        DrumPart("ohat", {"low": [""], "mid": ["..o...o...o...o."], "high": ["..x...x...x...x."]}, gain_db=-17,
                 pan=-0.15),
    ],
    drum_min_energy=0.3,
    humanize_ms=0,
    fx={"crash": 0.5, "riser": True, "swell": True, "impact": 0.8, "fill": "clap", "break": True,
        "hit": "crash"},
    reverb={"rt60": 1.8, "predelay": 0.02, "damping": 0.5},
    delay_beats=0.75, delay_feedback=0.35,
    melody_rhythms={"sparse": SPARSE_MELODY, "dense": DENSE_MELODY},
)

STYLES: dict[str, Style] = {s.name: s for s in (CORPORATE, AMBIENT, LOFI, CINEMATIC, ELECTRONIC)}
