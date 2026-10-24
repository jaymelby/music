from videoscore.theory import parse_key, roman_chord, scale_step, voice_chord


def pcs(symbol, key="C major"):
    return sorted(roman_chord(symbol, parse_key(key)).pcs)


def test_roman_numerals_in_major_and_minor():
    assert pcs("I") == [0, 4, 7]
    assert pcs("vi") == [0, 4, 9]
    assert pcs("V7") == [2, 5, 7, 11]
    assert pcs("Imaj7") == [0, 4, 7, 11]
    assert pcs("ii7") == [0, 2, 5, 9]
    assert pcs("Iadd9") == [0, 2, 4, 7]
    assert pcs("Vsus4") == [0, 2, 7]
    assert pcs("bVII") == [2, 5, 10]
    assert pcs("i", "A minor") == [0, 4, 9]
    assert pcs("VI", "A minor") == [0, 5, 9]
    assert pcs("V", "A minor") == [4, 8, 11]  # harmonic-minor dominant
    assert pcs("IV7", "D dorian") == [2, 5, 7, 11]  # G7: dorian's major IV with its flat seventh


def test_voice_leading_moves_little():
    key = parse_key("C major")
    prev = None
    total = 0
    for sym in ["I", "V", "vi", "IV"] * 3:
        v = voice_chord(roman_chord(sym, key), prev, 52, 76)
        assert len(v) == 4 and all(52 <= p <= 76 for p in v)
        assert {p % 12 for p in v} >= set(roman_chord(sym, key).pcs)
        if prev:
            total += sum(abs(a - b) for a, b in zip(v, prev))
        prev = v
    assert total / 11 <= 6  # under ~1.5 semitones per voice per change


def test_scale_step_stays_in_key():
    key = parse_key("D major")
    p = 62
    for step in [1, 1, 2, -3, 4, -1]:
        p = scale_step(p, step, key)
        assert p % 12 in key.scale
