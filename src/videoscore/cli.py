"""Command line interface: ``videoscore render | plan | styles | example``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .spec import SpecError, parse_spec
from .styles import STYLES

EXAMPLE = {
    "duration": 48.0,
    "style": "corporate",
    "seed": 7,
    "scenes": [
        {"start": 0.0, "label": "logo", "energy": "low"},
        {"start": 4.5, "label": "problem", "energy": 0.4},
        {"start": 15.2, "label": "demo", "energy": "high"},
        {"start": 33.8, "label": "results", "energy": 0.7},
        {"start": 42.0, "label": "call to action", "energy": 0.45},
    ],
    "hits": [24.0],
    "ending": "resolve",
    "loudness": -18,
}


def _load(args) -> dict:
    data: dict = {}
    if args.cue:
        path = Path(args.cue)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise SpecError(f"Cue sheet not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path} is not valid JSON: {exc}") from None
        vo = data.get("voiceover") if isinstance(data, dict) else None
        if isinstance(vo, str) and vo.strip() and not Path(vo).is_absolute():
            data["voiceover"] = str((path.parent / data["voiceover"]).resolve())
    overrides = {
        "duration": args.duration, "style": args.style, "seed": args.seed, "loudness": args.loudness,
        "ending": args.ending, "key": args.key, "tempo": args.tempo, "voiceover": args.voiceover,
    }
    # command-line times are always seconds; a cue sheet in frames gets them converted
    per_second = 1.0
    if data.get("unit") == "frames" and isinstance(data.get("fps"), (int, float)) and data["fps"] > 0:
        per_second = float(data["fps"])
    if args.duration is not None:
        overrides["duration"] = args.duration * per_second
    for k, v in overrides.items():
        if v is not None:
            data[k] = v
    if args.transitions is not None:
        scenes = data.pop("scenes", None)
        if "duration" not in data and isinstance(scenes, list) and scenes and all(
                isinstance(s, dict) and isinstance(s.get("duration"), (int, float)) for s in scenes):
            data["duration"] = sum(s["duration"] for s in scenes)  # keep the length the scenes implied
        try:
            data["transitions"] = [float(t) * per_second for t in args.transitions.split(",") if t.strip()]
        except ValueError:
            raise SpecError("--transitions must be comma-separated numbers of seconds, e.g. 4.5,15.2") from None
    if not data:
        raise SpecError("Give a cue sheet JSON file, or at least --duration.")
    return data


def _summary(rep: dict) -> str:
    lines = [f"{rep['style']} in {rep['key']}, base {rep['base_bpm']} BPM, {rep['duration']}s, ending: {rep['ending']}"]
    for c in rep["cues"]:
        if c["aligned_time"] is None:
            lines.append(f"  cue {c['time']:>8.3f}s {c['kind']:<7} NOT SYNCED: under a bar from another cue "
                         f"(nearest downbeat {c['nearest_downbeat']:.3f}s)")
        else:
            lines.append(f"  cue {c['time']:>8.3f}s {c['kind']:<7} -> downbeat of bar {c['bar']} "
                         f"(error {c['error_ms']:.3f} ms) {c['label']}")
    for s in rep["sections"]:
        lines.append(f"  section {s['start']:>7.2f}-{s['end']:<7.2f} energy {s['energy']:.2f} part {s['part']} "
                     f"{s['bpm']:.1f} BPM: {', '.join(s['layers'])}")
    for w in rep["warnings"]:
        lines.append(f"  warning: {w}")
    if "audio" in rep:
        a = rep["audio"]
        lines.append(f"wrote {a['path']} ({a['loudness_lufs']} LUFS, true peak {a['true_peak_dbtp']} dBTP, "
                     f"{a['render_seconds']}s) and {rep['report_path']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:  # output piped into `head` etc.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(
        prog="videoscore",
        description="Generate background music for a video, with every scene transition on a downbeat.",
        epilog="Example: videoscore render cue.json -o music.wav    |    videoscore example > cue.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("cue", nargs="?", help="cue sheet JSON (see `videoscore example`)")
        p.add_argument("--duration", type=float, help="video length in seconds (overrides the cue sheet)")
        p.add_argument("--transitions", help="comma-separated scene-change times in seconds, e.g. 4.5,15.2,33.8")
        p.add_argument("--style", choices=sorted(STYLES), help="musical style")
        p.add_argument("--seed", type=int, help="variation seed; same seed + cue sheet = identical music")
        p.add_argument("--key", help="force a key, e.g. 'D major' or 'A minor'")
        p.add_argument("--tempo", type=float, help="preferred BPM (otherwise the style's range is used)")
        p.add_argument("--ending", choices=["resolve", "fade"], help="ring out on a final chord, or fade out")
        p.add_argument("--loudness", type=float, help="integrated loudness target in LUFS (default -18)")
        p.add_argument("--voiceover", help="narration audio; the music ducks automatically while it speaks")

    r = sub.add_parser("render", help="compose and render audio + a JSON timing report")
    common(r)
    r.add_argument("-o", "--out", required=True, help="output audio: .wav, or .mp3/.m4a/.ogg/.flac/.opus (ffmpeg)")
    r.add_argument("--report", help="where to write the JSON report (default: <out>.json)")
    r.add_argument("--json", action="store_true", help="print the full report as JSON instead of a summary")

    p = sub.add_parser("plan", help="print the tempo map, sections and beat grid without rendering audio")
    common(p)
    p.add_argument("--out", help="write the plan JSON here instead of stdout")

    sub.add_parser("styles", help="list the available styles")
    sub.add_parser("example", help="print an example cue sheet")

    args = parser.parse_args(argv)
    if args.command == "styles":
        for st in STYLES.values():
            print(f"{st.name:<11} {st.tempo[0]:.0f}-{st.tempo[1]:.0f} BPM  {st.description}")
        return 0
    if args.command == "example":
        print(json.dumps(EXAMPLE, indent=2))
        return 0

    from .engine import plan, render

    try:
        spec = parse_spec(_load(args))
        if args.command == "plan":
            rep = plan(spec)
            text = json.dumps(rep, indent=2)
            if args.out:
                Path(args.out).write_text(text)
                print(_summary(rep))
            else:
                print(text)
            return 0
        rep = render(spec, args.out, args.report)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
        print(f"error: ffmpeg failed: {detail[-1] if detail else exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(rep, indent=2) if args.json else _summary(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
