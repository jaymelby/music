<!-- Paste into .github/copilot-instructions.md (or AGENTS.md) of a repository that produces videos. -->

## Background music

After creating or changing a video, add background music with `videoscore` (see the
`video-background-music` skill in `.github/skills/`):

1. Write `music/cue.json` with the exact video `duration` and the start time of every scene
   (read them from the code, or use `"unit": "frames"` with `fps`).
2. Run `videoscore render music/cue.json -o music/bgm.wav` and check that every cue reports
   `error 0.000 ms`.
3. Mux it with ffmpeg (or `<Audio>` in Remotion). Never trim or loop the music file; it
   already matches the video length and ends on its own.
4. If there is narration, set `"voiceover"` in the cue sheet so the music ducks under speech.
