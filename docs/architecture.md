# Architecture

`audio2md` follows the small, testable pipeline used successfully by `doc2md`:

```text
local media -> ASR adapter -> canonical Transcript -> validator -> renderers -> atomic bundle
```

- The ASR adapter is currently `faster-whisper`.
- `Transcript` and `Segment` are the canonical, engine-independent model.
- Structural validation runs before any final artifact is published.
- Renderers produce machine-readable segments and human-readable Markdown.
- `manifest.json` is written last, so its presence marks a completed bundle.

The CLI and Python API both call `run_transcription`; neither has a separate conversion path.
Acquisition and LLM synthesis are deliberately outside this core.

Long inputs are decoded once to mono 16 kHz PCM and split into deterministic core windows with context overlap.
A segment belongs to the core containing its midpoint. Each completed chunk and its signature are written
atomically below `.checkpoint/`; a retry reuses only chunks whose source, ASR and chunking signature matches.
