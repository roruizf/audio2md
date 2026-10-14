# Capabilities and boundaries

## Supported now

- Local audio/video accepted by FFmpeg through PyAV.
- Offline CPU transcription with `faster-whisper`.
- Fixed or automatically detected language.
- Domain vocabulary as prompt and persistent hotwords.
- Auditable Markdown, JSON segments, provenance and idempotent completed-run skipping.
- Interrupted-run recovery from deterministic, signed chunk checkpoints.
- Python 3.12 CLI and a small public Python API.

## Not yet supported

- Recursive batch orchestration and per-item failure reports.
- Platform downloaders, authentication, DRM bypass or protected-content acquisition.
- LLM summaries, manuals or checklists.

Acquisition must only process sources the operator is authorized to download or transform.
LLM synthesis should remain downstream so the transcript continues to be primary evidence.
