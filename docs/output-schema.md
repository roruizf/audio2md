# Output bundle contract (schema version 1)

Each successful item directory contains exactly three public artifacts:

- `manifest.json`: provenance, identity, run signature, runtime and quality signals.
- `segments.json`: ordered ASR segments with stable bundle-local `id`, `start`, `end` and `text`.
- `transcript.md`: YAML metadata plus timestamped transcript for humans and LLM ingestion.

`manifest.json` is committed atomically after the other two files. A matching manifest and
existing outputs allow an idempotent rerun to skip ASR. The signature has two layers: transcription
(source SHA-256, ASR configuration and chunking) and render (identity, title and source timeline offset).
Changing presentation metadata can therefore reuse valid ASR checkpoints while rebuilding the public bundle.
Segment IDs use `S000001`, `S000002`, ... and appear in both JSON and Markdown so derived claims can cite evidence.

## Quality behavior

Structurally invalid timestamps abort the run with exit code `1` and no completed manifest.
Non-fatal degradation is recorded under `result.quality_warnings`; the CLI still writes the
bundle but exits with code `2`. Clean completion and a clean idempotent skip return `0`.

Schema changes that remove or reinterpret fields require a new `schema_version`. Adding an
optional field is backward compatible within version 1.

`.checkpoint/` is private working state rather than part of the public bundle. It may be removed after a clean
run; keeping it permits recovery or regeneration. A checkpoint is reused only when its canonical signature matches.
