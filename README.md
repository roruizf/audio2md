# audio2md

Local-first CLI for turning authorized audio or video knowledge sources into auditable,
timestamped Markdown. The first version intentionally covers one vertical slice:

`local media → faster-whisper → segments.json + manifest.json + transcript.md`

It does not download protected course material, bypass DRM, or generate study
manuals with an LLM. Acquisition connectors and synthesis belong to later phases.

## Requirements

- Python 3.12 (selected explicitly; do not use the machine's Anaconda Python 3.8)
- `uv`
- About 1.1 GB RAM for the default `small.en` CPU/int8 configuration
- Network access on the first run to download the selected Whisper model

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev
```

Models are not stored in this repository. To choose a dedicated cache:

```bash
export HF_HOME="$HOME/.cache/audio2md/huggingface"
```

## First transcription

Create a UTF-8 vocabulary file with domain terms, names and acronyms. The pipeline
uses it both as the initial prompt and as persistent hotwords for long sources:

```text
ASHRAE 90.1, Appendix G, LEED v4, LEED v4.1, GBCI, IECC,
Gail Hansmeyer, Jamie Wilson, Pathfinder Engineers and Architects.
```

For a single source, the file stem provides the default item ID and title:

```bash
uv run audio2md "Appendix G introduction.mp3" -o output \
  --language en \
  --vocabulary-file vocabulary.txt
```

This writes to `output/appendix-g-introduction/`. Use explicit identity when the
source belongs to a collection or its filename is not descriptive:

```bash
uv run audio2md lesson.mp3 -o output \
  --collection-id ashrae-901-appendix-g \
  --item-id introduction \
  --title "Appendix G introduction"
```

The command creates:

```text
output/[<collection-id>/]<item-id>/
├── manifest.json
├── segments.json
└── transcript.md
```

Running the same command again skips transcription when the source SHA-256 and
configuration match. Use `--force` to regenerate it.

Long sources are checkpointed in deterministic 10-minute blocks with 30 seconds of context overlap. Use
`--chunk-seconds` and `--chunk-overlap-seconds` to tune this. After interruption, rerun the identical command;
completed blocks with a matching source and configuration signature are reused.

The CLI returns `0` for a clean bundle, `1` for a failed conversion and `2` when it
created a bundle with non-fatal quality warnings. See [the output contract](docs/output-schema.md).

## Design rules

- The source transcript is primary evidence; summaries are downstream artifacts.
- Every run records source hash, parameters, versions, duration and elapsed time.
- Timestamps are relative to the input file. `--source-offset-seconds` maps a clip
  back to its original video timeline.
- Source/domain-specific vocabulary is explicit input, never a hidden global prompt.
- Generated media, model weights, caches and transcripts stay out of Git.

## Current limitations

- LLM-based study artifacts and source-platform connectors are not implemented yet.

## Development

```bash
uv run pytest
uv run audio2md --help
```

Design references: [architecture](docs/architecture.md),
[output bundle](docs/output-schema.md), and [capabilities](docs/capabilities.md).

Prepare provider-neutral evidence packets for grounded synthesis:

```bash
uv run audio2md-study prepare output/my-item
uv run audio2md-study validate output/my-item output/my-item/study/manual.md
```

This step is local and does not call an LLM or send transcript content over the network. See the
[grounded synthesis contract](docs/synthesis-contract.md).

Hosted model adapters are optional and isolated from the transcription path. See [providers](docs/providers.md).
OpenCode Go can be used through an existing subscription without exposing or reusing project API keys.
