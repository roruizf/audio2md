# Grounded synthesis contract

Synthesis is downstream from transcription and never replaces the source bundle. The proposed flow is:

```text
segments.json -> evidence packets -> claim ledger -> manual/summary/checklist -> grounding audit
```

## Invariants

1. Every factual claim cites one or more segment IDs such as `[S000042]`.
2. A claim may combine segments, but must not introduce facts absent from those segments.
3. Inferences are explicitly labeled and kept separate from source claims.
4. Promotion, housekeeping and speaker logistics are excluded from study content but may be logged as exclusions.
5. Unclear terminology and contradictions remain visible; the model must not silently repair them.
6. Outputs are regenerable. `transcript.md` and `segments.json` remain primary evidence.

## Proposed derived bundle

```text
study/
├── claims.json
├── grounding-report.json
├── manual.md
├── summary.md
└── checklist.md
```

`claims.json` is the canonical intermediate artifact. Each entry contains a stable claim ID, claim text, evidence
segment IDs, classification (`source`, `inference`, `uncertain`) and optional terminology notes. Human-facing files
are rendered only from that ledger. The grounding report records unsupported, partially supported or contradictory
statements and blocks publication when unsupported claims remain.

## Execution boundary

Packet creation can be deterministic and local. Model execution must use an explicit adapter. Local models keep the
entire flow offline; sending transcripts to a hosted API requires separate approval for the provider and material.
The first implementation should accept an already-created transcript bundle, not acquire media itself.

## Local preparation

```bash
audio2md-study prepare path/to/bundle --max-words 1200 --overlap-segments 2
audio2md-study validate path/to/bundle path/to/bundle/study/manual.md
```

Preparation is idempotent and signs the source `segments.json` plus packet settings. `max-words` limits new core
evidence; explicitly labeled overlap context may increase a packet's total. Every packet after the first contributes
at least one new segment. Validation checks that content lines carry citations and that every cited segment exists.
Semantic entailment remains the later verifier model's job.

Hosted extraction is guarded by the explicit `--execute-hosted` flag and an idempotency signature covering the
evidence packets, prompt version, adapter and model. An identical completed request is skipped unless `--force` is
provided. Model output is validated against bundle-local segment IDs before any Markdown artifact is written.

The provider-neutral protocol and the first hosted implementation are documented in [providers](providers.md).
