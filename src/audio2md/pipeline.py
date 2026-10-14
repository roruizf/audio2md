from __future__ import annotations

import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable

from .checkpoint import CHECKPOINT_SCHEMA_VERSION
from .manifest import SCHEMA_VERSION, read_json, sha256_file, write_json_atomic, write_text_atomic
from .markdown import render_transcript_markdown
from .models import Transcript, TranscriptionConfig
from .transcribe import transcribe_media
from .validation import validate_transcript


Transcriber = Callable[[Path, TranscriptionConfig], Transcript]
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_slug(value: str, label: str) -> str:
    if not SLUG_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase kebab-case slug: {value!r}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_transcription(
    *,
    source: Path,
    output_root: Path,
    collection_id: str | None,
    item_id: str,
    title: str,
    config: TranscriptionConfig,
    source_offset_seconds: float = 0.0,
    force: bool = False,
    chunk_seconds: float = 600.0,
    chunk_overlap_seconds: float = 30.0,
    transcriber: Transcriber = transcribe_media,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")
    if collection_id is not None:
        validate_slug(collection_id, "collection-id")
    validate_slug(item_id, "item-id")
    if source_offset_seconds < 0:
        raise ValueError("source-offset-seconds cannot be negative")
    if chunk_seconds <= 0:
        raise ValueError("chunk-seconds must be positive")
    if chunk_overlap_seconds < 0 or chunk_overlap_seconds >= chunk_seconds:
        raise ValueError("chunk-overlap-seconds must be non-negative and less than chunk-seconds")

    output_dir = output_root.expanduser().resolve()
    if collection_id is not None:
        output_dir /= collection_id
    output_dir /= item_id
    manifest_path = output_dir / "manifest.json"
    source_hash = sha256_file(source)
    transcription_signature = {
        "source_sha256": source_hash,
        "config": config.as_dict(),
        "chunking": {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": chunk_overlap_seconds,
            "sample_rate": 16000,
        },
    }
    signature = {
        "transcription": transcription_signature,
        "render": {
            "collection_id": collection_id,
            "item_id": item_id,
            "title": title,
            "source_offset_seconds": source_offset_seconds,
        },
    }
    if not force and manifest_path.exists():
        previous = read_json(manifest_path)
        outputs_exist = all(
            (output_dir / name).is_file() for name in ("segments.json", "transcript.md")
        )
        if previous.get("signature") == signature and outputs_exist:
            quality_warnings = previous.get("result", {}).get("quality_warnings", [])
            return {
                "status": "skipped",
                "reason": "matching source and configuration already completed",
                "output_dir": str(output_dir),
                "quality_warnings": quality_warnings,
            }

    started_at = utc_now()
    started = time.perf_counter()
    if transcriber is transcribe_media:
        transcript = transcribe_media(
            source,
            config,
            checkpoint_dir=output_dir / ".checkpoint",
            checkpoint_signature=transcription_signature,
            chunk_seconds=chunk_seconds,
            overlap_seconds=chunk_overlap_seconds,
        )
    else:
        transcript = transcriber(source, config)
    elapsed_seconds = time.perf_counter() - started
    quality_warnings = [warning.as_dict() for warning in validate_transcript(transcript)]

    segments_payload = [
        {"id": f"S{index:06d}", **segment.as_dict()}
        for index, segment in enumerate(transcript.segments, start=1)
    ]
    markdown = render_transcript_markdown(
        collection_id=collection_id,
        item_id=item_id,
        title=title,
        source_name=source.name,
        source_sha256=source_hash,
        source_offset_seconds=source_offset_seconds,
        transcript=transcript,
    )
    try:
        package_version = version("audio2md")
    except Exception:
        package_version = "0.1.0"
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "source": {
            "path": str(source),
            "name": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": source_hash,
        },
        "identity": {"collection_id": collection_id, "item_id": item_id, "title": title},
        "signature": signature,
        "result": {
            "language": transcript.language,
            "language_probability": transcript.language_probability,
            "duration_seconds": transcript.duration_seconds,
            "segment_count": len(transcript.segments),
            "quality_warnings": quality_warnings,
        },
        "runtime": {
            "python": platform.python_version(),
            "audio2md": package_version,
        },
    }
    write_json_atomic(output_dir / "segments.json", segments_payload)
    write_text_atomic(output_dir / "transcript.md", markdown)
    write_json_atomic(manifest_path, manifest_payload)
    return {
        "status": "completed",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "segment_count": len(transcript.segments),
        "output_dir": str(output_dir),
        "quality_warnings": quality_warnings,
    }
