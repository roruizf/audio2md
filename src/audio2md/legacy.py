from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from .exceptions import InvalidBundleError
from .manifest import write_json_atomic, write_text_atomic
from .markdown import render_transcript_markdown
from .models import Segment, Transcript


def migrate_legacy_bundle(legacy_dir: Path, output_dir: Path) -> dict[str, object]:
    legacy_dir = legacy_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    try:
        legacy_manifest = json.loads((legacy_dir / "run.json").read_text(encoding="utf-8"))
        raw_segments = json.loads((legacy_dir / "segments.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidBundleError(f"Cannot read legacy bundle: {error}") from error
    if not isinstance(legacy_manifest, dict) or not isinstance(raw_segments, list):
        raise InvalidBundleError("Legacy run.json or segments.json has an invalid shape")
    segments = tuple(
        Segment(float(raw["start"]), float(raw["end"]), str(raw["text"]))
        for raw in raw_segments
    )
    result = legacy_manifest.get("result") or {}
    identity = legacy_manifest.get("identity") or {}
    source = legacy_manifest.get("source") or {}
    transcript = Transcript(
        language=str(result.get("language") or "unknown"),
        language_probability=float(result.get("language_probability") or 0.0),
        duration_seconds=float(result.get("duration_seconds") or 0.0),
        segments=segments,
    )
    segment_payload = [
        {"id": f"S{index:06d}", **segment.as_dict()}
        for index, segment in enumerate(segments, start=1)
    ]
    markdown = render_transcript_markdown(
        collection_id=identity.get("collection_id"),
        item_id=str(identity.get("item_id") or output_dir.name),
        title=str(identity.get("title") or output_dir.name),
        source_name=str(source.get("name") or "legacy-media"),
        source_sha256=str(source.get("sha256") or "unknown"),
        source_offset_seconds=float(
            (legacy_manifest.get("signature") or {}).get("source_offset_seconds") or 0.0
        ),
        transcript=transcript,
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "started_at": legacy_manifest.get("started_at"),
        "finished_at": legacy_manifest.get("finished_at"),
        "elapsed_seconds": legacy_manifest.get("elapsed_seconds"),
        "source": source,
        "identity": identity,
        "signature": legacy_manifest.get("signature"),
        "result": {**result, "segment_count": len(segments), "quality_warnings": []},
        "runtime": {"audio2md": version("audio2md")},
        "migration": {
            "from": "course-knowledge-pipeline/run.json",
            "preserved_transcription": True,
        },
    }
    write_json_atomic(output_dir / "segments.json", segment_payload)
    write_text_atomic(output_dir / "transcript.md", markdown)
    write_json_atomic(output_dir / "manifest.json", manifest)
    return {"status": "completed", "segment_count": len(segments), "output_dir": str(output_dir)}
