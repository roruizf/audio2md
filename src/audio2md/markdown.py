from __future__ import annotations

import json

from .models import Transcript


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_transcript_markdown(
    *,
    collection_id: str | None,
    item_id: str,
    title: str,
    source_name: str,
    source_sha256: str,
    source_offset_seconds: float,
    transcript: Transcript,
) -> str:
    lines = [
        "---",
        "type: source-transcript",
        "schema_version: 1",
        f"collection_id: {yaml_string(collection_id) if collection_id is not None else 'null'}",
        f"item_id: {yaml_string(item_id)}",
        f"title: {yaml_string(title)}",
        f"source_name: {yaml_string(source_name)}",
        f"source_sha256: {yaml_string(source_sha256)}",
        f"language: {yaml_string(transcript.language)}",
        f"source_offset_seconds: {source_offset_seconds:.3f}",
        f"duration_seconds: {transcript.duration_seconds:.3f}",
        "---",
        "",
        f"# {title}",
        "",
        "> Timestamps refer to the original source timeline.",
        "",
        "## Transcript",
        "",
    ]
    for index, segment in enumerate(transcript.segments, start=1):
        start = format_timestamp(segment.start + source_offset_seconds)
        end = format_timestamp(segment.end + source_offset_seconds)
        lines.append(f"[S{index:06d} | {start} → {end}] {segment.text.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
