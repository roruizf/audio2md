from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import InvalidBundleError
from .manifest import sha256_file, write_json_atomic


SEGMENT_ID_PATTERN = re.compile(r"^S\d{6}$")
CITATION_ID_PATTERN = re.compile(r"\bS\d{6}\b")
PACKETIZER_VERSION = 1


@dataclass(frozen=True)
class CitationIssue:
    code: str
    line: int
    message: str

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "line": self.line, "message": self.message}


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidBundleError(f"Cannot read valid JSON from {path}: {error}") from error


def load_segments(bundle_dir: Path) -> list[dict[str, object]]:
    path = bundle_dir.expanduser().resolve() / "segments.json"
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise InvalidBundleError(f"Expected a JSON array in {path}")
    segments: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise InvalidBundleError(f"Segment {index} is not an object")
        segment_id = raw.get("id")
        if not isinstance(segment_id, str) or not SEGMENT_ID_PATTERN.fullmatch(segment_id):
            raise InvalidBundleError(f"Segment {index} has an invalid or missing id")
        if segment_id in seen:
            raise InvalidBundleError(f"Duplicate segment id: {segment_id}")
        if not isinstance(raw.get("text"), str):
            raise InvalidBundleError(f"Segment {segment_id} has no text")
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidBundleError(f"Segment {segment_id} has invalid timestamps") from error
        if start < 0 or end < start:
            raise InvalidBundleError(f"Segment {segment_id} has an invalid interval")
        seen.add(segment_id)
        segments.append({"id": segment_id, "start": start, "end": end, "text": raw["text"]})
    return segments


def build_evidence_packets(
    segments: list[dict[str, object]], *, max_words: int, overlap_segments: int
) -> list[dict[str, object]]:
    if max_words <= 0:
        raise ValueError("max-words must be positive")
    if overlap_segments < 0:
        raise ValueError("overlap-segments cannot be negative")
    cores: list[list[dict[str, object]]] = []
    start = 0
    while start < len(segments):
        end = start
        core_word_count = 0
        while end < len(segments):
            segment_words = len(str(segments[end]["text"]).split())
            if end > start and core_word_count + segment_words > max_words:
                break
            core_word_count += segment_words
            end += 1
        cores.append(segments[start:end])
        start = end

    packets: list[dict[str, object]] = []
    previous_core: list[dict[str, object]] = []
    for core in cores:
        overlap = previous_core[-overlap_segments:] if overlap_segments else []
        packet_segments = overlap + core
        packets.append(
            {
                "id": f"P{len(packets) + 1:05d}",
                "segment_ids": [segment["id"] for segment in packet_segments],
                "word_count": sum(len(str(segment["text"]).split()) for segment in packet_segments),
                "core_segment_ids": [segment["id"] for segment in core],
                "overlap_segment_ids": [segment["id"] for segment in overlap],
                "segments": packet_segments,
            }
        )
        previous_core = core
    return packets


def prepare_evidence(
    bundle_dir: Path, *, max_words: int = 1200, overlap_segments: int = 2, force: bool = False
) -> dict[str, object]:
    bundle_dir = bundle_dir.expanduser().resolve()
    segments_path = bundle_dir / "segments.json"
    segments = load_segments(bundle_dir)
    signature = {
        "packetizer_version": PACKETIZER_VERSION,
        "segments_sha256": sha256_file(segments_path),
        "max_words": max_words,
        "overlap_segments": overlap_segments,
    }
    output_path = bundle_dir / "study" / "evidence-packets.json"
    if output_path.is_file() and not force:
        previous = _read_json(output_path)
        if isinstance(previous, dict) and previous.get("signature") == signature:
            return {"status": "skipped", "output": str(output_path)}
    packets = build_evidence_packets(
        segments, max_words=max_words, overlap_segments=overlap_segments
    )
    write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "signature": signature,
            "segment_count": len(segments),
            "packet_count": len(packets),
            "packets": packets,
        },
    )
    return {
        "status": "completed",
        "output": str(output_path),
        "segment_count": len(segments),
        "packet_count": len(packets),
    }


def validate_citations(
    markdown: str, valid_ids: set[str], *, require_citations: bool = True
) -> tuple[CitationIssue, ...]:
    issues: list[CitationIssue] = []
    in_frontmatter = False
    in_fence = False
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if line_number == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith(("#", ">")):
            continue
        citations = set(CITATION_ID_PATTERN.findall(stripped))
        for citation in sorted(citations - valid_ids):
            issues.append(
                CitationIssue(
                    "citation_not_found",
                    line_number,
                    f"Citation {citation} does not exist in segments.json",
                )
            )
        content = re.sub(r"\[[^]]*S\d{6}[^]]*\]", "", stripped).strip()
        has_prose = bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", content))
        if require_citations and has_prose and not citations:
            issues.append(
                CitationIssue(
                    "missing_citation",
                    line_number,
                    "Content line has no segment citation",
                )
            )
    return tuple(issues)
