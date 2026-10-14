from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .manifest import read_json, write_json_atomic
from .models import Segment, Transcript


ChunkTranscriber = Callable[[np.ndarray, str | None], Transcript]
CHECKPOINT_SCHEMA_VERSION = 1


def _signature_digest(signature: dict[str, object]) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_chunk(path: Path, digest: str, index: int) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("signature_digest") != digest or payload.get("index") != index:
        return None
    if not isinstance(payload.get("segments"), list):
        return None
    return payload


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split()).strip(" .,;:!?")


def _merge_suffix_prefix(previous: Segment, current: Segment) -> Segment | None:
    previous_words = previous.text.split()
    current_words = current.text.split()
    previous_keys = [re.sub(r"\W+", "", word.casefold()) for word in previous_words]
    current_keys = [re.sub(r"\W+", "", word.casefold()) for word in current_words]
    maximum = min(len(previous_keys), len(current_keys))
    for size in range(maximum, 3, -1):
        if previous_keys[-size:] == current_keys[:size]:
            text = " ".join(previous_words + current_words[size:])
            return Segment(previous.start, max(previous.end, current.end), text)
    return None


def _deduplicate_boundary_segments(segments: list[Segment]) -> list[Segment]:
    deduplicated: list[Segment] = []
    for segment in segments:
        if not deduplicated:
            deduplicated.append(segment)
            continue
        previous = deduplicated[-1]
        previous_text = _normalized_text(previous.text)
        current_text = _normalized_text(segment.text)
        overlaps = segment.start < previous.end + 0.1
        if overlaps and previous_text and previous_text in current_text:
            deduplicated[-1] = segment
            continue
        if overlaps and current_text and current_text in previous_text:
            continue
        if overlaps:
            merged = _merge_suffix_prefix(previous, segment)
            if merged is not None:
                deduplicated[-1] = merged
                continue
        if current_text == previous_text and abs(segment.start - previous.start) < 0.05:
            continue
        deduplicated.append(segment)
    return deduplicated


def transcribe_in_chunks(
    *,
    audio: np.ndarray,
    sample_rate: int,
    language: str | None,
    checkpoint_dir: Path,
    signature: dict[str, object],
    chunk_seconds: float,
    overlap_seconds: float,
    transcribe_chunk: ChunkTranscriber,
) -> Transcript:
    """Transcribe deterministic PCM chunks and reuse valid completed checkpoints."""
    if sample_rate <= 0:
        raise ValueError("sample-rate must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk-seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("chunk-overlap-seconds must be non-negative and less than chunk-seconds")
    if audio.ndim != 1:
        raise ValueError("decoded audio must be mono")

    duration = len(audio) / sample_rate
    if not len(audio):
        return Transcript(language or "unknown", 0.0, 0.0, ())

    chunk_count = math.ceil(duration / chunk_seconds)
    digest = _signature_digest(signature)
    chunks_dir = checkpoint_dir / "chunks"
    state_path = checkpoint_dir / "state.json"
    completed: list[int] = []
    detected_language = language
    language_probability = 1.0 if language else 0.0
    merged: list[Segment] = []

    for index in range(chunk_count):
        core_start = index * chunk_seconds
        core_end = min(duration, core_start + chunk_seconds)
        input_start = max(0.0, core_start - overlap_seconds)
        input_end = min(duration, core_end + overlap_seconds)
        path = chunks_dir / f"{index:06d}.json"
        payload = _load_chunk(path, digest, index)

        if payload is None:
            start_sample = round(input_start * sample_rate)
            end_sample = round(input_end * sample_rate)
            chunk_result = transcribe_chunk(audio[start_sample:end_sample], detected_language)
            if detected_language is None:
                detected_language = chunk_result.language
                language_probability = chunk_result.language_probability
            payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "signature_digest": digest,
                "index": index,
                "core_start": core_start,
                "core_end": core_end,
                "input_start": input_start,
                "input_end": input_end,
                "language": chunk_result.language,
                "language_probability": chunk_result.language_probability,
                "segments": [segment.as_dict() for segment in chunk_result.segments],
            }
            write_json_atomic(path, payload)
        elif detected_language is None:
            detected_language = str(payload.get("language") or "unknown")
            language_probability = float(payload.get("language_probability") or 0.0)

        raw_segments = payload["segments"]
        assert isinstance(raw_segments, list)
        for raw in raw_segments:
            if not isinstance(raw, dict):
                continue
            segment = Segment(float(raw["start"]), float(raw["end"]), str(raw["text"]))
            absolute = Segment(
                start=segment.start + input_start,
                end=segment.end + input_start,
                text=segment.text,
            )
            midpoint = (absolute.start + absolute.end) / 2
            owns_segment = core_start <= midpoint < core_end
            if index == chunk_count - 1 and midpoint == core_end:
                owns_segment = True
            if owns_segment:
                merged.append(absolute)

        completed.append(index)
        write_json_atomic(
            state_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "signature_digest": digest,
                "sample_rate": sample_rate,
                "sample_count": len(audio),
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": overlap_seconds,
                "chunk_count": chunk_count,
                "completed_chunks": completed,
            },
        )

    merged.sort(key=lambda segment: (segment.start, segment.end))
    deduplicated = _deduplicate_boundary_segments(merged)
    return Transcript(
        language=detected_language or "unknown",
        language_probability=language_probability,
        duration_seconds=duration,
        segments=tuple(deduplicated),
    )
