from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .exceptions import InvalidTranscriptError
from .models import Transcript


@dataclass(frozen=True)
class QualityWarning:
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_transcript(transcript: Transcript) -> tuple[QualityWarning, ...]:
    """Reject structurally unsafe ASR output and report non-fatal quality signals."""
    if not math.isfinite(transcript.duration_seconds) or transcript.duration_seconds < 0:
        raise InvalidTranscriptError("Transcript duration must be a finite non-negative number")
    if not math.isfinite(transcript.language_probability):
        raise InvalidTranscriptError("Language probability must be finite")

    warnings: list[QualityWarning] = []
    if not transcript.segments:
        warnings.append(
            QualityWarning("empty_transcript", "The transcription produced no speech segments")
        )
    if transcript.language_probability < 0.5:
        warnings.append(
            QualityWarning(
                "low_language_probability",
                f"Detected language confidence is low ({transcript.language_probability:.3f})",
            )
        )

    previous_start = -1.0
    empty_text_count = 0
    for index, segment in enumerate(transcript.segments):
        if not math.isfinite(segment.start) or not math.isfinite(segment.end):
            raise InvalidTranscriptError(f"Segment {index} has a non-finite timestamp")
        if segment.start < 0 or segment.end < segment.start:
            raise InvalidTranscriptError(
                f"Segment {index} has an invalid interval: {segment.start}..{segment.end}"
            )
        if segment.start < previous_start:
            warnings.append(
                QualityWarning(
                    "non_monotonic_timestamps",
                    f"Segment {index} starts before the previous segment",
                )
            )
        previous_start = segment.start
        if not segment.text.strip():
            empty_text_count += 1

    if empty_text_count:
        warnings.append(
            QualityWarning(
                "empty_segments",
                f"{empty_text_count} segment(s) contain no text",
            )
        )
    return tuple(warnings)
