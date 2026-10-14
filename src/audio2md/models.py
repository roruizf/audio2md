from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 6
    beam_size: int = 5
    vad_filter: bool = True
    language: str | None = "en"
    initial_prompt: str | None = None
    hotwords: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Transcript:
    language: str
    language_probability: float
    duration_seconds: float
    segments: tuple[Segment, ...]
