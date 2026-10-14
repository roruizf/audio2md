from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import transcribe_in_chunks
from .models import Segment, Transcript, TranscriptionConfig


def _to_transcript(raw_segments: Any, info: Any) -> Transcript:
    segments = tuple(
        Segment(start=item.start, end=item.end, text=item.text.strip())
        for item in raw_segments
        if item.text.strip()
    )
    return Transcript(
        language=info.language,
        language_probability=info.language_probability,
        duration_seconds=info.duration,
        segments=segments,
    )


def transcribe_media(
    source: Path,
    config: TranscriptionConfig,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_signature: dict[str, object] | None = None,
    chunk_seconds: float = 600.0,
    overlap_seconds: float = 30.0,
) -> Transcript:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
    )
    def invoke(media: str | np.ndarray, language: str | None) -> Transcript:
        raw_segments, info = model.transcribe(
            media,
            language=language,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            condition_on_previous_text=True,
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
        )
        return _to_transcript(raw_segments, info)

    if checkpoint_dir is None:
        return invoke(str(source), config.language)

    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(source), sampling_rate=16000)
    return transcribe_in_chunks(
        audio=audio,
        sample_rate=16000,
        language=config.language,
        checkpoint_dir=checkpoint_dir,
        signature=checkpoint_signature or {},
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        transcribe_chunk=invoke,
    )
