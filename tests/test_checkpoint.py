from pathlib import Path

import numpy as np
import pytest

from audio2md.checkpoint import _deduplicate_boundary_segments, transcribe_in_chunks
from audio2md.models import Segment, Transcript


def test_checkpoint_resumes_after_interruption(tmp_path: Path) -> None:
    audio = np.zeros(250, dtype=np.float32)
    checkpoint_dir = tmp_path / ".checkpoint"
    first_calls = 0

    def interrupted(chunk: np.ndarray, language: str | None) -> Transcript:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            raise RuntimeError("simulated interruption")
        return Transcript("en", 0.99, len(chunk) / 10, (Segment(3.0, 4.0, "speech"),))

    kwargs = {
        "audio": audio,
        "sample_rate": 10,
        "language": None,
        "checkpoint_dir": checkpoint_dir,
        "signature": {"source_sha256": "abc", "model": "fake"},
        "chunk_seconds": 10.0,
        "overlap_seconds": 2.0,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        transcribe_in_chunks(**kwargs, transcribe_chunk=interrupted)

    assert (checkpoint_dir / "chunks" / "000000.json").is_file()
    resumed_calls = 0

    def resumed(chunk: np.ndarray, language: str | None) -> Transcript:
        nonlocal resumed_calls
        resumed_calls += 1
        assert language == "en"
        return Transcript("en", 0.99, len(chunk) / 10, (Segment(3.0, 4.0, "speech"),))

    result = transcribe_in_chunks(**kwargs, transcribe_chunk=resumed)
    assert resumed_calls == 2
    assert [round(segment.start, 1) for segment in result.segments] == [3.0, 11.0, 21.0]
    assert result.duration_seconds == 25.0


def test_changed_signature_does_not_reuse_chunks(tmp_path: Path) -> None:
    audio = np.zeros(50, dtype=np.float32)
    calls = 0

    def transcriber(chunk: np.ndarray, language: str | None) -> Transcript:
        nonlocal calls
        calls += 1
        return Transcript("en", 1.0, 5.0, (Segment(1.0, 2.0, "text"),))

    common = {
        "audio": audio,
        "sample_rate": 10,
        "language": "en",
        "checkpoint_dir": tmp_path / ".checkpoint",
        "chunk_seconds": 5.0,
        "overlap_seconds": 1.0,
        "transcribe_chunk": transcriber,
    }
    transcribe_in_chunks(**common, signature={"source": "one"})
    transcribe_in_chunks(**common, signature={"source": "two"})
    assert calls == 2


def test_boundary_dedup_prefers_segment_with_complete_containing_text() -> None:
    result = _deduplicate_boundary_segments(
        [
            Segment(54.63, 61.43, "Our learning objectives are to identify project opportunities"),
            Segment(
                55.0,
                80.0,
                "Our learning objectives are to identify project opportunities best suited for Appendix G",
            ),
        ]
    )
    assert result == [
        Segment(
            55.0,
            80.0,
            "Our learning objectives are to identify project opportunities best suited for Appendix G",
        )
    ]


def test_boundary_dedup_merges_repeated_suffix_and_prefix() -> None:
    result = _deduplicate_boundary_segments(
        [
            Segment(
                54.63,
                61.43,
                "our learning objectives. Our learning objectives are to identify project opportunities best suited",
            ),
            Segment(
                55.0,
                80.0,
                "Our learning objectives are to identify project opportunities best suited for Appendix G",
            ),
        ]
    )
    assert result == [
        Segment(
            54.63,
            80.0,
            "our learning objectives. Our learning objectives are to identify project opportunities best suited for Appendix G",
        )
    ]
