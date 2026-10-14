from pathlib import Path

import pytest

import audio2md.cli as cli
from audio2md.cli import slugify, title_from_stem
from audio2md.models import Segment, Transcript, TranscriptionConfig
from audio2md.pipeline import run_transcription, validate_slug
from audio2md.exceptions import InvalidTranscriptError
from audio2md.validation import validate_transcript


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, source: Path, config: TranscriptionConfig) -> Transcript:
        self.calls += 1
        return Transcript(
            language="en",
            language_probability=0.99,
            duration_seconds=2.0,
            segments=(Segment(0.25, 1.75, "LEED and ASHRAE."),),
        )


def test_pipeline_writes_outputs_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp3"
    source.write_bytes(b"fake media")
    fake = FakeTranscriber()
    config = TranscriptionConfig(initial_prompt="LEED, ASHRAE", hotwords="LEED, ASHRAE")

    first = run_transcription(
        source=source,
        output_root=tmp_path / "output",
        collection_id="collection-one",
        item_id="item-one",
        title="Lesson One",
        config=config,
        source_offset_seconds=120,
        transcriber=fake,
    )
    second = run_transcription(
        source=source,
        output_root=tmp_path / "output",
        collection_id="collection-one",
        item_id="item-one",
        title="Lesson One",
        config=config,
        source_offset_seconds=120,
        transcriber=fake,
    )

    output = tmp_path / "output" / "collection-one" / "item-one"
    assert first["status"] == "completed"
    assert second["status"] == "skipped"
    assert fake.calls == 1
    assert (output / "manifest.json").is_file()
    assert (output / "segments.json").is_file()
    assert "S000001 | 00:02:00.250" in (output / "transcript.md").read_text(encoding="utf-8")
    assert '"id": "S000001"' in (output / "segments.json").read_text(encoding="utf-8")
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert '"hotwords": "LEED, ASHRAE"' in manifest
    assert '"audio2md": "0.1.0"' in manifest
    assert '"quality_warnings": []' in manifest


def test_pipeline_without_collection_writes_directly_under_item(tmp_path: Path) -> None:
    source = tmp_path / "Interview.mp3"
    source.write_bytes(b"fake media")

    result = run_transcription(
        source=source,
        output_root=tmp_path / "output",
        collection_id=None,
        item_id="interview",
        title="Interview",
        config=TranscriptionConfig(),
        transcriber=FakeTranscriber(),
    )

    output = tmp_path / "output" / "interview"
    assert result["output_dir"] == str(output)
    assert (output / "manifest.json").is_file()
    assert not (tmp_path / "output" / "None").exists()


def test_title_is_part_of_render_signature(tmp_path: Path) -> None:
    source = tmp_path / "interview.mp3"
    source.write_bytes(b"fake media")
    fake = FakeTranscriber()
    common = {
        "source": source,
        "output_root": tmp_path / "output",
        "collection_id": None,
        "item_id": "interview",
        "config": TranscriptionConfig(),
        "transcriber": fake,
    }
    run_transcription(**common, title="First title")
    result = run_transcription(**common, title="Changed title")
    assert result["status"] == "completed"
    assert fake.calls == 2
    assert "# Changed title" in (tmp_path / "output/interview/transcript.md").read_text()


@pytest.mark.parametrize(
    ("stem", "item_id", "title"),
    [
        ("My Great Interview", "my-great-interview", "My Great Interview"),
        ("sesión_01-final", "sesion-01-final", "sesión 01 final"),
    ],
)
def test_identity_inference_from_stem(stem: str, item_id: str, title: str) -> None:
    assert slugify(stem) == item_id
    assert title_from_stem(stem) == title


def test_cli_infers_identity_and_passes_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "Expert Interview 01.mp3"
    captured: dict[str, object] = {}

    def fake_run_transcription(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "completed", "quality_warnings": []}

    monkeypatch.setattr(cli, "run_transcription", fake_run_transcription)

    assert cli.main([str(source), "-o", str(tmp_path / "output")]) == 0
    assert captured["item_id"] == "expert-interview-01"
    assert captured["title"] == "Expert Interview 01"
    assert captured["collection_id"] is None
    assert captured["output_root"] == tmp_path / "output"


@pytest.mark.parametrize("value", ["UPPER", "has spaces", "under_score", "-edge"])
def test_invalid_slugs_fail(value: str) -> None:
    with pytest.raises(ValueError):
        validate_slug(value, "collection-id")


def test_empty_transcript_is_a_quality_warning() -> None:
    warnings = validate_transcript(Transcript("en", 0.9, 0.0, ()))
    assert [warning.code for warning in warnings] == ["empty_transcript"]


def test_invalid_segment_interval_is_rejected() -> None:
    transcript = Transcript("en", 0.9, 2.0, (Segment(2.0, 1.0, "bad"),))
    with pytest.raises(InvalidTranscriptError, match="invalid interval"):
        validate_transcript(transcript)


def test_cli_returns_two_for_completed_output_with_quality_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "silence.mp3"
    source.write_bytes(b"fake")
    monkeypatch.setattr(
        cli,
        "run_transcription",
        lambda **kwargs: {
            "status": "completed",
            "quality_warnings": [{"code": "empty_transcript", "message": "No speech"}],
        },
    )
    assert cli.main([str(source), "-o", str(tmp_path / "output")]) == 2


def test_cli_reports_expected_errors_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.mp3"
    assert cli.main([str(missing), "-o", str(tmp_path / "output")]) == 1
    assert "audio2md: error: Media file not found" in capsys.readouterr().err
