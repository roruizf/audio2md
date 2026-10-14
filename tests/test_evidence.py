import json
from pathlib import Path

from audio2md.evidence import build_evidence_packets, prepare_evidence, validate_citations


def sample_segments() -> list[dict[str, object]]:
    return [
        {"id": f"S{index:06d}", "start": index - 1.0, "end": float(index), "text": text}
        for index, text in enumerate(
            ("one two three", "four five six", "seven eight nine", "ten eleven twelve"),
            start=1,
        )
    ]


def test_packets_are_deterministic_and_overlap_by_segment() -> None:
    packets = build_evidence_packets(sample_segments(), max_words=6, overlap_segments=1)
    assert [packet["segment_ids"] for packet in packets] == [
        ["S000001", "S000002"],
        ["S000002", "S000003", "S000004"],
    ]
    assert packets[1]["core_segment_ids"] == ["S000003", "S000004"]
    assert packets[1]["overlap_segment_ids"] == ["S000002"]


def test_prepare_evidence_writes_and_skips_matching_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "segments.json").write_text(json.dumps(sample_segments()), encoding="utf-8")
    first = prepare_evidence(bundle, max_words=6, overlap_segments=1)
    second = prepare_evidence(bundle, max_words=6, overlap_segments=1)
    assert first["status"] == "completed"
    assert second["status"] == "skipped"
    payload = json.loads((bundle / "knowledge/evidence-packets.json").read_text())
    assert payload["packet_count"] == 2


def test_citation_validator_reports_missing_and_unknown_ids() -> None:
    issues = validate_citations(
        "# Manual\n\nSupported fact. [S000001]\n\nUnsupported fact.\n\nWrong. [S999999]\n",
        {"S000001"},
    )
    assert [(issue.code, issue.line) for issue in issues] == [
        ("missing_citation", 5),
        ("citation_not_found", 7),
    ]


def test_citation_validator_treats_wrapped_markdown_as_logical_blocks() -> None:
    markdown = """# Manual

A factual paragraph wraps across
two source lines. [S000001]

- A checklist item also wraps
  before its citation. [S000001]

| Topic | Evidence |
| --- | --- |
| Appendix G | Discussed. [S000001] |
"""
    assert validate_citations(markdown, {"S000001"}) == ()
