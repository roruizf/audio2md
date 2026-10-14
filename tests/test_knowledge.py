import json
from pathlib import Path

from audio2md.llm import LlmRequest, LlmResponse
from audio2md.legacy import migrate_legacy_bundle
from audio2md.knowledge import compose_knowledge, extract_and_render


class FakeAdapter:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls += 1
        payload = {
            "claims": [
                {
                    "claim_id": "C00001",
                    "text": "Appendix G is discussed.",
                    "evidence": ["S000001"],
                    "classification": "source",
                }
            ],
            "exclusions": [],
            "uncertainties": [],
            "glossary": [],
        }
        return LlmResponse(json.dumps(payload), "fake-model", "fake", {"total": 10}, 0.0)


class FakeCompositionAdapter:
    model = "fake-compose-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls += 1
        payload = {
            "manual": "# Manual\n\nAppendix G is discussed. [S000001]",
            "summary": "# Summary\n\nAppendix G is discussed. [S000001]",
            "checklist": "# Checklist\n\n- [ ] Verify Appendix G. [S000001]",
        }
        return LlmResponse(json.dumps(payload), self.model, "fake", {"total": 20}, 0.0)


def test_extract_and_render_creates_cited_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    knowledge = bundle / "knowledge"
    knowledge.mkdir(parents=True)
    segments = [{"id": "S000001", "start": 0, "end": 1, "text": "Appendix G."}]
    (bundle / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    evidence = {
        "packets": [{"id": "P00001", "segments": segments}],
    }
    (knowledge / "evidence-packets.json").write_text(json.dumps(evidence), encoding="utf-8")
    adapter = FakeAdapter()
    result = extract_and_render(bundle, adapter)
    second = extract_and_render(bundle, adapter)
    assert result["claim_count"] == 1
    assert second["status"] == "skipped"
    assert adapter.calls == 1
    assert "[S000001]" in (knowledge / "manual.md").read_text()
    assert json.loads((knowledge / "claims-run.json").read_text())["provider"] == "fake"


def test_migrate_legacy_bundle_adds_stable_ids(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run.json").write_text(
        json.dumps(
            {
                "source": {"name": "lesson.mp3", "sha256": "abc"},
                "identity": {"collection_id": "course", "item_id": "lesson", "title": "Lesson"},
                "signature": {"source_offset_seconds": 0},
                "result": {"language": "en", "language_probability": 1, "duration_seconds": 2},
            }
        ),
        encoding="utf-8",
    )
    (legacy / "segments.json").write_text(
        json.dumps([{"start": 0, "end": 2, "text": "Evidence."}]), encoding="utf-8"
    )
    output = tmp_path / "output"
    result = migrate_legacy_bundle(legacy, output)
    assert result["segment_count"] == 1
    assert json.loads((output / "segments.json").read_text())[0]["id"] == "S000001"
    assert "[S000001 |" in (output / "transcript.md").read_text()


def test_compose_knowledge_creates_compact_cited_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    knowledge = bundle / "knowledge"
    knowledge.mkdir(parents=True)
    segments = [{"id": "S000001", "start": 0, "end": 1, "text": "Appendix G."}]
    (bundle / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    ledger = {
        "claims": [
            {
                "claim_id": "C00001",
                "text": "Appendix G is discussed.",
                "evidence": ["S000001"],
                "classification": "source",
            }
        ],
        "exclusions": [],
        "uncertainties": [],
        "glossary": [],
    }
    (knowledge / "claims.json").write_text(json.dumps(ledger), encoding="utf-8")
    adapter = FakeCompositionAdapter()
    result = compose_knowledge(bundle, adapter)
    second = compose_knowledge(bundle, adapter)
    assert result["artifacts"] == ["checklist.md", "manual.md", "summary.md"]
    assert second["status"] == "skipped"
    assert adapter.calls == 1
    assert "Verify Appendix G" in (knowledge / "checklist.md").read_text()
    assert json.loads((knowledge / "citation-report.json").read_text())["status"] == "pass"
    assert json.loads((knowledge / "compose-attempt.json").read_text())["status"] == "accepted"


def test_compose_knowledge_preserves_rejected_attempt(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    knowledge = bundle / "knowledge"
    knowledge.mkdir(parents=True)
    segments = [{"id": "S000001", "start": 0, "end": 1, "text": "Appendix G."}]
    (bundle / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    ledger = {
        "claims": [
            {
                "claim_id": "C00001",
                "text": "Appendix G is discussed.",
                "evidence": ["S000001"],
                "classification": "source",
            }
        ],
        "exclusions": [],
        "uncertainties": [],
        "glossary": [],
    }
    (knowledge / "claims.json").write_text(json.dumps(ledger), encoding="utf-8")

    class UncitedAdapter(FakeCompositionAdapter):
        def complete(self, request: LlmRequest) -> LlmResponse:
            response = super().complete(request)
            payload = json.loads(response.text)
            payload["summary"] = "# Summary\n\nAn uncited statement."
            return LlmResponse(json.dumps(payload), self.model, "fake", {"total": 20}, 0.0)

    try:
        compose_knowledge(bundle, UncitedAdapter())
    except Exception as error:
        assert "failed citation validation" in str(error)
    else:
        raise AssertionError("Expected citation validation to fail")
    attempt = json.loads((knowledge / "compose-attempt.json").read_text())
    assert attempt["status"] == "rejected"
    assert attempt["validation"]["artifacts"]["summary.md"]["issue_count"] == 1
