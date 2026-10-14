from __future__ import annotations

import json
from pathlib import Path

from .evidence import load_segments, validate_citations
from .exceptions import InvalidBundleError
from .llm import LlmAdapter, LlmRequest
from .manifest import sha256_file, write_json_atomic, write_text_atomic


EXTRACT_PROMPT = """Extract grounded claims from the supplied transcript evidence.
Return JSON only with arrays: claims, exclusions, uncertainties, glossary.
Every claim must contain claim_id, text, evidence, and classification: source, inference, or uncertain.
Use only supplied segments. Preserve numbers, standards, versions, conditions and exceptions.
Do not add outside knowledge. Put promotion and logistics in exclusions.
Record likely ASR errors in uncertainties instead of silently correcting them."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "array", "items": {"type": "object"}}
        for key in ("claims", "exclusions", "uncertainties", "glossary")
    },
    "required": ["claims", "exclusions", "uncertainties", "glossary"],
    "additionalProperties": False,
}


def _citation(item: dict[str, object]) -> str:
    evidence = item.get("evidence") or []
    return "[" + ", ".join(str(value) for value in evidence) + "]"


def _validate_ledger(payload: object, valid_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    if not isinstance(payload, dict):
        raise InvalidBundleError("Model response is not a JSON object")
    validated: dict[str, list[dict[str, object]]] = {}
    for group in ("claims", "exclusions", "uncertainties", "glossary"):
        items = payload.get(group)
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise InvalidBundleError(f"Model response has an invalid {group} array")
        for item in items:
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise InvalidBundleError(f"An item in {group} has no evidence")
            unknown = set(map(str, evidence)) - valid_ids
            if unknown:
                raise InvalidBundleError(f"Unknown evidence IDs in {group}: {sorted(unknown)}")
        validated[group] = items
    return validated


def extract_and_render(
    bundle_dir: Path,
    adapter: LlmAdapter,
    *,
    force: bool = False,
) -> dict[str, object]:
    bundle_dir = bundle_dir.expanduser().resolve()
    evidence_path = bundle_dir / "study" / "evidence-packets.json"
    if not evidence_path.is_file():
        raise InvalidBundleError("Run `audio2md-study prepare BUNDLE` first")
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    packets = evidence_payload.get("packets") if isinstance(evidence_payload, dict) else None
    if not isinstance(packets, list):
        raise InvalidBundleError("evidence-packets.json has an invalid shape")
    study_dir = bundle_dir / "study"
    request_signature = {
        "prompt_version": 1,
        "evidence_packets_sha256": sha256_file(evidence_path),
        "adapter": type(adapter).__name__,
        "model": getattr(adapter, "model", "unknown"),
    }
    run_path = study_dir / "claims-run.json"
    expected_outputs = [study_dir / name for name in ("claims.json", "manual.md", "summary.md", "checklist.md")]
    if run_path.is_file() and not force and all(path.is_file() for path in expected_outputs):
        previous_run = json.loads(run_path.read_text(encoding="utf-8"))
        if previous_run.get("signature") == request_signature:
            return {
                "status": "skipped",
                "reason": "matching evidence, prompt, adapter and model already completed",
                "output_dir": str(study_dir),
            }
    sanitized = {
        "packets": [
            {"id": packet.get("id"), "segments": packet.get("segments")}
            for packet in packets
            if isinstance(packet, dict)
        ]
    }
    response = adapter.complete(
        LlmRequest(
            "extract-claims",
            EXTRACT_PROMPT,
            json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")),
            EXTRACT_SCHEMA,
        )
    )
    try:
        raw_ledger = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise InvalidBundleError(f"Model returned invalid JSON: {error}") from error
    segments = load_segments(bundle_dir)
    valid_ids = {str(segment["id"]) for segment in segments}
    ledger = _validate_ledger(raw_ledger, valid_ids)

    write_json_atomic(study_dir / "claims.json", ledger)
    run = {
        "schema_version": 1,
        "status": "completed",
        "provider": response.provider,
        "model": response.model,
        "cost_usd": response.cost_usd,
        "tokens": response.usage,
        "signature": request_signature,
        "input": {
            "evidence_packets_sha256": sha256_file(evidence_path),
            "packet_count": len(packets),
            "segment_count": len(valid_ids),
            "sent_fields": ["id", "start", "end", "text"],
        },
    }
    write_json_atomic(study_dir / "claims-run.json", run)

    claims = ledger["claims"]
    uncertainties = ledger["uncertainties"]
    exclusions = ledger["exclusions"]
    manual_lines = ["# Grounded study manual", "", "## Source-grounded points", ""]
    manual_lines += [f"- {item['text']} {_citation(item)}" for item in claims]
    manual_lines += ["", "## Uncertainties", ""]
    manual_lines += [f"- {item['text']} {_citation(item)}" for item in uncertainties]
    manual_lines += ["", "## Excluded material", ""]
    manual_lines += [
        f"- {item.get('reason') or item.get('text')} {_citation(item)}" for item in exclusions
    ]
    summary_lines = ["# Grounded summary", ""] + [
        f"{item['text']} {_citation(item)}" for item in claims
    ]
    checklist_claims = [
        item for item in claims if item.get("classification") == "source"
    ]
    checklist_lines = ["# Study checklist", ""] + [
        f"- [ ] Review and explain: {item['text']} {_citation(item)}"
        for item in checklist_claims
    ]
    outputs = {
        "manual.md": "\n".join(manual_lines).rstrip() + "\n",
        "summary.md": "\n".join(summary_lines).rstrip() + "\n",
        "checklist.md": "\n".join(checklist_lines).rstrip() + "\n",
    }
    for name, content in outputs.items():
        issues = validate_citations(content, valid_ids)
        if issues:
            raise InvalidBundleError(f"Generated {name} failed citation validation")
        write_text_atomic(study_dir / name, content)
    return {
        "status": "completed",
        "claim_count": len(claims),
        "uncertainty_count": len(uncertainties),
        "cost_usd": response.cost_usd,
        "output_dir": str(study_dir),
    }
