from __future__ import annotations

import json
import re
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
Record likely ASR errors in uncertainties instead of silently correcting them.
Treat claims as statements made by the source, not independently verified facts. Preserve explicit speaker
attribution and temporal context for opinions, forecasts, recommendations, disputed claims, and time-sensitive
information. Never convert financial, medical, legal, or other consequential commentary into professional advice."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "array", "items": {"type": "object"}}
        for key in ("claims", "exclusions", "uncertainties", "glossary")
    },
    "required": ["claims", "exclusions", "uncertainties", "glossary"],
    "additionalProperties": False,
}

COMPOSE_PROMPT = """Turn the supplied grounded claim ledger into three useful Markdown artifacts.
Return JSON only with the string fields manual, summary, and checklist.

The manual must be organized by concepts rather than transcript order and target 2500-3500 words.
The summary must stand alone and target 500-800 words.
The checklist must contain 20-30 concrete actions, decisions, or verification points.
Do not turn every claim into a separate bullet and do not use generic tasks such as "review and explain".
Every factual paragraph or list item must end with its supporting transcript segment IDs, for example
[S000042, S000043]. Use only source claims for factual prose. Keep uncertainties clearly labeled and do not
silently correct them. Omit exclusions unless they explain a material limitation. Do not add outside knowledge,
cite claim IDs, or wrap the Markdown in code fences. Preserve attribution and temporal context for opinions,
forecasts, recommendations, and time-sensitive claims; do not present consequential commentary as professional
advice. Checklist items must verify or analyze the source's claims. Never instruct the reader to buy, sell, hold,
wait, act, or make a financial, medical, legal, or other consequential decision."""

COMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "string", "minLength": 1}
        for key in ("manual", "summary", "checklist")
    },
    "required": ["manual", "summary", "checklist"],
    "additionalProperties": False,
}

CONSEQUENTIAL_ACTION_PATTERNS = (
    re.compile(r"\bbefore acting\b|\bantes de actuar\b", re.IGNORECASE),
    re.compile(
        r"\bbefore (?:considering|deciding on) (?:any )?(?:purchase|sale)\b|"
        r"\bantes de (?:considerar|decidir) (?:cualquier )?(?:compra|venta)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^- \[ \] (?:buy|sell|hold|wait|anticipate|comprar|vender|mantener|esperar|anticipar)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^- \[ \] (?:verify that nobody buys or sells|verificar que no se (?:compra|vende))\b",
        re.IGNORECASE,
    ),
)


def _citation(item: dict[str, object]) -> str:
    evidence = item.get("evidence") or []
    return "[" + ", ".join(str(value) for value in evidence) + "]"


def _normalize_evidence(evidence: object, group: str) -> list[str]:
    if not isinstance(evidence, list) or not evidence:
        raise InvalidBundleError(f"An item in {group} has no evidence")
    normalized: list[str] = []
    for value in evidence:
        if isinstance(value, str):
            normalized.append(value)
        elif isinstance(value, dict) and isinstance(value.get("segment_id"), str):
            normalized.append(value["segment_id"])
        else:
            raise InvalidBundleError(f"An item in {group} has invalid evidence")
    return normalized


def _validate_ledger(payload: object, valid_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    if not isinstance(payload, dict):
        raise InvalidBundleError("Model response is not a JSON object")
    validated: dict[str, list[dict[str, object]]] = {}
    for group in ("claims", "exclusions", "uncertainties", "glossary"):
        items = payload.get(group)
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise InvalidBundleError(f"Model response has an invalid {group} array")
        for item in items:
            evidence_source = item.get("evidence") or item.get("segment_ids")
            evidence = _normalize_evidence(evidence_source, group)
            unknown = set(evidence) - valid_ids
            if unknown:
                raise InvalidBundleError(f"Unknown evidence IDs in {group}: {sorted(unknown)}")
            item["evidence"] = evidence
            item.pop("segment_ids", None)
            if group == "uncertainties" and not isinstance(item.get("text"), str):
                issue = item.get("issue")
                if not isinstance(issue, str) or not issue.strip():
                    raise InvalidBundleError("An item in uncertainties has no text")
                item["text"] = issue
        validated[group] = items
    return validated


def _validate_composition(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise InvalidBundleError("Model composition is not a JSON object")
    expected = {"manual", "summary", "checklist"}
    if set(payload) != expected:
        raise InvalidBundleError("Model composition must contain only manual, summary, and checklist")
    outputs: dict[str, str] = {}
    for key in sorted(expected):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise InvalidBundleError(f"Model composition has an invalid {key}")
        if "```" in value:
            raise InvalidBundleError(f"Model composition wrapped {key} in a code fence")
        outputs[f"{key}.md"] = value.rstrip() + "\n"
    return outputs


def _consequential_action_issues(content: str) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.startswith("- [ ] "):
            continue
        for pattern in CONSEQUENTIAL_ACTION_PATTERNS:
            if pattern.search(line):
                issues.append({"line": line_number, "text": line})
                break
    return issues


def compose_knowledge(
    bundle_dir: Path,
    adapter: LlmAdapter,
    *,
    force: bool = False,
) -> dict[str, object]:
    bundle_dir = bundle_dir.expanduser().resolve()
    knowledge_dir = bundle_dir / "knowledge"
    claims_path = knowledge_dir / "claims.json"
    if not claims_path.is_file():
        raise InvalidBundleError("Run `audio2md-generate extract BUNDLE` first")
    try:
        raw_ledger = json.loads(claims_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InvalidBundleError(f"claims.json contains invalid JSON: {error}") from error

    segments = load_segments(bundle_dir)
    valid_ids = {str(segment["id"]) for segment in segments}
    ledger = _validate_ledger(raw_ledger, valid_ids)
    ledger_ids = {
        str(segment_id)
        for group in ledger.values()
        for item in group
        for segment_id in item["evidence"]
    }
    request_signature = {
        "prompt_version": 3,
        "claims_sha256": sha256_file(claims_path),
        "adapter": type(adapter).__name__,
        "model": getattr(adapter, "model", "unknown"),
    }
    run_path = knowledge_dir / "compose-run.json"
    output_paths = [knowledge_dir / name for name in ("manual.md", "summary.md", "checklist.md")]
    if run_path.is_file() and not force and all(path.is_file() for path in output_paths):
        previous_run = json.loads(run_path.read_text(encoding="utf-8"))
        if previous_run.get("signature") == request_signature:
            return {
                "status": "skipped",
                "reason": "matching claims, prompt, adapter and model already completed",
                "output_dir": str(knowledge_dir),
            }

    response = adapter.complete(
        LlmRequest(
            "compose-knowledge",
            COMPOSE_PROMPT,
            json.dumps(ledger, ensure_ascii=False, separators=(",", ":")),
            COMPOSE_SCHEMA,
        )
    )
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise InvalidBundleError(f"Model returned invalid composition JSON: {error}") from error
    outputs = _validate_composition(payload)
    validation: dict[str, object] = {"status": "pass", "artifacts": {}}
    for name, content in outputs.items():
        issues = validate_citations(content, ledger_ids)
        action_issues = _consequential_action_issues(content) if name == "checklist.md" else []
        validation["artifacts"][name] = {
            "issue_count": len(issues),
            "issues": [issue.as_dict() for issue in issues],
            "consequential_action_issue_count": len(action_issues),
            "consequential_action_issues": action_issues,
        }
        if issues or action_issues:
            validation["status"] = "failed"
    attempt = {
        "schema_version": 1,
        "status": "accepted" if validation["status"] == "pass" else "rejected",
        "provider": response.provider,
        "model": response.model,
        "cost_usd": response.cost_usd,
        "tokens": response.usage,
        "signature": request_signature,
        "validation": validation,
        "output": payload,
    }
    write_json_atomic(knowledge_dir / "compose-attempt.json", attempt)
    write_json_atomic(knowledge_dir / "citation-report.json", validation)
    if validation["status"] != "pass":
        raise InvalidBundleError("Composed artifacts failed citation or safety validation")

    for name, content in outputs.items():
        write_text_atomic(knowledge_dir / name, content)
    write_json_atomic(
        run_path,
        {
            "schema_version": 1,
            "status": "completed",
            "provider": response.provider,
            "model": response.model,
            "cost_usd": response.cost_usd,
            "tokens": response.usage,
            "signature": request_signature,
            "input": {
                "claims_sha256": sha256_file(claims_path),
                "claim_count": len(ledger["claims"]),
                "sent_fields": ["claims", "exclusions", "uncertainties", "glossary"],
            },
        },
    )
    return {
        "status": "completed",
        "cost_usd": response.cost_usd,
        "output_dir": str(knowledge_dir),
        "artifacts": sorted(outputs),
    }


def extract_and_render(
    bundle_dir: Path,
    adapter: LlmAdapter,
    *,
    force: bool = False,
) -> dict[str, object]:
    bundle_dir = bundle_dir.expanduser().resolve()
    evidence_path = bundle_dir / "knowledge" / "evidence-packets.json"
    if not evidence_path.is_file():
        raise InvalidBundleError("Run `audio2md-generate prepare BUNDLE` first")
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    packets = evidence_payload.get("packets") if isinstance(evidence_payload, dict) else None
    if not isinstance(packets, list):
        raise InvalidBundleError("evidence-packets.json has an invalid shape")
    knowledge_dir = bundle_dir / "knowledge"
    request_signature = {
        "prompt_version": 2,
        "evidence_packets_sha256": sha256_file(evidence_path),
        "adapter": type(adapter).__name__,
        "model": getattr(adapter, "model", "unknown"),
    }
    run_path = knowledge_dir / "claims-run.json"
    expected_outputs = [
        knowledge_dir / name
        for name in ("claims.json", "manual.md", "summary.md", "checklist.md")
    ]
    if run_path.is_file() and not force and all(path.is_file() for path in expected_outputs):
        previous_run = json.loads(run_path.read_text(encoding="utf-8"))
        if previous_run.get("signature") == request_signature:
            return {
                "status": "skipped",
                "reason": "matching evidence, prompt, adapter and model already completed",
                "output_dir": str(knowledge_dir),
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
    attempt_path = knowledge_dir / "claims-attempt.json"
    try:
        raw_ledger = json.loads(response.text)
    except json.JSONDecodeError as error:
        write_json_atomic(
            attempt_path,
            {
                "schema_version": 1,
                "status": "rejected",
                "provider": response.provider,
                "model": response.model,
                "cost_usd": response.cost_usd,
                "tokens": response.usage,
                "signature": request_signature,
                "error": f"invalid JSON: {error}",
                "output_text": response.text,
            },
        )
        raise InvalidBundleError(f"Model returned invalid JSON: {error}") from error
    segments = load_segments(bundle_dir)
    valid_ids = {str(segment["id"]) for segment in segments}
    try:
        ledger = _validate_ledger(raw_ledger, valid_ids)
    except InvalidBundleError as error:
        write_json_atomic(
            attempt_path,
            {
                "schema_version": 1,
                "status": "rejected",
                "provider": response.provider,
                "model": response.model,
                "cost_usd": response.cost_usd,
                "tokens": response.usage,
                "signature": request_signature,
                "error": str(error),
                "output": raw_ledger,
            },
        )
        raise

    write_json_atomic(
        attempt_path,
        {
            "schema_version": 1,
            "status": "accepted",
            "provider": response.provider,
            "model": response.model,
            "cost_usd": response.cost_usd,
            "tokens": response.usage,
            "signature": request_signature,
            "output": ledger,
        },
    )

    write_json_atomic(knowledge_dir / "claims.json", ledger)
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
    write_json_atomic(knowledge_dir / "claims-run.json", run)

    claims = ledger["claims"]
    uncertainties = ledger["uncertainties"]
    exclusions = ledger["exclusions"]
    manual_lines = ["# Grounded knowledge manual", "", "## Source-grounded points", ""]
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
    checklist_lines = ["# Knowledge checklist", ""] + [
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
        write_text_atomic(knowledge_dir / name, content)
    return {
        "status": "completed",
        "claim_count": len(claims),
        "uncertainty_count": len(uncertainties),
        "cost_usd": response.cost_usd,
        "output_dir": str(knowledge_dir),
    }
