import json
import subprocess
from pathlib import Path

import pytest

from audio2md.llm import LlmRequest
from audio2md.opencode_adapter import OpenCodeAdapter, OpenCodeLlmError


def test_opencode_adapter_uses_pure_mode_and_parses_events() -> None:
    captured: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        events = [
            {"type": "text", "part": {"text": '{"claims": []}'}},
            {
                "type": "step_finish",
                "part": {
                    "tokens": {"input": 10, "output": 4, "reasoning": 2, "total": 16},
                    "cost": 0.001,
                },
            },
        ]
        return subprocess.CompletedProcess(command, 0, "\n".join(map(json.dumps, events)), "")

    response = OpenCodeAdapter(runner=runner).complete(
        LlmRequest(
            "extract",
            "Use evidence only.",
            "packet",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == ["opencode", "run", "--pure"]
    assert "--auto" not in command
    assert "opencode-go/deepseek-v4.1-flash" in command
    assert "JSON Schema" in command[-1]
    assert response.text == '{"claims": []}'
    assert response.provider == "opencode-go"
    assert response.usage["total"] == 16
    assert response.cost_usd == 0.001


def test_opencode_adapter_aggregates_multi_step_usage_and_cost() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        events = [
            {
                "type": "step_finish",
                "part": {
                    "tokens": {"input": 10, "output": 4, "reasoning": 2, "total": 16},
                    "cost": 0.001,
                },
            },
            {"type": "text", "part": {"text": '{"manual":"m"}'}},
            {
                "type": "step_finish",
                "part": {
                    "tokens": {"input": 8, "output": 3, "reasoning": 1, "total": 12},
                    "cost": 0.002,
                },
            },
        ]
        return subprocess.CompletedProcess(command, 0, "\n".join(map(json.dumps, events)), "")

    response = OpenCodeAdapter(runner=runner).complete(LlmRequest("compose", "system", "ledger"))
    assert response.usage == {"input": 18, "output": 7, "reasoning": 3, "total": 28}
    assert response.cost_usd == pytest.approx(0.003)


def test_opencode_adapter_timeout_is_concise_and_recoverable() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 300)

    with pytest.raises(OpenCodeLlmError, match="local session may be recoverable") as error:
        OpenCodeAdapter(runner=runner).complete(LlmRequest("compose", "system", "private ledger"))
    assert "private ledger" not in str(error.value)


def test_opencode_adapter_reports_cli_failure() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7, "", "subscription unavailable")

    with pytest.raises(OpenCodeLlmError, match="subscription unavailable"):
        OpenCodeAdapter(runner=runner).complete(LlmRequest("extract", "system", "packet"))
