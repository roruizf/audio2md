import json
import urllib.request

import pytest

from audio2md.llm import LlmRequest
from audio2md.openrouter import HostedLlmError, OpenRouterAdapter


CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "object"}}},
    "required": ["claims"],
    "additionalProperties": False,
}


def test_openrouter_adapter_builds_private_structured_request() -> None:
    captured: dict[str, object] = {}

    def transport(request: urllib.request.Request, timeout: float) -> dict[str, object]:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data or b"{}")
        captured["timeout"] = timeout
        return {
            "model": "deepseek/deepseek-chat-v3.1",
            "choices": [{"message": {"content": '{"claims": []}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }

    adapter = OpenRouterAdapter(api_key="test-key", transport=transport)
    response = adapter.complete(LlmRequest("extract_claims", "system", "packet", CLAIMS_SCHEMA))

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["X-openrouter-cache"] == "false"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}
    assert response.text == '{"claims": []}'
    assert response.usage["total_tokens"] == 14


def test_openrouter_adapter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(HostedLlmError, match="OPENROUTER_API_KEY"):
        OpenRouterAdapter().complete(LlmRequest("task", "system", "input"))
