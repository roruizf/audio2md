from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .exceptions import Audio2MdError
from .llm import LlmRequest, LlmResponse


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"
JsonTransport = Callable[[urllib.request.Request, float], dict[str, Any]]


class HostedLlmError(Audio2MdError):
    """Raised when a hosted model request fails or returns an invalid response."""


def _urlopen_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise HostedLlmError(f"OpenRouter returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise HostedLlmError(f"Cannot reach OpenRouter: {error.reason}") from error
    if not isinstance(payload, dict):
        raise HostedLlmError("OpenRouter returned a non-object response")
    return payload


class OpenRouterAdapter:
    """Minimal OpenRouter adapter with no transcript persistence or implicit retries."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        endpoint: str = OPENROUTER_ENDPOINT,
        timeout: float = 120.0,
        transport: JsonTransport = _urlopen_json,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self._transport = transport

    def complete(self, request: LlmRequest) -> LlmResponse:
        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise HostedLlmError("OPENROUTER_API_KEY is not set")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.input_text},
            ],
            "temperature": 0,
            "stream": False,
        }
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.task,
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
            body["provider"] = {
                "require_parameters": True,
                "allow_fallbacks": False,
                "data_collection": "deny",
                "zdr": True,
            }
        wire_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Cache": "false",
            },
            method="POST",
        )
        payload = self._transport(wire_request, self.timeout)
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise HostedLlmError("OpenRouter response has no assistant content") from error
        if not isinstance(text, str):
            raise HostedLlmError("OpenRouter assistant content is not text")
        raw_usage = payload.get("usage") or {}
        usage = {
            key: int(raw_usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return LlmResponse(
            text=text,
            model=str(payload.get("model") or self.model),
            provider="openrouter",
            usage=usage,
        )
