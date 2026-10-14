from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LlmRequest:
    task: str
    system_prompt: str
    input_text: str
    response_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class LlmResponse:
    text: str
    model: str
    provider: str
    usage: dict[str, int]


class LlmAdapter(Protocol):
    """Provider-neutral boundary; implementations own credentials and transport."""

    def complete(self, request: LlmRequest) -> LlmResponse: ...
