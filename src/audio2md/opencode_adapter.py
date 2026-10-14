from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .exceptions import Audio2MdError
from .llm import LlmRequest, LlmResponse


DEFAULT_MODEL = "opencode-go/deepseek-v4.1-flash"
Runner = Callable[..., subprocess.CompletedProcess[str]]


class OpenCodeLlmError(Audio2MdError):
    """Raised when the OpenCode CLI cannot produce a model response."""


class OpenCodeAdapter:
    """Use an existing OpenCode subscription without reading API keys."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        workdir: Path = Path("/tmp"),
        timeout: float = 180.0,
        runner: Runner = subprocess.run,
    ) -> None:
        self.model = model
        self.workdir = workdir
        self.timeout = timeout
        self._runner = runner

    def complete(self, request: LlmRequest) -> LlmResponse:
        schema_instruction = ""
        if request.response_schema is not None:
            schema_instruction = (
                "\n\nReturn JSON only and conform to this JSON Schema:\n"
                + json.dumps(request.response_schema, ensure_ascii=False, separators=(",", ":"))
            )
        prompt = request.system_prompt + schema_instruction + "\n\nINPUT:\n" + request.input_text
        command = [
            "opencode",
            "run",
            "--pure",
            "--model",
            self.model,
            "--format",
            "json",
            "--title",
            f"audio2md-{request.task}",
            prompt,
        ]
        try:
            completed = self._runner(
                command,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OpenCodeLlmError(f"Cannot execute OpenCode: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise OpenCodeLlmError(
                f"OpenCode exited with {completed.returncode}: {detail or 'no error detail'}"
            )

        text_response: str | None = None
        usage: dict[str, int] = {}
        cost_usd: float | None = None
        for line in completed.stdout.splitlines():
            try:
                event: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get("part") or {}
            if event.get("type") == "text" and isinstance(part.get("text"), str):
                text_response = part["text"]
            if event.get("type") == "step_finish":
                raw_tokens = part.get("tokens") or {}
                usage = {
                    key: int(raw_tokens.get(key, 0))
                    for key in ("input", "output", "reasoning", "total")
                }
                if part.get("cost") is not None:
                    cost_usd = float(part["cost"])
        if text_response is None:
            raise OpenCodeLlmError("OpenCode returned no text response")
        return LlmResponse(
            text=text_response,
            model=self.model.split("/", 1)[-1],
            provider=self.model.split("/", 1)[0],
            usage=usage,
            cost_usd=cost_usd,
        )
