import asyncio
import logging
import os
import random
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agent import sessions

log = logging.getLogger(__name__)

RETRYABLE = {"rate_limit_error", "server_error", "timeout", "api_error"}


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class Completion(BaseModel):
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class ModelClient:
    # gpt-5 models reject any temperature other than the default, so none is ever sent.
    def __init__(self, primary: str | None = None, fallback: str | None = None) -> None:
        self._client = AsyncOpenAI()
        self.primary = primary or os.getenv("OPENAI_MODEL_PRIMARY", "gpt-5-mini")
        self.fallback = fallback or os.getenv("OPENAI_MODEL_FALLBACK", "gpt-5-nano")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        session_id: str | None = None,
    ) -> Completion:
        chain = [model] if model else [self.primary, self.fallback]
        last: Exception | None = None
        for candidate in chain:
            try:
                return await self._attempt(
                    candidate, messages, tools, response_format, session_id
                )
            except Exception as exc:
                last = exc
                log.warning("model %s failed, falling back: %s", candidate, exc)
        raise RuntimeError(f"every model in {chain} failed; last error: {last}")

    async def _attempt(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        session_id: str | None,
        attempts: int = 3,
    ) -> Completion:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        delay = 1.0
        for attempt in range(attempts):
            started = time.monotonic()
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                sessions.log_call(session_id, f"model:{model}", "error", elapsed)
                log.warning("%s attempt %d failed: %s", model, attempt + 1, exc)
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay *= 2
                continue

            elapsed = int((time.monotonic() - started) * 1000)
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            sessions.log_call(
                session_id, f"model:{model}", "ok", elapsed, tokens_in, tokens_out
            )
            if session_id:
                sessions.add_tokens(session_id, tokens_in + tokens_out)

            choice = response.choices[0].message
            return Completion(
                model=model,
                content=choice.content,
                tool_calls=[
                    ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
                    for c in (choice.tool_calls or [])
                ],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        raise RuntimeError("unreachable")
