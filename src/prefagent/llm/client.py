"""One client for every LLM in agents.yaml.

All providers speak the OpenAI Chat Completions protocol, so we use the `openai` SDK
with a per-provider base_url. On top of it this module adds what free tiers need:
per-provider rate limiting, retries with backoff, a persistent cache, and parallelism.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from prefagent.llm.cache import ResponseCache, request_key

log = logging.getLogger(__name__)
T = TypeVar("T")

# Transient errors worth retrying. A 400 (bad request) is NOT retried: repeating an
# invalid request just burns quota.
RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


@dataclass
class ChatRequest:
    model: str                      # a key of `llms` in agents.yaml, e.g. "gptoss120b"
    messages: list[dict]
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    json_mode: bool = False
    # Distinguishes repeated samples of the same request (temperature > 0). Part of the
    # cache key, so sample #3 of a prompt is reproducible yet different from sample #2.
    sample_idx: int = 0
    # Re-asks after an unparseable reply. Part of the cache key; otherwise a retry would
    # just return the same cached bad answer.
    attempt: int = 0
    tag: str = ""                   # free-form label for logs (e.g. "judge:r1")


@dataclass
class ChatResponse:
    text: str
    model: str
    cached: bool
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)


class RateLimiter:
    """Spaces requests to at most `rpm` per minute (thread-safe). Simple fixed spacing
    rather than a token bucket: bursts are what trigger free-tier 429s."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            time.sleep(start - now)


class LLMClient:
    def __init__(self, cfg: dict):
        load_dotenv()
        self._llms = cfg["llms"]
        self._providers = cfg["providers"]
        retry_cfg = cfg["retry"]
        self._retry_kwargs = dict(
            stop=stop_after_attempt(retry_cfg["max_attempts"]),
            wait=wait_random_exponential(multiplier=retry_cfg["initial_wait_s"],
                                         max=retry_cfg["max_wait_s"]),
            retry=retry_if_exception_type(RETRYABLE),
            reraise=True,
        )
        self._clients: dict[str, openai.OpenAI] = {}
        self._limiters = {name: RateLimiter(p["rpm"]) for name, p in self._providers.items()}
        self._no_json_mode: set[str] = set()   # providers that rejected response_format
        cache_cfg = cfg["cache"]
        self.cache = (ResponseCache(Path(cfg["paths"]["cache_dir"]) / "llm_cache.sqlite")
                      if cache_cfg["enabled"] else None)

    # -- public API ---------------------------------------------------------

    def chat(self, req: ChatRequest) -> ChatResponse:
        spec = self._llms[req.model]
        payload = self._payload(req, spec)
        # Key on the *logical* request, not the final payload: payload details (e.g. JSON
        # mode dropped after a provider rejects it) must not change cache identity.
        key = request_key({
            "provider": spec["provider"], "model": spec["id"], "extra": spec.get("extra"),
            "messages": req.messages, "temperature": req.temperature, "top_p": req.top_p,
            "max_tokens": req.max_tokens, "json_mode": req.json_mode,
            "sample_idx": req.sample_idx, "attempt": req.attempt,
        })

        if self.cache and (hit := self.cache.get(key)) is not None:
            return ChatResponse(text=hit["text"], model=req.model, cached=True,
                                finish_reason=hit.get("finish_reason"),
                                usage=hit.get("usage") or {})

        raw = self._call_with_retry(spec["provider"], payload)
        choice = raw.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason == "length":
            # Truncated output: for a candidate answer this means an incomplete response;
            # for a judge it usually means unparseable JSON. Log it so it can be audited.
            log.warning("%s [%s] hit max_tokens=%d", req.model, req.tag, req.max_tokens)
        usage = raw.usage.model_dump() if raw.usage else {}
        result = {"text": text, "finish_reason": choice.finish_reason, "usage": usage}
        if self.cache:
            self.cache.put(key, req.model, payload, result)
        return ChatResponse(text=text, model=req.model, cached=False,
                            finish_reason=choice.finish_reason, usage=usage)

    def chat_many(self, reqs: Sequence[ChatRequest], max_workers: int = 8,
                  desc: str = "llm") -> list[ChatResponse]:
        """Run requests in parallel, preserving order. Rate limiters still apply per
        provider, so parallelism mostly helps when requests span several providers."""
        return parallel_map(self.chat, reqs, max_workers=max_workers, desc=desc)

    # -- internals ----------------------------------------------------------

    def _payload(self, req: ChatRequest, spec: dict) -> dict:
        payload = {
            "model": spec["id"],
            "messages": req.messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
        }
        if req.json_mode and spec["provider"] not in self._no_json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Provider-specific knobs from agents.yaml (e.g. reasoning_effort for gpt-oss).
        if extra := spec.get("extra"):
            payload["extra_body"] = dict(extra)
        return payload

    def _client(self, provider: str) -> openai.OpenAI:
        if provider not in self._clients:
            p = self._providers[provider]
            key_env = p.get("api_key_env")
            api_key = os.environ.get(key_env) if key_env else "ollama"
            if not api_key:
                raise RuntimeError(f"Set {key_env} in .env to use provider {provider!r}")
            # max_retries=0: tenacity owns retrying, so attempts are not multiplied.
            self._clients[provider] = openai.OpenAI(
                base_url=p["base_url"], api_key=api_key, max_retries=0, timeout=120)
        return self._clients[provider]

    def _call_with_retry(self, provider: str, payload: dict):
        @retry(**self._retry_kwargs)
        def _call():
            self._limiters[provider].wait()
            try:
                return self._client(provider).chat.completions.create(**payload)
            except openai.BadRequestError as e:
                # Some OpenAI-compatible endpoints don't support JSON mode. Drop it once
                # for that provider and rely on the tolerant parser instead.
                if "response_format" in payload and "response_format" in str(e):
                    log.warning("%s rejected response_format; disabling JSON mode", provider)
                    self._no_json_mode.add(provider)
                    payload.pop("response_format")
                    return self._client(provider).chat.completions.create(**payload)
                raise
        return _call()


def parallel_map(fn: Callable[..., T], items: Sequence, max_workers: int = 8,
                 desc: str = "") -> list[T]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc, leave=False))
