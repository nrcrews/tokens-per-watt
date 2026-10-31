"""Closed-loop load against an OpenAI-compatible completions endpoint such as vLLM or SGLang."""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

Kind = Literal["decode", "prefill"]
KINDS: tuple[Kind, ...] = ("decode", "prefill")

# Random token IDs give prompts an exact length and no shared prefix to cache.
TOKEN_IDS = range(1000, 30000)
DECODE_PROMPT_TOKENS = 32


@dataclass(frozen=True)
class Workload:
    kind: Kind
    concurrency: int
    duration_s: float
    decode_tokens: int = 512
    prefill_tokens: int = 4096

    @property
    def prompt_tokens(self) -> int:
        return DECODE_PROMPT_TOKENS if self.kind == "decode" else self.prefill_tokens

    @property
    def output_tokens(self) -> int:
        return self.decode_tokens if self.kind == "decode" else 1


@dataclass
class LoadResult:
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    ttft_s: list[float] = field(default_factory=list)
    chunk_gaps_s: list[float] = field(default_factory=list)

    def work_tokens(self, kind: Kind) -> int:
        """The tokens the workload exists to process: output for decode, prompt for prefill."""
        return self.output_tokens if kind == "decode" else self.prompt_tokens


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, `q` in [0, 1]."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


async def _request(
    client: httpx.AsyncClient, url: str, body: dict[str, Any], out: LoadResult
) -> None:
    start = time.perf_counter()
    last: float | None = None
    usage = None
    async with client.stream("POST", f"{url}/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            usage = chunk.get("usage") or usage
            if chunk.get("choices"):
                now = time.perf_counter()
                if last is None:
                    out.ttft_s.append(now - start)
                else:
                    out.chunk_gaps_s.append(now - last)
                last = now

    if usage is None:
        raise RuntimeError("no usage in the stream; the server must honor stream_options")
    out.requests += 1
    out.prompt_tokens += usage["prompt_tokens"]
    out.output_tokens += usage["completion_tokens"]


async def run(url: str, model: str, workload: Workload, seed: int = 0) -> LoadResult:
    """Keeps `concurrency` requests in flight for `duration_s`, then lets the last ones finish."""
    rng = random.Random(seed)
    out = LoadResult()
    limits = httpx.Limits(max_connections=workload.concurrency + 4)

    def body() -> dict[str, Any]:
        return {
            "model": model,
            "prompt": rng.choices(TOKEN_IDS, k=workload.prompt_tokens),
            "max_tokens": workload.output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    async with httpx.AsyncClient(timeout=900, limits=limits) as client:
        start = time.perf_counter()
        deadline = start + workload.duration_s

        async def worker() -> None:
            while time.perf_counter() < deadline:
                await _request(client, url, body(), out)

        await asyncio.gather(*(worker() for _ in range(workload.concurrency)))
        out.elapsed_s = time.perf_counter() - start
    return out
