"""Closed-loop load against an OpenAI-compatible completions endpoint (vLLM, SGLang).

Prompts are random token IDs: their length is exact and no two share a prefix, so the
prefix cache cannot skip prefill. `ignore_eos` holds every decode to its full length.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

Kind = Literal["decode", "prefill"]

# Inside every common tokenizer's vocabulary and clear of special tokens.
TOKEN_ID_RANGE = (1000, 30000)


@dataclass(frozen=True)
class Workload:
    kind: Kind
    concurrency: int
    duration_s: float
    decode_tokens: int = 512
    decode_prompt_tokens: int = 32
    prefill_tokens: int = 4096

    def request_shape(self) -> tuple[int, int]:
        """(prompt tokens, output tokens) of each request."""
        if self.kind == "decode":
            return self.decode_prompt_tokens, self.decode_tokens
        return self.prefill_tokens, 1


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
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(q * 100) - 1]


async def _request(
    client: httpx.AsyncClient, url: str, model: str, prompt: list[int], max_tokens: int, out: LoadResult
) -> None:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    last: float | None = None
    usage = None
    async with client.stream("POST", f"{url}/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("choices"):
                now = time.perf_counter()
                if last is None:
                    out.ttft_s.append(now - start)
                else:
                    out.chunk_gaps_s.append(now - last)
                last = now
    if usage is None:
        raise RuntimeError("the server sent no usage; it must support stream_options.include_usage")
    out.requests += 1
    out.prompt_tokens += usage["prompt_tokens"]
    out.output_tokens += usage["completion_tokens"]


async def run(url: str, model: str, workload: Workload, seed: int = 0) -> LoadResult:
    """Keeps `concurrency` requests in flight for `duration_s`, then lets the last ones finish."""
    rng = random.Random(seed)
    prompt_len, max_tokens = workload.request_shape()
    out = LoadResult()
    limits = httpx.Limits(max_connections=workload.concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900), limits=limits) as client:
        start = time.perf_counter()
        deadline = start + workload.duration_s

        async def worker() -> None:
            while time.perf_counter() < deadline:
                prompt = [rng.randrange(*TOKEN_ID_RANGE) for _ in range(prompt_len)]
                await _request(client, url, model, prompt, max_tokens, out)

        await asyncio.gather(*(worker() for _ in range(workload.concurrency)))
        out.elapsed_s = time.perf_counter() - start
    return out
