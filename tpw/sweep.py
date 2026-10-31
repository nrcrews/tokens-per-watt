"""Sweeps a GPU's power limit and measures decode and prefill at each step.

Writes a `meta` line, then one `run` line per (limit, workload, concurrency, repeat).
The card's original limit is restored on exit, Ctrl-C and SIGTERM included.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import json
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, TypeVar

import httpx

from tpw import load
from tpw.gpu import Gpu, NvmlGpu, Sampler

T = TypeVar("T")


@dataclass(frozen=True)
class Target:
    gpu: Gpu
    url: str
    model: str
    engine: str
    simulated: bool


def power_limits(min_w: float, max_w: float, steps: int) -> list[float]:
    """Evenly spaced from max to min, the steps between rounded to 5 W."""
    if steps < 2:
        raise ValueError("need at least two power limits")
    step = (max_w - min_w) / (steps - 1)
    between = [round((max_w - step * i) / 5) * 5 for i in range(1, steps - 1)]
    return [max_w, *between, min_w]


def csv(parse: Callable[[str], T]) -> Callable[[str], list[T]]:
    return lambda text: [parse(item) for item in text.split(",")]


def parse_kind(text: str) -> load.Kind:
    if text not in load.KINDS:
        raise argparse.ArgumentTypeError(f"kind must be one of {', '.join(load.KINDS)}")
    return text


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000", help="OpenAI-compatible server")
    p.add_argument("--model", help="served model (default: the server's first)")
    p.add_argument("--gpu", type=int, default=0, help="NVML index of the server's GPU")
    p.add_argument("--limits", type=csv(float), help="watts (default: --steps from max to min)")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--kinds", type=csv(parse_kind), default=list(load.KINDS))
    p.add_argument("--concurrency", type=csv(int), default=[1, 32], help="requests in flight")
    p.add_argument("--duration", type=float, default=20.0, help="seconds of load per run")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--prefill-tokens", type=int, default=4096)
    p.add_argument("--settle", type=float, default=5.0, help="seconds after a limit change")
    p.add_argument("--out", type=Path, default=Path("results.jsonl"))
    p.add_argument("--simulate", action="store_true", help="dry run on a made-up GPU and engine")
    return p.parse_args(argv)


@contextmanager
def connect(args: argparse.Namespace) -> Iterator[Target]:
    if not args.simulate:
        with httpx.Client(base_url=args.url, timeout=10) as client:
            engine = client.get("/version").json().get("version", "unknown")
            model = args.model or client.get("/v1/models").json()["data"][0]["id"]
        yield Target(NvmlGpu(args.gpu), args.url, model, engine, simulated=False)
        return

    from tpw import sim

    gpu = sim.SimGpu()
    url, server = sim.serve(gpu)
    try:
        yield Target(gpu, url, args.model or "simulated-model", "simulated", simulated=True)
    finally:
        server.shutdown()


def measure(target: Target, workload: load.Workload, seed: int) -> dict[str, Any]:
    with Sampler(target.gpu) as sampler:
        before_j = target.gpu.energy_j()
        result = asyncio.run(load.run(target.url, target.model, workload, seed))
        energy_j = target.gpu.energy_j() - before_j

    tokens = result.work_tokens(workload.kind)
    return {
        "requests": result.requests,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "work_tokens": tokens,
        "elapsed_s": result.elapsed_s,
        "tokens_per_s": tokens / result.elapsed_s,
        "energy_j": energy_j,
        "tokens_per_j": tokens / energy_j if energy_j > 0 else None,
        "mean_power_w": energy_j / result.elapsed_s,
        "peak_power_w": sampler.peak_power_w(),
        "median_sm_clock_mhz": sampler.median_clock_mhz(),
        "ttft_p50_s": load.percentile(result.ttft_s, 0.5),
        "chunk_gap_p50_s": load.percentile(result.chunk_gaps_s, 0.5),
        "chunk_gap_p99_s": load.percentile(result.chunk_gaps_s, 0.99),
    }


def sweep(target: Target, args: argparse.Namespace, out: TextIO) -> None:
    info = target.gpu.info()
    limits = args.limits or power_limits(info.min_limit_w, info.max_limit_w, args.steps)
    seeds = itertools.count(1)

    for limit in limits:
        target.gpu.set_limit_w(limit)
        time.sleep(args.settle)
        applied = target.gpu.limit_w()
        log(f"limit {limit:.0f} W (card reports {applied:.0f} W)")

        for kind, concurrency in itertools.product(args.kinds, args.concurrency):
            workload = load.Workload(
                kind, concurrency, args.duration, args.decode_tokens, args.prefill_tokens
            )
            warmup = dataclasses.replace(workload, duration_s=min(3.0, args.duration))
            asyncio.run(load.run(target.url, target.model, warmup, seed=0))

            for repeat in range(args.repeats):
                row = measure(target, workload, next(seeds))
                record = {
                    "type": "run",
                    "limit_w": limit,
                    "applied_limit_w": applied,
                    "kind": kind,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **row,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                log(
                    f"  {kind:7} c={concurrency:<3} #{repeat}  "
                    f"{row['tokens_per_s']:9.1f} tok/s  {row['mean_power_w']:6.1f} W"
                )


def meta(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    info = target.gpu.info()
    return {
        "type": "meta",
        "simulated": target.simulated,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": info.name,
        "driver": info.driver,
        "min_limit_w": info.min_limit_w,
        "max_limit_w": info.max_limit_w,
        "default_limit_w": info.default_limit_w,
        "model": target.model,
        "engine_version": target.engine,
        "decode_prompt_tokens": load.DECODE_PROMPT_TOKENS,
        "decode_tokens": args.decode_tokens,
        "prefill_tokens": args.prefill_tokens,
        "duration_s": args.duration,
    }


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    with connect(args) as target, args.out.open("w") as out:
        original_w = target.gpu.limit_w()
        out.write(json.dumps(meta(target, args)) + "\n")
        try:
            sweep(target, args, out)
        finally:
            target.gpu.set_limit_w(original_w)
            log(f"restored the power limit to {original_w:.0f} W")


if __name__ == "__main__":
    main()
