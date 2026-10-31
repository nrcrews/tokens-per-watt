"""Sweeps a GPU's power limit and measures decode and prefill at each step.

Writes one JSON line per run: a `meta` line first, then a `run` line per
(power limit, workload, concurrency, repeat). The card's original limit is put back
on exit, including on Ctrl-C and SIGTERM.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import httpx

from tpw import load
from tpw.gpu import Gpu, NvmlGpu, Sampler


def power_limits(min_w: float, max_w: float, steps: int) -> list[float]:
    """Evenly spaced limits from the card's maximum down to its minimum, rounded to 5 W."""
    if steps < 2:
        raise ValueError("need at least two power limits")
    span = max_w - min_w
    limits = [max_w - span * i / (steps - 1) for i in range(steps)]
    return [max_w] + [round(w / 5) * 5 for w in limits[1:-1]] + [min_w]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000", help="OpenAI-compatible server")
    p.add_argument("--model", help="model name the server serves (default: its first model)")
    p.add_argument("--gpu", type=int, default=0, help="NVML index of the GPU the server runs on")
    p.add_argument("--limits", help="comma-separated watts; default: --steps from max to min")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--kinds", default="decode,prefill")
    p.add_argument("--concurrency", default="1,32", help="comma-separated requests in flight")
    p.add_argument("--duration", type=float, default=20.0, help="seconds of load per run")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--prefill-tokens", type=int, default=4096)
    p.add_argument("--settle", type=float, default=5.0, help="seconds to wait after a limit change")
    p.add_argument("--out", type=Path, default=Path("results.jsonl"))
    p.add_argument("--simulate", action="store_true", help="dry run on a made-up GPU and engine")
    return p.parse_args(argv)


def server_facts(url: str, model: str | None) -> tuple[str, str]:
    """(model, engine version) from the server itself."""
    with httpx.Client(timeout=10) as client:
        version = client.get(f"{url}/version").json().get("version", "unknown")
        if model is None:
            model = client.get(f"{url}/v1/models").json()["data"][0]["id"]
    return model, version


def measure(gpu: Gpu, url: str, model: str, workload: load.Workload, seed: int) -> dict[str, object]:
    with Sampler(gpu) as sampler:
        energy_before = gpu.energy_j()
        result = asyncio.run(load.run(url, model, workload, seed=seed))
        energy_j = gpu.energy_j() - energy_before
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


def sweep(gpu: Gpu, url: str, model: str, args: argparse.Namespace, out: TextIO) -> None:
    info = gpu.info()
    limits = (
        [float(w) for w in args.limits.split(",")]
        if args.limits
        else power_limits(info.min_limit_w, info.max_limit_w, args.steps)
    )
    kinds = args.kinds.split(",")
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    seed = 0
    for limit in limits:
        gpu.set_limit_w(limit)
        time.sleep(args.settle)
        applied = gpu.limit_w()
        print(f"limit {limit:.0f} W (card reports {applied:.0f} W)", file=sys.stderr)
        for kind in kinds:
            for concurrency in concurrencies:
                workload = load.Workload(
                    kind=kind,  # type: ignore[arg-type]
                    concurrency=concurrency,
                    duration_s=args.duration,
                    decode_tokens=args.decode_tokens,
                    prefill_tokens=args.prefill_tokens,
                )
                warmup = dataclasses.replace(workload, duration_s=min(3.0, args.duration))
                asyncio.run(load.run(url, model, warmup, seed=-1))
                for repeat in range(args.repeats):
                    seed += 1
                    row = measure(gpu, url, model, workload, seed)
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
                    print(
                        f"  {kind:7} c={concurrency:<3} #{repeat}: {row['tokens_per_s']:9.1f} tok/s"
                        f"  {row['mean_power_w']:6.1f} W",
                        file=sys.stderr,
                    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.simulate:
        from tpw import sim

        sim_gpu = sim.SimGpu()
        gpu: Gpu = sim_gpu
        args.url, server = sim.serve(sim_gpu)
        model, engine = args.model or "simulated-model", "simulated"
    else:
        gpu = NvmlGpu(args.gpu)
        model, engine = server_facts(args.url, args.model)

    info = gpu.info()
    original_limit = gpu.limit_w()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with args.out.open("w") as out:
        meta = {
            "type": "meta",
            "simulated": bool(args.simulate),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu": info.name,
            "driver": info.driver,
            "min_limit_w": info.min_limit_w,
            "max_limit_w": info.max_limit_w,
            "default_limit_w": info.default_limit_w,
            "model": model,
            "engine_version": engine,
            "decode_tokens": args.decode_tokens,
            "decode_prompt_tokens": load.Workload("decode", 1, 0).decode_prompt_tokens,
            "prefill_tokens": args.prefill_tokens,
            "duration_s": args.duration,
        }
        out.write(json.dumps(meta) + "\n")
        try:
            sweep(gpu, args.url, model, args, out)
        finally:
            gpu.set_limit_w(original_limit)
            print(f"restored the power limit to {original_limit:.0f} W", file=sys.stderr)
    if args.simulate:
        server.shutdown()


if __name__ == "__main__":
    main()
