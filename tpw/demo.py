"""Streams one long answer under a live status bar: tokens per second, watts, power limit."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time

import httpx

from tpw.gpu import NvmlGpu

PROMPT = (
    "Write a long, detailed story about a lighthouse keeper who discovers that the lighthouse "
    "is also a radio telescope. Keep going for as long as you can."
)
WINDOW_S = 2.0
REFRESH_S = 0.25


class StatusBar:
    """Pins one line to the bottom of the terminal while text scrolls above it."""

    def __init__(self) -> None:
        self.rows = shutil.get_terminal_size().lines
        sys.stdout.write(f"\033[2J\033[1;{self.rows - 1}r\033[H")

    def show(self, text: str) -> None:
        sys.stdout.write(f"\0337\033[{self.rows};1H\033[2K\033[7m {text} \033[0m\0338")
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write(f"\033[r\033[{self.rows};1H\n")


class Rate:
    """Events per second over a sliding window."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.times: collections.deque[float] = collections.deque()

    def tick(self, now: float) -> None:
        self.times.append(now)
        while now - self.times[0] > self.window_s:
            self.times.popleft()

    def per_s(self) -> float:
        span = self.times[-1] - self.times[0] if self.times else 0.0
        return (len(self.times) - 1) / span if span else 0.0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default="", help="shown in the status bar, e.g. 'capped'")
    p.add_argument("--max-tokens", type=int, default=2000)
    args = p.parse_args(argv)

    gpu = NvmlGpu(args.gpu)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    rate, bar, shown = Rate(WINDOW_S), StatusBar(), 0.0

    try:
        with httpx.stream("POST", f"{args.url}/v1/chat/completions", json=body, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line.removeprefix("data: "))["choices"]
                sys.stdout.write((choices[0]["delta"].get("content") if choices else None) or "")

                now = time.perf_counter()
                rate.tick(now)
                if now - shown > REFRESH_S:
                    speed = f"{rate.per_s():5.1f} tok/s"
                    power = f"{gpu.power_w():4.0f} W   limit {gpu.limit_w():.0f} W"
                    bar.show(f"{args.label}  {speed}   {power}")
                    shown = now
    finally:
        bar.close()


if __name__ == "__main__":
    main()
