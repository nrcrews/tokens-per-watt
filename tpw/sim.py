"""A made-up GPU and engine for dry runs and tests. Nothing it produces is a measurement.

The model is a cartoon of the real effect: decode steps are mostly a fixed memory-bound
cost, prefill is compute that slows with the clock, and a power cap lowers the clock.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tpw.gpu import GpuInfo

IDLE_W = 90.0
MAX_CLOCK_MHZ = 1980.0
DECODE_MEMORY_S = 0.008  # per step, whatever the batch
DECODE_COMPUTE_S = 0.00012  # per sequence per step at full clock
PREFILL_COMPUTE_S = 0.00004  # per prompt token at full clock


class SimGpu:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limit = 700.0
        self._active = {"decode": 0, "prefill": 0}
        self._energy = 0.0
        self._last = time.perf_counter()

    def info(self) -> GpuInfo:
        return GpuInfo("Simulated GPU", "simulated", 200.0, 700.0, 700.0)

    def limit_w(self) -> float:
        return self._limit

    def set_limit_w(self, watts: float) -> None:
        with self._lock:
            self._advance()
            self._limit = watts

    def energy_j(self) -> float:
        with self._lock:
            self._advance()
            return self._energy

    def power_w(self) -> float:
        demand = self._demand_w()
        return IDLE_W if demand is None else min(self._limit, demand)

    def sm_clock_mhz(self) -> float:
        return MAX_CLOCK_MHZ * self.clock_fraction() if self._demand_w() else 345.0

    def clock_fraction(self) -> float:
        demand = self._demand_w()
        if demand is None or self._limit >= demand:
            return 1.0
        # Dynamic power grows roughly with the cube of the clock.
        return ((self._limit - IDLE_W) / (demand - IDLE_W)) ** (1 / 3)

    def begin(self, kind: str) -> None:
        with self._lock:
            self._advance()
            self._active[kind] += 1

    def end(self, kind: str) -> None:
        with self._lock:
            self._advance()
            self._active[kind] -= 1

    def decode_step_s(self) -> float:
        return DECODE_MEMORY_S + self._active["decode"] * DECODE_COMPUTE_S / self.clock_fraction()

    def prefill_s(self, tokens: int) -> float:
        return tokens * max(1, self._active["prefill"]) * PREFILL_COMPUTE_S / self.clock_fraction()

    def _demand_w(self) -> float | None:
        if self._active["prefill"]:
            return 690.0
        if self._active["decode"]:
            return min(640.0, 300.0 + 4.0 * self._active["decode"])
        return None

    def _advance(self) -> None:
        now = time.perf_counter()
        self._energy += self.power_w() * (now - self._last)
        self._last = now


def _handler(gpu: SimGpu) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            pass

        def do_GET(self) -> None:
            body = json.dumps({"version": "simulated"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt, max_tokens = request["prompt"], request["max_tokens"]
            prompt_len = len(prompt) if isinstance(prompt, list) else len(prompt.split())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            kind = "prefill" if max_tokens == 1 else "decode"
            gpu.begin(kind)
            try:
                if kind == "prefill":
                    time.sleep(gpu.prefill_s(prompt_len))
                for _ in range(max_tokens):
                    if kind == "decode":
                        time.sleep(gpu.decode_step_s())
                    self._send({"choices": [{"index": 0, "text": " tok"}]})
            finally:
                gpu.end(kind)
            usage = {"prompt_tokens": prompt_len, "completion_tokens": max_tokens}
            self._send({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _send(self, chunk: dict[str, object]) -> None:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256  # the default of 5 resets connections under real concurrency


def serve(gpu: SimGpu) -> tuple[str, ThreadingHTTPServer]:
    """Starts the simulated engine on a free local port; returns its URL and server."""
    server = _Server(("127.0.0.1", 0), _handler(gpu))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server
