"""Power control and readings for one NVIDIA GPU through NVML."""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from typing import Protocol

import pynvml


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str
    min_limit_w: float
    max_limit_w: float
    default_limit_w: float


class Gpu(Protocol):
    def info(self) -> GpuInfo: ...
    def limit_w(self) -> float: ...
    def set_limit_w(self, watts: float) -> None: ...
    def energy_j(self) -> float: ...
    def power_w(self) -> float: ...
    def sm_clock_mhz(self) -> float: ...


class PowerLimitDenied(RuntimeError):
    pass


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


class NvmlGpu:
    def __init__(self, index: int) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def info(self) -> GpuInfo:
        low_mw, high_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self._handle)
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self._handle)
        return GpuInfo(
            name=_text(pynvml.nvmlDeviceGetName(self._handle)),
            driver=_text(pynvml.nvmlSystemGetDriverVersion()),
            min_limit_w=low_mw / 1000,
            max_limit_w=high_mw / 1000,
            default_limit_w=default_mw / 1000,
        )

    def limit_w(self) -> float:
        return float(pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)) / 1000

    def set_limit_w(self, watts: float) -> None:
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(self._handle, round(watts * 1000))
        except pynvml.NVMLError_NoPermission as err:
            raise PowerLimitDenied(
                "setting a power limit needs root, and some hosts forbid it"
            ) from err

    def energy_j(self) -> float:
        """The card's cumulative energy counter, more exact than integrating sampled power."""
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)) / 1000

    def power_w(self) -> float:
        return float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000

    def sm_clock_mhz(self) -> float:
        return float(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM))


class Sampler:
    """Samples power and SM clock in the background while a load runs."""

    def __init__(self, gpu: Gpu, interval_s: float = 0.1) -> None:
        self._gpu = gpu
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.power_w: list[float] = []
        self.clock_mhz: list[float] = []

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.power_w.append(self._gpu.power_w())
            self.clock_mhz.append(self._gpu.sm_clock_mhz())

    def peak_power_w(self) -> float | None:
        return max(self.power_w, default=None)

    def median_clock_mhz(self) -> float | None:
        return statistics.median(self.clock_mhz) if self.clock_mhz else None
