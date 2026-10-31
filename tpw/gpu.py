"""Power control and readings for one NVIDIA GPU through NVML.

Setting a power limit needs root, and some clouds block it inside their VMs.
Energy comes from the card's own cumulative counter, not from sampled power.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Protocol


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
        import pynvml

        self._nv = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def info(self) -> GpuInfo:
        nv, h = self._nv, self._handle
        low_mw, high_mw = nv.nvmlDeviceGetPowerManagementLimitConstraints(h)
        return GpuInfo(
            name=_text(nv.nvmlDeviceGetName(h)),
            driver=_text(nv.nvmlSystemGetDriverVersion()),
            min_limit_w=low_mw / 1000,
            max_limit_w=high_mw / 1000,
            default_limit_w=nv.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
        )

    def limit_w(self) -> float:
        return self._nv.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000

    def set_limit_w(self, watts: float) -> None:
        try:
            self._nv.nvmlDeviceSetPowerManagementLimit(self._handle, int(round(watts * 1000)))
        except self._nv.NVMLError_NoPermission as err:
            raise PowerLimitDenied(
                "setting the power limit needs root, and this host may not allow it at all"
            ) from err

    def energy_j(self) -> float:
        return self._nv.nvmlDeviceGetTotalEnergyConsumption(self._handle) / 1000

    def power_w(self) -> float:
        return self._nv.nvmlDeviceGetPowerUsage(self._handle) / 1000

    def sm_clock_mhz(self) -> float:
        return float(self._nv.nvmlDeviceGetClockInfo(self._handle, self._nv.NVML_CLOCK_SM))


class Sampler:
    """Samples power and SM clock in the background while a load runs."""

    def __init__(self, gpu: Gpu, interval_s: float = 0.1) -> None:
        self._gpu = gpu
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.power_w: list[float] = []
        self.clock_mhz: list[float] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            self.power_w.append(self._gpu.power_w())
            self.clock_mhz.append(self._gpu.sm_clock_mhz())
            time.sleep(self._interval)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def median_clock_mhz(self) -> float | None:
        return statistics.median(self.clock_mhz) if self.clock_mhz else None

    def peak_power_w(self) -> float | None:
        return max(self.power_w) if self.power_w else None
