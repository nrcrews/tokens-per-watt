"""Turns sweep results into charts whose titles are computed from the data.

Given several results files (one per GPU), each chip is normalised to its own uncapped run,
lines show the median across chips and a shaded band shows their range.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Metric = Literal["tokens_per_s", "tokens_per_j", "sm_clock_mhz"]
Key = tuple[str, int]

SIZE_IN = (8.0, 4.5)  # 1600 x 900 at 200 dpi
DPI = 200
KEEP = 95.0  # % of uncapped decode speed that defines the sweet spot
MIN_LABEL_GAP = 6.0  # percentage points between end labels
FULL_CLOCK = 97.0  # % of uncapped clock that still counts as full

FONTS = ["Inter", "Arial", "DejaVu Sans"]
LINE: dict[str, Any] = {"linewidth": 1.75, "marker": "o", "markersize": 6.5, "zorder": 3}
END_LABEL: dict[str, Any] = {"xytext": (-10, 0), "textcoords": "offset points", "va": "center"}
LEGEND: dict[str, Any] = {"loc": "upper left", "bbox_to_anchor": (0.03, 0.835), "frameon": False}
WATERMARK: dict[str, Any] = {"fontsize": 32, "fontweight": "bold", "alpha": 0.22, "rotation": 18}


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    wash: str
    series: tuple[str, str, str]


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        wash="#f0efec",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        wash="#262624",
        series=("#3987e5", "#d95926", "#199e70"),
    ),
}


@dataclass(frozen=True)
class Point:
    limit_w: float
    tokens_per_s: float
    tokens_per_j: float
    mean_power_w: float
    sm_clock_mhz: float = math.nan


@dataclass(frozen=True)
class Band:
    """A metric at one limit as a % of uncapped: the median across chips and their range."""

    limit_w: float
    mid: float
    low: float
    high: float


Series = dict[Key, list[Point]]


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta = next(row for row in rows if row["type"] == "meta")
    return meta, [row for row in rows if row["type"] == "run"]


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(statistics.median(values)) if values else math.nan


def summarize(runs: list[dict[str, Any]]) -> Series:
    """The median of the repeats at each (kind, concurrency, limit), sorted by limit."""
    groups: defaultdict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["kind"], run["concurrency"], run["limit_w"])].append(run)

    series: defaultdict[Key, list[Point]] = defaultdict(list)
    for (kind, concurrency, limit), rows in sorted(groups.items()):
        keys = ("tokens_per_s", "tokens_per_j", "mean_power_w", "median_sm_clock_mhz")
        metrics = (_median(rows, key) for key in keys)
        series[(kind, concurrency)].append(Point(limit, *metrics))
    return dict(series)


def shares(points: list[Point], metric: Metric) -> list[float]:
    """Each point's metric as a % of the same metric at the highest limit."""
    base = getattr(points[-1], metric)
    return [100 * getattr(p, metric) / base for p in points]


def bands(chips: list[Series], key: Key, metric: Metric, relative: bool = True) -> list[Band]:
    """Median and range across chips at each limit, as % of uncapped unless `relative` is off."""
    limits = [p.limit_w for p in chips[0][key]]
    if any([p.limit_w for p in chip[key]] != limits for chip in chips):
        raise ValueError("every chip must be swept over the same power limits")
    per_chip = [
        shares(chip[key], metric) if relative else [getattr(p, metric) for p in chip[key]]
        for chip in chips
    ]
    return [
        Band(limit, statistics.median(values), min(values), max(values))
        for limit, values in zip(limits, zip(*per_chip, strict=True), strict=True)
    ]


def at(curve: list[Band], limit_w: float) -> float:
    return next(b.mid for b in curve if b.limit_w == limit_w)


def sweet_spot(speed: list[Band], keep: float = KEEP) -> float:
    """The lowest limit at which it and every higher limit keep `keep`% of full speed."""
    i = len(speed) - 1
    while i > 0 and speed[i - 1].mid >= keep:
        i -= 1
    return speed[i].limit_w


def holds_until(curve: list[Band], keep: float) -> float:
    """The lowest limit down to which a curve stays within `keep`% of its uncapped value."""
    top = curve[-1].mid
    return sweet_spot([Band(b.limit_w, 100 * b.mid / top, 0, 0) for b in curve], keep)


def slows_prefill_not_decode(decode: list[Band], prefill: list[Band], limit_w: float) -> bool:
    """True when the cap costs prefill at least twice what it costs decode."""
    return 100 - at(prefill, limit_w) >= 2 * max(100 - at(decode, limit_w), 1.0)


def speed_title(decode: list[Band], prefill: list[Band], spot: float) -> str:
    if slows_prefill_not_decode(decode, prefill, spot):
        return "A power cap slows prefill, not decode"
    return (
        f"At {spot:.0f} W, decode keeps {at(decode, spot):.0f}%, prefill {at(prefill, spot):.0f}%"
    )


def efficiency_title(decode: list[Band]) -> str:
    peak = max(decode, key=lambda b: b.mid)
    if peak.mid <= 100:
        return "A power cap doesn't make decode cheaper here"
    if peak is decode[0]:
        return "Every cut makes each decode token cheaper"
    return f"Lower clocks make each decode token cheaper, down to {peak.limit_w:.0f} W"


def concurrency_title(small: list[Band], large: list[Band]) -> tuple[str, float]:
    """The title and the limit where the larger batch falls furthest behind."""
    gap, limit = max((s.mid - b.mid, s.limit_w) for s, b in zip(small, large, strict=True))
    if gap >= 2:
        return "Bigger batches do more math, so they slow sooner", limit
    return "Batch size barely changes what the cap costs decode", limit


class Canvas:
    def __init__(self, meta: dict[str, Any], chips: int, theme: Theme) -> None:
        self.meta = meta
        self.chips = chips
        self.t = theme
        plt.rcParams["font.sans-serif"] = FONTS

    def blank(self) -> Figure:
        fig = plt.figure(figsize=SIZE_IN, dpi=DPI, facecolor=self.t.surface)
        self.source(fig)
        return fig

    def chart(self, title: str, subtitle: str) -> tuple[Figure, Axes]:
        fig = self.blank()
        fig.text(0.04, 0.92, title, fontsize=15, fontweight="bold", color=self.t.ink)
        fig.text(0.04, 0.86, subtitle, fontsize=10, color=self.t.ink2)

        ax = fig.add_axes((0.09, 0.15, 0.86, 0.6), facecolor=self.t.surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(self.t.axis)
        ax.tick_params(colors=self.t.muted, length=0, labelsize=9)
        ax.grid(axis="y", color=self.t.grid, linewidth=0.75)
        ax.set_axisbelow(True)
        return fig, ax

    def source(self, fig: Figure) -> None:
        """The footer naming what was measured, and a watermark on simulated data."""
        m = self.meta
        gpu = (
            f"{m['gpu']}, median of {self.chips} chips (band: range)"
            if self.chips > 1
            else m["gpu"]
        )
        footer = f"{gpu} · {m['model']} · vLLM {m['engine_version']} · GPU power from NVML"
        fig.text(0.04, 0.035, footer, fontsize=7.5, color=self.t.muted)
        if m["simulated"]:
            mark = "SIMULATED — NOT MEASURED"
            fig.text(0.5, 0.48, mark, color=self.t.muted, ha="center", va="center", **WATERMARK)

    def line(self, ax: Axes, curve: list[Band], slot: int, label: str) -> None:
        xs, color = [b.limit_w for b in curve], self.t.series[slot]
        ax.fill_between(xs, [b.low for b in curve], [b.high for b in curve], color=color, alpha=0.1)
        # A surface-coloured marker edge keeps dots legible where lines cross.
        ax.plot(xs, [b.mid for b in curve], label=label, color=color, mec=self.t.surface, **LINE)

    def end_label(self, ax: Axes, curve: list[Band], text: str) -> None:
        """Labels a line at its lowest limit, where the lines spread apart."""
        xy = (curve[0].limit_w, curve[0].mid)
        ax.annotate(text, xy, ha="right", fontsize=9, color=self.t.ink, **END_LABEL)

    def legend(self, fig: Figure, ax: Axes) -> None:
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=len(labels), fontsize=9, **LEGEND)
        for text in legend.get_texts():
            text.set_color(self.t.ink2)

    def save(self, fig: Figure, path: Path) -> None:
        fig.savefig(path, facecolor=self.t.surface)
        plt.close(fig)


def watts_axis(ax: Axes, limits: list[float]) -> None:
    span = limits[-1] - limits[0]
    ax.set_xticks(limits, [f"{w:.0f} W" for w in limits])
    ax.set_xlim(limits[0] - span * 0.18, limits[-1] + span * 0.03)


def percent_axis(ax: Axes, low: float = 0, high: float = 108) -> None:
    ax.set_ylim(low, high)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")


def decode_and_prefill(
    c: Canvas, decode: list[Band], prefill: list[Band], title: str, subtitle: str
) -> tuple[Figure, Axes]:
    fig, ax = c.chart(title, subtitle)
    c.line(ax, decode, 0, "Decode (writing tokens)")
    c.line(ax, prefill, 1, "Prefill (reading the prompt)")
    # Labels that would overlap are dropped; the legend still names both lines.
    if abs(decode[0].mid - prefill[0].mid) >= MIN_LABEL_GAP:
        c.end_label(ax, decode, "Decode")
        c.end_label(ax, prefill, "Prefill")
    c.legend(fig, ax)
    watts_axis(ax, [b.limit_w for b in decode])
    return fig, ax


def speed_chart(c: Canvas, chips: list[Series], concurrency: int, out: Path) -> None:
    decode = bands(chips, ("decode", concurrency), "tokens_per_s")
    prefill = bands(chips, ("prefill", concurrency), "tokens_per_s")
    spot, top = sweet_spot(decode), decode[-1].limit_w
    d, p = at(decode, spot), at(prefill, spot)
    title = speed_title(decode, prefill, spot)
    subtitle = (
        f"Speed vs uncapped {top:.0f} W, {concurrency} in flight. "
        f"At {spot:.0f} W decode keeps {d:.0f}%, prefill {p:.0f}%."
    )

    fig, ax = decode_and_prefill(c, decode, prefill, title, subtitle)
    ax.axvspan(spot, top, color=c.t.wash, zorder=0, linewidth=0)
    band = f"decode keeps ≥{KEEP:.0f}%"
    ax.text((spot + top) / 2, 6, band, ha="center", fontsize=8.5, color=c.t.ink2)
    percent_axis(ax)
    c.save(fig, out / "speed.png")


def efficiency_chart(c: Canvas, chips: list[Series], concurrency: int, out: Path) -> None:
    speed = bands(chips, ("decode", concurrency), "tokens_per_s")
    decode = bands(chips, ("decode", concurrency), "tokens_per_j")
    prefill = bands(chips, ("prefill", concurrency), "tokens_per_j")
    spot, top = sweet_spot(speed), decode[-1].limit_w
    title = efficiency_title(decode)
    subtitle = (
        f"Tokens per joule vs uncapped {top:.0f} W, {concurrency} in flight. "
        f"At {spot:.0f} W decode {at(decode, spot) - 100:+.0f}%, "
        f"prefill {at(prefill, spot) - 100:+.0f}%."
    )

    fig, ax = decode_and_prefill(c, decode, prefill, title, subtitle)
    ax.axhline(100, color=c.t.axis, linewidth=0.75, zorder=1)
    values = [v for b in decode + prefill for v in (b.low, b.high)]
    percent_axis(ax, min(80.0, min(values) - 5), max(values) + 10)
    c.save(fig, out / "efficiency.png")


def concurrency_chart(c: Canvas, chips: list[Series], out: Path) -> None:
    keys = sorted(key for key in chips[0] if key[0] == "decode")[:3]
    if len(keys) < 2:
        return
    curves = {n: bands(chips, (kind, n), "tokens_per_s") for kind, n in keys}
    title, limit = concurrency_title(curves[min(curves)], curves[max(curves)])
    kept = ", ".join(f"{at(curve, limit):.0f}% with {n}" for n, curve in curves.items())
    subtitle = f"Decode speed vs uncapped. At {limit:.0f} W: {kept} in flight."

    fig, ax = c.chart(title, subtitle)
    for slot, (n, curve) in enumerate(curves.items()):
        c.line(ax, curve, slot, f"{n} in flight")
    c.legend(fig, ax)
    watts_axis(ax, [b.limit_w for b in curves[max(curves)]])
    percent_axis(ax)
    c.save(fig, out / "concurrency.png")


def clock_chart(c: Canvas, chips: list[Series], concurrency: int, out: Path) -> None:
    decode_key, prefill_key = ("decode", concurrency), ("prefill", concurrency)
    if any(
        math.isnan(p.sm_clock_mhz) for chip in chips for p in chip[decode_key] + chip[prefill_key]
    ):
        return
    decode = bands(chips, decode_key, "sm_clock_mhz", relative=False)
    prefill = bands(chips, prefill_key, "sm_clock_mhz", relative=False)
    decode_speed = bands(chips, decode_key, "tokens_per_s")
    prefill_speed = bands(chips, prefill_key, "tokens_per_s")
    if slows_prefill_not_decode(decode_speed, prefill_speed, sweet_spot(decode_speed)):
        title = "The cap lowers the clock. Only prefill needs it."
    else:
        title = "Core clock under each power limit"
    full = holds_until(decode, FULL_CLOCK)
    subtitle = (
        f"Median core clock in MHz, {concurrency} in flight. "
        f"Decode holds full clock down to {full:.0f} W."
    )

    fig, ax = c.chart(title, subtitle)
    c.line(ax, decode, 0, "Decode (writing tokens)")
    c.line(ax, prefill, 1, "Prefill (reading the prompt)")
    c.legend(fig, ax)
    watts_axis(ax, [b.limit_w for b in decode])
    ax.set_ylim(0, max(b.high for b in decode + prefill) * 1.12)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    c.save(fig, out / "clock.png")


def headline(c: Canvas, chips: list[Series], concurrency: int, out: Path) -> None:
    """The one number: more decode tokens from the same power budget, at the sweet spot."""
    speed = bands(chips, ("decode", concurrency), "tokens_per_s")
    energy = bands(chips, ("decode", concurrency), "tokens_per_j")
    spot, top = sweet_spot(speed), speed[-1].limit_w
    detail = (
        f"Every GPU capped at {spot:.0f} W ({100 * spot / top:.0f}% of max): "
        f"decode {100 - at(speed, spot):.0f}% slower, {concurrency} requests in flight"
    )
    caveat = "GPU power only: CPU, memory and cooling overheads are not counted."

    fig = c.blank()
    big = f"{at(energy, spot) - 100:+.0f}%"
    fig.text(0.07, 0.62, big, fontsize=96, fontweight="bold", va="center", color=c.t.ink)
    fig.text(0.07, 0.36, "more decode tokens from the same megawatt", fontsize=20, color=c.t.ink)
    fig.text(0.07, 0.27, detail, fontsize=11, color=c.t.ink2)
    fig.text(0.07, 0.10, caveat, fontsize=8.5, color=c.t.muted)
    c.save(fig, out / "headline.png")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="*", default=[Path("results.jsonl")])
    p.add_argument("--out", type=Path, default=Path("charts"))
    p.add_argument("--concurrency", type=int, help="batch for the main charts (default: largest)")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    args = p.parse_args(argv)

    loaded = [load(path) for path in args.results]
    meta = loaded[0][0]
    if any((m["gpu"], m["model"]) != (meta["gpu"], meta["model"]) for m, _ in loaded):
        raise SystemExit("every results file must come from the same GPU type and model")
    chips = [summarize(runs) for _, runs in loaded]

    concurrency = args.concurrency or max(c for kind, c in chips[0] if kind == "decode")
    canvas = Canvas(meta, len(chips), THEMES[args.theme])
    args.out.mkdir(parents=True, exist_ok=True)

    speed_chart(canvas, chips, concurrency, args.out)
    clock_chart(canvas, chips, concurrency, args.out)
    efficiency_chart(canvas, chips, concurrency, args.out)
    concurrency_chart(canvas, chips, args.out)
    headline(canvas, chips, concurrency, args.out)
    print(f"wrote charts to {args.out}/")


if __name__ == "__main__":
    main()
