"""Turns a sweep's results.jsonl into charts whose titles are computed from the data."""

from __future__ import annotations

import argparse
import json
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

Metric = Literal["tokens_per_s", "tokens_per_j"]

SIZE_IN = (8.0, 4.5)  # 1600 x 900 at 200 dpi
DPI = 200
KEEP = 95.0  # % of uncapped decode speed that defines the sweet spot

FONTS = ["Inter", "Arial", "DejaVu Sans"]
LINE: dict[str, Any] = {
    "linewidth": 1.75,
    "marker": "o",
    "markersize": 6.5,
    "markeredgewidth": 1.5,
    "zorder": 3,
}
END_LABEL: dict[str, Any] = {
    "xytext": (-10, 0),
    "textcoords": "offset points",
    "ha": "right",
    "va": "center",
}
LEGEND: dict[str, Any] = {
    "loc": "upper left",
    "bbox_to_anchor": (0.03, 0.835),
    "frameon": False,
    "fontsize": 9,
}
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


Series = dict[tuple[str, int], list[Point]]


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta = next(row for row in rows if row["type"] == "meta")
    return meta, [row for row in rows if row["type"] == "run"]


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(row[key] for row in rows))


def summarize(runs: list[dict[str, Any]]) -> Series:
    """The median of the repeats at each (kind, concurrency, limit), sorted by limit."""
    groups: defaultdict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["kind"], run["concurrency"], run["limit_w"])].append(run)

    series: defaultdict[tuple[str, int], list[Point]] = defaultdict(list)
    for (kind, concurrency, limit), rows in sorted(groups.items()):
        metrics = (_median(rows, key) for key in ("tokens_per_s", "tokens_per_j", "mean_power_w"))
        series[(kind, concurrency)].append(Point(limit, *metrics))
    return dict(series)


def shares(points: list[Point], metric: Metric) -> list[float]:
    """Each point's metric as a % of the same metric at the highest limit."""
    base = getattr(points[-1], metric)
    return [100 * getattr(p, metric) / base for p in points]


def share_at(points: list[Point], limit_w: float, metric: Metric) -> float:
    return shares(points, metric)[[p.limit_w for p in points].index(limit_w)]


def sweet_spot(points: list[Point], keep: float = KEEP) -> Point:
    """The lowest limit at which it and every higher limit keep `keep`% of full speed."""
    speeds = shares(points, "tokens_per_s")
    i = len(points) - 1
    while i > 0 and speeds[i - 1] >= keep:
        i -= 1
    return points[i]


class Canvas:
    def __init__(self, meta: dict[str, Any], theme: Theme) -> None:
        self.meta = meta
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
        footer = f"{m['gpu']} · {m['model']} · engine {m['engine_version']} · GPU power from NVML"
        fig.text(0.04, 0.035, footer, fontsize=7.5, color=self.t.muted)
        if m["simulated"]:
            mark = "SIMULATED — NOT MEASURED"
            fig.text(0.5, 0.48, mark, color=self.t.muted, ha="center", va="center", **WATERMARK)

    def line(self, ax: Axes, points: list[Point], metric: Metric, slot: int, label: str) -> None:
        xs, ys = [p.limit_w for p in points], shares(points, metric)
        # A surface-coloured marker edge keeps dots legible where lines cross.
        ax.plot(xs, ys, label=label, color=self.t.series[slot], mec=self.t.surface, **LINE)

    def end_label(self, ax: Axes, points: list[Point], metric: Metric, text: str) -> None:
        """Labels a line at its lowest limit, where the lines spread apart."""
        xy = (points[0].limit_w, shares(points, metric)[0])
        ax.annotate(text, xy, fontsize=9, color=self.t.ink, **END_LABEL)

    def legend(self, fig: Figure, ax: Axes) -> None:
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=len(labels), **LEGEND)
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
    c: Canvas, s: Series, concurrency: int, metric: Metric, title: str, subtitle: str
) -> tuple[Figure, Axes]:
    decode, prefill = s[("decode", concurrency)], s[("prefill", concurrency)]
    fig, ax = c.chart(title, subtitle)
    c.line(ax, decode, metric, 0, "Decode (writing tokens)")
    c.line(ax, prefill, metric, 1, "Prefill (reading the prompt)")
    c.end_label(ax, decode, metric, "Decode")
    c.end_label(ax, prefill, metric, "Prefill")
    c.legend(fig, ax)
    watts_axis(ax, [p.limit_w for p in decode])
    return fig, ax


def speed_chart(c: Canvas, s: Series, concurrency: int, out: Path) -> None:
    decode, prefill = s[("decode", concurrency)], s[("prefill", concurrency)]
    spot, top = sweet_spot(decode).limit_w, decode[-1].limit_w
    d, p = share_at(decode, spot, "tokens_per_s"), share_at(prefill, spot, "tokens_per_s")
    title = f"At {spot:.0f} W, decode keeps {d:.0f}% of its speed. Prefill keeps {p:.0f}%."
    subtitle = f"Throughput as a share of the uncapped {top:.0f} W, {concurrency} in flight"

    fig, ax = decode_and_prefill(c, s, concurrency, "tokens_per_s", title, subtitle)
    ax.axvspan(spot, top, color=c.t.wash, zorder=0, linewidth=0)
    band = f"decode keeps ≥{KEEP:.0f}%"
    ax.text((spot + top) / 2, 6, band, ha="center", fontsize=8.5, color=c.t.ink2)
    percent_axis(ax)
    c.save(fig, out / "speed.png")


def efficiency_chart(c: Canvas, s: Series, concurrency: int, out: Path) -> None:
    decode, prefill = s[("decode", concurrency)], s[("prefill", concurrency)]
    spot, top = sweet_spot(decode).limit_w, decode[-1].limit_w
    gain = share_at(decode, spot, "tokens_per_j") - 100
    title = f"Capped at {spot:.0f} W, decode gets {gain:+.0f}% tokens per joule"
    subtitle = f"Tokens per joule as a share of the uncapped {top:.0f} W, {concurrency} in flight"

    fig, ax = decode_and_prefill(c, s, concurrency, "tokens_per_j", title, subtitle)
    ax.axhline(100, color=c.t.axis, linewidth=0.75, zorder=1)
    values = shares(decode, "tokens_per_j") + shares(prefill, "tokens_per_j")
    percent_axis(ax, min(80.0, min(values) - 5), max(values) + 10)
    c.save(fig, out / "efficiency.png")


def concurrency_chart(c: Canvas, s: Series, out: Path) -> None:
    keys = sorted(key for key in s if key[0] == "decode")[:3]
    if len(keys) < 2:
        return
    spot = sweet_spot(s[keys[-1]]).limit_w
    kept = ", ".join(f"{share_at(s[k], spot, 'tokens_per_s'):.0f}% at {k[1]}" for k in keys)
    title = f"At {spot:.0f} W, decode keeps {kept} in flight"
    subtitle = "Decode throughput as a share of uncapped, by requests in flight"

    fig, ax = c.chart(title, subtitle)
    for slot, key in enumerate(keys):
        c.line(ax, s[key], "tokens_per_s", slot, f"{key[1]} in flight")
    c.legend(fig, ax)
    watts_axis(ax, [p.limit_w for p in s[keys[-1]]])
    percent_axis(ax)
    c.save(fig, out / "concurrency.png")


def headline(c: Canvas, s: Series, concurrency: int, out: Path) -> None:
    """The one number: more decode tokens from the same power budget, at the sweet spot."""
    decode = s[("decode", concurrency)]
    spot, top = sweet_spot(decode).limit_w, decode[-1].limit_w
    gain = share_at(decode, spot, "tokens_per_j") - 100
    slower = 100 - share_at(decode, spot, "tokens_per_s")
    detail = (
        f"Every GPU capped at {spot:.0f} W ({100 * spot / top:.0f}% of max): "
        f"decode {slower:.0f}% slower, {concurrency} requests in flight"
    )
    caveat = "GPU power only: CPU, memory and cooling overheads are not counted."

    fig = c.blank()
    big = f"{gain:+.0f}%"
    fig.text(0.07, 0.62, big, fontsize=96, fontweight="bold", va="center", color=c.t.ink)
    fig.text(0.07, 0.36, "more decode tokens from the same megawatt", fontsize=20, color=c.t.ink)
    fig.text(0.07, 0.27, detail, fontsize=11, color=c.t.ink2)
    fig.text(0.07, 0.10, caveat, fontsize=8.5, color=c.t.muted)
    c.save(fig, out / "headline.png")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="?", default=Path("results.jsonl"))
    p.add_argument("--out", type=Path, default=Path("charts"))
    p.add_argument("--concurrency", type=int, help="batch for the main charts (default: largest)")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    args = p.parse_args(argv)

    meta, runs = load(args.results)
    s = summarize(runs)
    concurrency = args.concurrency or max(c for kind, c in s if kind == "decode")
    canvas = Canvas(meta, THEMES[args.theme])
    args.out.mkdir(parents=True, exist_ok=True)

    speed_chart(canvas, s, concurrency, args.out)
    efficiency_chart(canvas, s, concurrency, args.out)
    concurrency_chart(canvas, s, args.out)
    headline(canvas, s, concurrency, args.out)
    print(f"wrote charts to {args.out}/")


if __name__ == "__main__":
    main()
