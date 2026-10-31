"""Turns a sweep's results.jsonl into the charts.

Every title is computed from the data, so a run that disagrees with the expected story
says so in its own charts. A simulated run is watermarked on every image.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "wash": "#f0efec",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "wash": "#262624",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}
SIZE_IN = (8.0, 4.5)  # 1600 x 900 at 200 dpi
DPI = 200


@dataclass(frozen=True)
class Point:
    limit_w: float
    tokens_per_s: float
    tokens_per_j: float
    mean_power_w: float


Series = dict[tuple[str, int], list[Point]]


def load(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta = next(line for line in lines if line["type"] == "meta")
    return meta, [line for line in lines if line["type"] == "run"]


def summarize(runs: list[dict[str, object]]) -> Series:
    """Median of the repeats at each (workload, concurrency, limit), sorted by limit."""
    groups: dict[tuple[str, int, float], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        groups[(str(run["kind"]), int(run["concurrency"]), float(run["limit_w"]))].append(run)
    series: Series = defaultdict(list)
    for (kind, concurrency, limit), rows in groups.items():
        series[(kind, concurrency)].append(
            Point(
                limit_w=limit,
                tokens_per_s=statistics.median(float(r["tokens_per_s"]) for r in rows),
                tokens_per_j=statistics.median(float(r["tokens_per_j"]) for r in rows),
                mean_power_w=statistics.median(float(r["mean_power_w"]) for r in rows),
            )
        )
    return {key: sorted(points, key=lambda p: p.limit_w) for key, points in series.items()}


def relative(points: list[Point], metric: str) -> list[float]:
    """Each point's metric as a percentage of the same metric at the highest limit."""
    base = getattr(points[-1], metric)
    return [100 * getattr(p, metric) / base for p in points]


def sweet_spot(points: list[Point], keep: float = 95.0) -> Point:
    """The lowest limit at which this and every higher limit keep `keep`% of full speed."""
    speeds = relative(points, "tokens_per_s")
    best = points[-1]
    for point, speed in zip(reversed(points), reversed(speeds)):
        if speed < keep:
            break
        best = point
    return best


def at(points: list[Point], limit_w: float) -> Point:
    return next(p for p in points if p.limit_w == limit_w)


def pct(points: list[Point], limit_w: float, metric: str) -> float:
    return relative(points, metric)[points.index(at(points, limit_w))]


class Canvas:
    def __init__(self, meta: dict[str, object], theme: str) -> None:
        self.meta = meta
        self.t = THEMES[theme]
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
                "font.size": 10,
            }
        )

    def figure(self, title: str, subtitle: str) -> tuple[Figure, Axes]:
        fig = plt.figure(figsize=SIZE_IN, dpi=DPI, facecolor=self.t["surface"])
        ax = fig.add_axes((0.09, 0.15, 0.86, 0.6), facecolor=self.t["surface"])
        fig.text(0.04, 0.92, title, fontsize=15, fontweight="bold", color=self.t["ink"])
        fig.text(0.04, 0.86, subtitle, fontsize=10, color=self.t["ink2"])
        self._footer(fig)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(self.t["axis"])
        ax.tick_params(colors=self.t["muted"], length=0, labelsize=9)
        ax.grid(axis="y", color=self.t["grid"], linewidth=0.75)
        ax.set_axisbelow(True)
        return fig, ax

    def _footer(self, fig: Figure) -> None:
        m = self.meta
        source = (
            f"{m['gpu']} · {m['model']} · engine {m['engine_version']} · GPU power from NVML's energy counter"
        )
        fig.text(0.04, 0.035, source, fontsize=7.5, color=self.t["muted"])
        if m.get("simulated"):
            fig.text(
                0.5, 0.48, "SIMULATED — NOT MEASURED", fontsize=32, fontweight="bold",
                color=self.t["muted"], alpha=0.22, rotation=18, ha="center", va="center",
            )

    def line(self, ax: Axes, xs: list[float], ys: list[float], slot: int, label: str) -> None:
        ax.plot(
            xs, ys, color=self.t["series"][slot], linewidth=1.75, solid_capstyle="round",
            solid_joinstyle="round", marker="o", markersize=6.5,
            markeredgecolor=self.t["surface"], markeredgewidth=1.5, label=label, zorder=3,
        )

    def end_label(self, ax: Axes, x: float, y: float, text: str) -> None:
        ax.annotate(
            text, (x, y), xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
            fontsize=9, color=self.t["ink"],
        )

    def legend(self, fig: Figure, ax: Axes) -> None:
        """One row under the subtitle, clear of the data."""
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(0.03, 0.835), ncol=len(labels),
            frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.8,
        )
        for text in legend.get_texts():
            text.set_color(self.t["ink2"])

    def save(self, fig: Figure, path: Path) -> None:
        fig.savefig(path, facecolor=self.t["surface"])
        plt.close(fig)


def watts_axis(ax: Axes, limits: list[float], max_w: float) -> None:
    ax.set_xticks(limits)
    ax.set_xticklabels([f"{w:.0f} W" for w in limits])
    ax.set_xlim(limits[0] - (max_w - limits[0]) * 0.18, max_w + (max_w - limits[0]) * 0.03)


def speed_chart(c: Canvas, s: Series, concurrency: int, out: Path) -> None:
    decode, prefill = s[("decode", concurrency)], s[("prefill", concurrency)]
    spot = sweet_spot(decode)
    limits = [p.limit_w for p in decode]
    d, p = pct(decode, spot.limit_w, "tokens_per_s"), pct(prefill, spot.limit_w, "tokens_per_s")
    title = f"At {spot.limit_w:.0f} W, decode keeps {d:.0f}% of its speed. Prefill keeps {p:.0f}%."
    subtitle = (
        f"Throughput as a share of the uncapped {limits[-1]:.0f} W, {concurrency} requests in flight"
    )
    fig, ax = c.figure(title, subtitle)
    ax.axvspan(spot.limit_w, limits[-1], color=c.t["wash"], zorder=0, linewidth=0)
    ax.text(
        (spot.limit_w + limits[-1]) / 2, 6, "decode keeps ≥95%", ha="center", fontsize=8.5,
        color=c.t["ink2"],
    )
    c.line(ax, limits, relative(decode, "tokens_per_s"), 0, "Decode (writing tokens)")
    c.line(ax, [q.limit_w for q in prefill], relative(prefill, "tokens_per_s"), 1, "Prefill (reading the prompt)")
    c.end_label(ax, limits[0], relative(decode, "tokens_per_s")[0], "Decode")
    c.end_label(ax, prefill[0].limit_w, relative(prefill, "tokens_per_s")[0], "Prefill")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    watts_axis(ax, limits, limits[-1])
    c.legend(fig, ax)
    c.save(fig, out / "speed.png")


def efficiency_chart(c: Canvas, s: Series, concurrency: int, out: Path) -> None:
    decode, prefill = s[("decode", concurrency)], s[("prefill", concurrency)]
    spot = sweet_spot(decode)
    gain = pct(decode, spot.limit_w, "tokens_per_j") - 100
    limits = [p.limit_w for p in decode]
    title = f"Capped at {spot.limit_w:.0f} W, decode gets {gain:+.0f}% tokens per joule"
    subtitle = f"Tokens per joule as a share of the uncapped {limits[-1]:.0f} W, {concurrency} requests in flight"
    fig, ax = c.figure(title, subtitle)
    d_eff, p_eff = relative(decode, "tokens_per_j"), relative(prefill, "tokens_per_j")
    ax.axhline(100, color=c.t["axis"], linewidth=0.75, zorder=1)
    c.line(ax, limits, d_eff, 0, "Decode (writing tokens)")
    c.line(ax, [q.limit_w for q in prefill], p_eff, 1, "Prefill (reading the prompt)")
    c.end_label(ax, limits[0], d_eff[0], "Decode")
    c.end_label(ax, prefill[0].limit_w, p_eff[0], "Prefill")
    top = max(d_eff + p_eff)
    ax.set_ylim(min(80.0, min(d_eff + p_eff) - 5), top + (top - 80) * 0.15)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    watts_axis(ax, limits, limits[-1])
    c.legend(fig, ax)
    c.save(fig, out / "efficiency.png")


def concurrency_chart(c: Canvas, s: Series, out: Path) -> None:
    keys = sorted(k for k in s if k[0] == "decode")[:3]
    if len(keys) < 2:
        return
    high = s[keys[-1]]
    spot = sweet_spot(high)
    kept = ", ".join(f"{pct(s[k], spot.limit_w, 'tokens_per_s'):.0f}% at {k[1]}" for k in keys)
    title = f"At {spot.limit_w:.0f} W, decode keeps {kept} in flight"
    subtitle = "Decode throughput as a share of uncapped, by requests in flight: bigger batches do more math"
    fig, ax = c.figure(title, subtitle)
    for slot, key in enumerate(keys):
        points = s[key]
        c.line(ax, [p.limit_w for p in points], relative(points, "tokens_per_s"), slot, f"{key[1]} in flight")
    limits = [p.limit_w for p in high]
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    watts_axis(ax, limits, limits[-1])
    c.legend(fig, ax)
    c.save(fig, out / "concurrency.png")


def headline(c: Canvas, s: Series, concurrency: int, out: Path) -> None:
    """The one number: more decode tokens from the same power budget, at the sweet spot."""
    decode = s[("decode", concurrency)]
    spot = sweet_spot(decode)
    gain = pct(decode, spot.limit_w, "tokens_per_j") - 100
    slower = 100 - pct(decode, spot.limit_w, "tokens_per_s")
    share = 100 * spot.limit_w / decode[-1].limit_w
    fig = plt.figure(figsize=SIZE_IN, dpi=DPI, facecolor=c.t["surface"])
    fig.text(0.07, 0.62, f"{gain:+.0f}%", fontsize=96, fontweight="bold", color=c.t["ink"], va="center")
    fig.text(0.07, 0.36, "more decode tokens from the same megawatt", fontsize=20, color=c.t["ink"])
    fig.text(
        0.07, 0.27,
        f"Every GPU capped at {spot.limit_w:.0f} W ({share:.0f}% of max): decode {slower:.0f}% slower, "
        f"{concurrency} requests in flight",
        fontsize=11, color=c.t["ink2"],
    )
    fig.text(
        0.07, 0.10, "GPU power only. A facility's CPU, memory and cooling overheads are not counted.",
        fontsize=8.5, color=c.t["muted"],
    )
    c._footer(fig)
    c.save(fig, out / "headline.png")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="?", default=Path("results.jsonl"))
    p.add_argument("--out", type=Path, default=Path("charts"))
    p.add_argument("--concurrency", type=int, help="the batch the main charts use (default: largest)")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    args = p.parse_args(argv)

    meta, runs = load(args.results)
    series = summarize(runs)
    concurrency = args.concurrency or max(k[1] for k in series if k[0] == "decode")
    args.out.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(meta, args.theme)
    speed_chart(canvas, series, concurrency, args.out)
    efficiency_chart(canvas, series, concurrency, args.out)
    concurrency_chart(canvas, series, args.out)
    headline(canvas, series, concurrency, args.out)
    print(f"wrote charts to {args.out}/")


if __name__ == "__main__":
    main()
