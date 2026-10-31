import json
from pathlib import Path

import pytest

from tpw import load, plot, sim, sweep


def point(limit_w: float, tokens_per_s: float) -> plot.Point:
    return plot.Point(limit_w, tokens_per_s, tokens_per_j=1.0, mean_power_w=limit_w)


def curve(*mids: tuple[float, float]) -> list[plot.Band]:
    return [plot.Band(limit, mid, mid, mid) for limit, mid in mids]


def test_power_limits_run_from_max_to_min_with_the_steps_between_rounded():
    assert sweep.power_limits(200, 700, 6) == [700, 600, 500, 400, 300, 200]
    assert sweep.power_limits(150.5, 350, 3) == [350, 250, 150.5]
    with pytest.raises(ValueError):
        sweep.power_limits(200, 700, 1)


def test_sweet_spot_is_the_lowest_limit_that_keeps_speed_all_the_way_up():
    speed = curve((200, 90), (300, 97), (400, 93), (500, 98), (600, 100))
    assert plot.sweet_spot(speed) == 500


def test_sweet_spot_is_the_max_when_any_cut_costs_too_much():
    assert plot.sweet_spot(curve((500, 80), (600, 100))) == 600


def test_summarize_takes_the_median_of_repeats_in_limit_order():
    run = {"kind": "decode", "concurrency": 1, "tokens_per_j": 1, "mean_power_w": 1}
    run["median_sm_clock_mhz"] = 1
    speeds = [(300, 10), (300, 30), (300, 20), (200, 5)]
    runs = [{**run, "limit_w": w, "tokens_per_s": s} for w, s in speeds]

    points = plot.summarize(runs)[("decode", 1)]
    assert points == [plot.Point(200, 5, 1, 1, 1), plot.Point(300, 20, 1, 1, 1)]


def test_bands_normalise_each_chip_then_take_the_median_and_range():
    key = ("decode", 8)
    chips = [{key: [point(300, 90 * scale), point(400, 100 * scale)]} for scale in (1, 2, 3)]
    chips[2][key][0] = point(300, 3 * 80)

    low, top = plot.bands(chips, key, "tokens_per_s")
    assert low == plot.Band(300, mid=90, low=80, high=90)
    assert top == plot.Band(400, mid=100, low=100, high=100)


def test_bands_can_keep_raw_values():
    key = ("decode", 8)
    chips = [{key: [plot.Point(300, 1, 1, 1, sm_clock_mhz=900), plot.Point(400, 1, 1, 1, 1980)]}]
    assert [b.mid for b in plot.bands(chips, key, "sm_clock_mhz", relative=False)] == [900, 1980]


def test_titles_give_the_cause_only_when_the_data_shows_it():
    decode, prefill = (
        curve((300, 90), (400, 96), (500, 100)),
        curve((300, 50), (400, 75), (500, 100)),
    )
    assert plot.speed_title(decode, prefill, 400) == "A power cap slows prefill, not decode"
    assert plot.speed_title(decode, decode, 400).startswith("At 400 W")

    rising_then_falling = curve((300, 120), (400, 140), (500, 100))
    assert plot.efficiency_title(rising_then_falling).endswith("down to 400 W")
    assert plot.efficiency_title(curve((300, 130), (500, 100))).startswith("Every cut")
    assert plot.efficiency_title(curve((300, 90), (500, 100))).startswith("A power cap doesn't")

    title, limit = plot.concurrency_title(decode, prefill)
    assert title.startswith("Bigger batches") and limit == 300


def test_holds_until_finds_where_a_curve_leaves_its_uncapped_value():
    clock = curve((200, 345), (300, 1500), (400, 1950), (500, 1980))
    assert plot.holds_until(clock, 97) == 400


def test_bands_refuse_chips_swept_over_different_limits():
    key = ("decode", 8)
    chips = [{key: [point(300, 1), point(400, 1)]}, {key: [point(250, 1), point(400, 1)]}]
    with pytest.raises(ValueError):
        plot.bands(chips, key, "tokens_per_s")


def test_percentile_is_nearest_rank():
    assert load.percentile([], 0.5) is None
    assert load.percentile([3, 1, 2], 0.5) == 2
    assert load.percentile(list(range(100)), 0.99) == 99


def test_sweep_restores_the_original_limit_when_a_run_fails(monkeypatch, tmp_path: Path):
    gpu = sim.SimGpu()
    gpu.set_limit_w(650)
    monkeypatch.setattr(sim, "SimGpu", lambda: gpu)

    def fail(*_args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(sweep, "measure", fail)
    with pytest.raises(RuntimeError, match="boom"):
        sweep.main(f"--simulate --limits 300 --settle 0 --out {tmp_path / 'r.jsonl'}".split())
    assert gpu.limit_w() == 650


def test_a_simulated_sweep_records_every_run_and_plots_every_chart(tmp_path: Path):
    results, charts = tmp_path / "results.jsonl", tmp_path / "charts"
    sweep.main(
        f"--simulate --limits 700,450,250 --concurrency 1,8 --duration 0.4 --repeats 1 --settle 0 "
        f"--decode-tokens 16 --prefill-tokens 512 --out {results}".split()
    )

    meta, *runs = (json.loads(line) for line in results.read_text().splitlines())
    assert meta["type"] == "meta" and meta["simulated"]
    assert len(runs) == 3 * 2 * 2
    assert all(run["work_tokens"] > 0 and run["energy_j"] > 0 for run in runs)

    s = plot.summarize(runs)
    decode, prefill = (plot.shares(s[(kind, 8)], "tokens_per_s") for kind in ("decode", "prefill"))
    assert prefill[0] < decode[0], "the simulator's premise: a deep cut costs prefill more"

    plot.main([str(results), "--out", str(charts)])
    names = {path.name for path in charts.iterdir()}
    assert names == {"speed.png", "clock.png", "efficiency.png", "concurrency.png", "headline.png"}
