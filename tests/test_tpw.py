import json
from pathlib import Path

import pytest

from tpw import plot, sweep


def test_power_limits_run_from_max_to_min_rounded_between():
    assert sweep.power_limits(200, 700, 6) == [700, 600, 500, 400, 300, 200]
    assert sweep.power_limits(150.5, 350, 3) == [350, 250, 150.5]
    with pytest.raises(ValueError):
        sweep.power_limits(200, 700, 1)


def point(limit, speed, per_j=1.0):
    return plot.Point(limit_w=limit, tokens_per_s=speed, tokens_per_j=per_j, mean_power_w=limit)


def test_sweet_spot_is_lowest_limit_that_keeps_speed_all_the_way_up():
    points = [point(200, 90), point(300, 97), point(400, 93), point(500, 98), point(600, 100)]
    assert plot.sweet_spot(points).limit_w == 500


def test_sweet_spot_is_the_max_when_any_cut_costs_too_much():
    assert plot.sweet_spot([point(500, 80), point(600, 100)]).limit_w == 600


def test_summarize_takes_the_median_of_repeats():
    runs = [
        {"kind": "decode", "concurrency": 1, "limit_w": 300, "tokens_per_s": s, "tokens_per_j": 1, "mean_power_w": 1}
        for s in (10, 30, 20)
    ]
    assert plot.summarize(runs)[("decode", 1)] == [plot.Point(300, 20, 1, 1)]


def test_sweep_restores_the_original_limit_when_a_run_fails(monkeypatch):
    from tpw import sim

    gpu = sim.SimGpu()
    gpu.set_limit_w(650)
    monkeypatch.setattr(sim, "SimGpu", lambda: gpu)
    monkeypatch.setattr(sweep, "measure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        sweep.main(["--simulate", "--limits", "300", "--duration", "0.1", "--settle", "0", "--out", "/dev/null"])
    assert gpu.limit_w() == 650


def test_simulated_sweep_writes_results_and_plots(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    sweep.main(
        [
            "--simulate", "--limits", "700,450,250", "--concurrency", "1,8",
            "--duration", "0.4", "--repeats", "1", "--settle", "0",
            "--decode-tokens", "16", "--prefill-tokens", "512", "--out", str(results),
        ]
    )
    lines = [json.loads(line) for line in results.read_text().splitlines()]
    meta, runs = lines[0], lines[1:]
    assert meta["type"] == "meta" and meta["simulated"] is True
    assert len(runs) == 3 * 2 * 2
    assert all(r["work_tokens"] > 0 and r["energy_j"] > 0 for r in runs)

    by = plot.summarize(runs)
    decode_speed = plot.relative(by[("decode", 8)], "tokens_per_s")
    prefill_speed = plot.relative(by[("prefill", 8)], "tokens_per_s")
    # The cartoon's premise: a deep cut costs prefill more than decode.
    assert prefill_speed[0] < decode_speed[0] < 100.5

    charts = tmp_path / "charts"
    plot.main([str(results), "--out", str(charts)])
    assert sorted(p.name for p in charts.iterdir()) == [
        "concurrency.png", "efficiency.png", "headline.png", "speed.png",
    ]
