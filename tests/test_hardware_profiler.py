"""
Phase 6 — pytest for hardware profiler.
Run: python -m pytest tests/test_hardware_profiler.py -v
"""
import os
import sys
import csv
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer.hardware_profiler import (
    BenchStats,
    make_test_image,
    save_csv,
    CONFIGS,
    PKG_BASE,
    PKG_INSTR,
    CSV_OUT,
)


# ── Test image ────────────────────────────────────────────

def test_test_image_shape():
    img = make_test_image()
    assert img.size == (224, 224)
    assert img.mode == "RGB"

def test_test_image_deterministic():
    img1 = make_test_image(seed=42)
    img2 = make_test_image(seed=42)
    assert list(img1.getdata()) == list(img2.getdata())

def test_test_image_different_seeds():
    img1 = make_test_image(seed=0)
    img2 = make_test_image(seed=1)
    assert list(img1.getdata()) != list(img2.getdata())


# ── BenchStats ────────────────────────────────────────────

def make_stat(mean=5.0, std=0.5, n=100):
    raw = np.random.normal(mean, std, n).tolist()
    raw_arr = np.array(raw)
    return BenchStats(
        config="ALL (ANE+CPU+GPU)",
        model="Base model",
        mean_ms=float(raw_arr.mean()),
        std_ms=float(raw_arr.std()),
        min_ms=float(raw_arr.min()),
        p50_ms=float(np.percentile(raw_arr, 50)),
        p95_ms=float(np.percentile(raw_arr, 95)),
        throughput=1000.0 / float(raw_arr.mean()),
        raw=raw,
    )

def test_benchstat_throughput():
    s = make_stat(mean=10.0)
    assert abs(s.throughput - 100.0) < 5.0   # ~100 fps at 10ms

def test_benchstat_p95_gt_p50():
    s = make_stat(mean=5.0, std=1.0, n=200)
    assert s.p95_ms >= s.p50_ms

def test_benchstat_mean_in_range():
    s = make_stat(mean=5.0, std=0.1, n=500)
    assert 4.0 < s.mean_ms < 6.0


# ── CSV export ────────────────────────────────────────────

def test_csv_save_and_read(tmp_path):
    stats = [
        BenchStats("ALL (ANE+CPU+GPU)", "Base", 3.5, 0.2, 2.9, 3.4, 4.1, 285.7),
        BenchStats("CPU Only",          "Base", 18.2, 1.1, 15.0, 18.0, 20.5, 54.9),
    ]
    path = str(tmp_path / "test_bench.csv")
    save_csv(stats, path)
    assert os.path.exists(path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["config"] == "ALL (ANE+CPU+GPU)"
    assert float(rows[0]["mean_ms"]) == pytest.approx(3.5, abs=0.01)
    assert float(rows[1]["throughput_fps"]) == pytest.approx(54.9, abs=0.1)

def test_csv_has_all_fields(tmp_path):
    stats = [make_stat()]
    path  = str(tmp_path / "fields.csv")
    save_csv(stats, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
    required = ["model","config","mean_ms","std_ms",
                "min_ms","p50_ms","p95_ms","throughput_fps"]
    for field in required:
        assert field in fields, f"Missing CSV field: {field}"


# ── Config coverage ───────────────────────────────────────

def test_all_compute_configs_defined():
    assert len(CONFIGS) == 4
    assert "ALL (ANE+CPU+GPU)" in CONFIGS
    assert "CPU Only"          in CONFIGS
    assert "CPU + NE"          in CONFIGS
    assert "CPU + GPU"         in CONFIGS


# ── Integration: all phases intact ───────────────────────

def test_phase1_pkg_intact():
    assert os.path.exists(PKG_BASE),  "Phase 1 .mlpackage missing"

def test_phase2_pkg_intact():
    assert os.path.exists(PKG_INSTR), "Phase 2 .mlpackage missing"

def test_phase3_engine_binary():
    if not os.path.exists("engine/build/engine"):
        pytest.skip("Build engine first")
    assert os.access("engine/build/engine", os.X_OK)

def test_phase4_renderer_importable():
    from visualizer.activation_renderer import HeatmapRenderer
    assert HeatmapRenderer

def test_phase5_overlay_importable():
    from visualizer.overlay import compute_saliency
    assert compute_saliency

def test_profiler_importable():
    from visualizer.hardware_profiler import (
        ModelBenchmarker, plot_dashboard, save_csv, main
    )
    assert all([ModelBenchmarker, plot_dashboard, save_csv, main])