"""
Phase 7 — pytest for unified dashboard.
Run: python -m pytest tests/test_dashboard.py -v
"""
import os
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer.dashboard import (
    render_activation_grid,
    render_hud,
    render_waiting,
    TensorBuffer,
    PANEL_W, PANEL_H, HUD_H, TOTAL_W, TOTAL_H,
    LAYER_MAP, LAYER_MAX_CHANNELS,
)


# ── Layout constants ──────────────────────────────────────

def test_total_dimensions():
    assert TOTAL_W == PANEL_W * 2
    assert TOTAL_H == PANEL_H + HUD_H

def test_layer_map_keys():
    assert ord("1") in LAYER_MAP
    assert ord("4") in LAYER_MAP
    assert ord("b") in LAYER_MAP
    assert ord("e") in LAYER_MAP
    assert LAYER_MAP[ord("b")] == 11
    assert LAYER_MAP[ord("e")] == 18


# ── Activation grid ───────────────────────────────────────

def test_grid_shape_layer1():
    import cv2
    arr = np.random.randn(16, 112, 112).astype(np.float32)
    out = render_activation_grid(arr, PANEL_W, PANEL_H,
                                  cv2.COLORMAP_JET, 16, 1)
    assert out.shape == (PANEL_H, PANEL_W, 3)
    assert out.dtype == np.uint8

def test_grid_shape_layer18():
    import cv2
    arr = np.random.randn(1280, 7, 7).astype(np.float32)
    out = render_activation_grid(arr, PANEL_W, PANEL_H,
                                  cv2.COLORMAP_VIRIDIS, 64, 18)
    assert out.shape == (PANEL_H, PANEL_W, 3)

def test_grid_all_layers():
    import cv2
    shapes = {1:(16,112,112), 4:(32,28,28), 11:(96,14,14), 18:(1280,7,7)}
    for idx, shape in shapes.items():
        arr = np.random.randn(*shape).astype(np.float32)
        out = render_activation_grid(arr, PANEL_W, PANEL_H,
                                      cv2.COLORMAP_HOT,
                                      LAYER_MAX_CHANNELS[idx], idx)
        assert out.shape == (PANEL_H, PANEL_W, 3), f"Layer {idx} grid wrong shape"

def test_grid_caps_channels():
    import cv2
    arr = np.random.randn(1280, 7, 7).astype(np.float32)
    out = render_activation_grid(arr, PANEL_W, PANEL_H,
                                  cv2.COLORMAP_JET, 16, 18)
    assert out.shape == (PANEL_H, PANEL_W, 3)

def test_grid_flat_activation_no_crash():
    """All-zero activation should not crash (no divide by zero)."""
    import cv2
    arr = np.zeros((16, 112, 112), dtype=np.float32)
    out = render_activation_grid(arr, PANEL_W, PANEL_H,
                                  cv2.COLORMAP_JET, 16, 1)
    assert out.shape == (PANEL_H, PANEL_W, 3)


# ── HUD ───────────────────────────────────────────────────

def test_hud_shape():
    state = {"layer": 1, "alpha": 0.4, "cmap_idx": 0, "show_pred": True}
    meta  = {"inference_ms": 4.3, "zmq_fps": 30.0}
    hud   = render_hud(TOTAL_W, state, meta, 28.0, False)
    assert hud.shape == (HUD_H, TOTAL_W, 3)
    assert hud.dtype == np.uint8

def test_hud_recording_flag():
    state = {"layer": 1, "alpha": 0.4, "cmap_idx": 0, "show_pred": True}
    meta  = {"inference_ms": 4.3, "zmq_fps": 30.0}
    hud_rec = render_hud(TOTAL_W, state, meta, 28.0, True)
    hud_nor = render_hud(TOTAL_W, state, meta, 28.0, False)
    # Pixel values must differ when recording (red channel changes)
    assert not np.array_equal(hud_rec, hud_nor)


# ── Waiting screen ────────────────────────────────────────

def test_waiting_shape():
    out = render_waiting(PANEL_W, PANEL_H, "Waiting...")
    assert out.shape == (PANEL_H, PANEL_W, 3)


# ── TensorBuffer ──────────────────────────────────────────

def test_buffer_stores_all_tensors():
    buf = TensorBuffer()
    buf.put("activation_layer_1",  np.zeros((16,112,112)), {})
    buf.put("activation_layer_18", np.zeros((1280,7,7)),   {})
    buf.put("logits",              np.zeros(1000),          {})
    t, _ = buf.get_all()
    assert "activation_layer_1"  in t
    assert "activation_layer_18" in t
    assert "logits"              in t

def test_buffer_latest_wins():
    buf = TensorBuffer()
    buf.put("logits", np.ones(1000),  {})
    buf.put("logits", np.zeros(1000), {})
    t, _ = buf.get_all()
    assert np.allclose(t["logits"], 0.0)


# ── Integration: all phases intact ───────────────────────

def test_phase1_intact():
    assert os.path.exists("models/pretrained/mobilenetv2.mlpackage")

def test_phase2_intact():
    assert os.path.exists("models/pretrained/mobilenetv2_instrumented.mlpackage")

def test_phase3_engine():
    if not os.path.exists("engine/build/engine"):
        pytest.skip("Build engine first")
    assert os.access("engine/build/engine", os.X_OK)

def test_phase4_importable():
    from visualizer.activation_renderer import HeatmapRenderer
    assert HeatmapRenderer

def test_phase5_importable():
    from visualizer.overlay import compute_saliency, alpha_blend
    assert compute_saliency and alpha_blend

def test_phase6_importable():
    from visualizer.hardware_profiler import ModelBenchmarker
    assert ModelBenchmarker

def test_dashboard_importable():
    from visualizer.dashboard import main, render_activation_grid, render_hud
    assert all([main, render_activation_grid, render_hud])