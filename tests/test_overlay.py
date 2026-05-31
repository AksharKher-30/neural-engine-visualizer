"""
Phase 5 — pytest for overlay pipeline.
Run: python -m pytest tests/test_overlay.py -v
"""
import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer.overlay import (
    compute_saliency,
    saliency_to_heatmap,
    alpha_blend,
    softmax,
    load_imagenet_labels,
    TensorBuffer,
    LAYER_NAMES,
    COLORMAPS,
)


# ── Saliency computation ──────────────────────────────────

def test_saliency_shape():
    arr = np.random.randn(16, 112, 112).astype(np.float32)
    s   = compute_saliency(arr)
    assert s.shape == (112, 112)

def test_saliency_range():
    arr = np.random.randn(96, 14, 14).astype(np.float32)
    s   = compute_saliency(arr)
    assert s.min() >= 0.0 - 1e-6
    assert s.max() <= 1.0 + 1e-6

def test_saliency_relu():
    """Negative-only activation → all zeros after ReLU."""
    arr = -np.abs(np.random.randn(16, 112, 112).astype(np.float32))
    s   = compute_saliency(arr)
    assert np.allclose(s, 0.0)

def test_saliency_all_layers():
    shapes = {1:(16,112,112), 4:(32,28,28), 11:(96,14,14), 18:(1280,7,7)}
    for idx, shape in shapes.items():
        arr = np.random.randn(*shape).astype(np.float32)
        s   = compute_saliency(arr)
        assert s.shape == shape[1:], f"Layer {idx} saliency shape wrong"

def test_saliency_flat_input():
    arr = np.ones((16, 112, 112), dtype=np.float32)
    s   = compute_saliency(arr)
    assert np.allclose(s, 0.0)


# ── Heatmap ───────────────────────────────────────────────

def test_heatmap_shape():
    import cv2
    s = np.random.rand(112, 112).astype(np.float32)
    h = saliency_to_heatmap(s, 480, 640, cv2.COLORMAP_JET)
    assert h.shape == (480, 640, 3)
    assert h.dtype == np.uint8

def test_heatmap_all_colormaps():
    import cv2
    s = np.random.rand(14, 14).astype(np.float32)
    for cmap in COLORMAPS:
        h = saliency_to_heatmap(s, 240, 320, cmap)
        assert h.shape == (240, 320, 3)


# ── Alpha blend ───────────────────────────────────────────

def test_alpha_blend_shape():
    frame   = np.random.randint(0, 255, (480,640,3), dtype=np.uint8)
    heatmap = np.random.randint(0, 255, (480,640,3), dtype=np.uint8)
    out = alpha_blend(frame, heatmap, alpha=0.4)
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8

def test_alpha_zero_returns_frame():
    frame   = np.full((480,640,3), 100, dtype=np.uint8)
    heatmap = np.full((480,640,3), 200, dtype=np.uint8)
    out = alpha_blend(frame, heatmap, alpha=0.0)
    assert np.allclose(out.astype(float), frame.astype(float), atol=2)

def test_alpha_one_returns_heatmap():
    frame   = np.full((480,640,3), 100, dtype=np.uint8)
    heatmap = np.full((480,640,3), 200, dtype=np.uint8)
    out = alpha_blend(frame, heatmap, alpha=1.0)
    assert np.allclose(out.astype(float), heatmap.astype(float), atol=2)


# ── Softmax + predictions ─────────────────────────────────

def test_softmax_sums_to_one():
    x = np.random.randn(1000).astype(np.float32)
    p = softmax(x)
    assert abs(p.sum() - 1.0) < 1e-5

def test_softmax_all_positive():
    x = np.random.randn(1000).astype(np.float32)
    p = softmax(x)
    assert (p >= 0).all()

def test_imagenet_labels_count():
    if not os.path.exists("data/imagenet_labels.txt"):
        pytest.skip("Run: curl ... imagenet_labels.txt")
    labels = load_imagenet_labels()
    assert len(labels) == 1000


# ── TensorBuffer ──────────────────────────────────────────

def test_buffer_stores_multiple_tensors():
    buf = TensorBuffer()
    buf.put("activation_layer_1",  np.zeros((16,112,112)), {})
    buf.put("activation_layer_18", np.zeros((1280,7,7)),   {})
    buf.put("logits",              np.zeros((1000,)),       {})
    tensors, _ = buf.get_all()
    assert "activation_layer_1"  in tensors
    assert "activation_layer_18" in tensors
    assert "logits"              in tensors

def test_buffer_has_update():
    buf = TensorBuffer()
    assert not buf.has_update()
    buf.put("logits", np.zeros((1000,)), {})
    assert buf.has_update()
    buf.get_all()
    assert not buf.has_update()


# ── Integration: all phases intact ────────────────────────

def test_phase1_intact():
    assert os.path.exists("models/pretrained/mobilenetv2.mlpackage")

def test_phase2_intact():
    assert os.path.exists("models/pretrained/mobilenetv2_instrumented.mlpackage")

def test_phase3_engine_binary():
    if not os.path.exists("engine/build/engine"):
        pytest.skip("Build engine first")
    assert os.access("engine/build/engine", os.X_OK)

def test_phase4_renderer_importable():
    from visualizer.activation_renderer import HeatmapRenderer
    assert HeatmapRenderer

def test_overlay_importable():
    from visualizer.overlay import main, compute_saliency, alpha_blend
    assert all([main, compute_saliency, alpha_blend])