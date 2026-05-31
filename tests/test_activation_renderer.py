"""
Phase 4 — pytest for activation renderer.
Run: python -m pytest tests/test_activation_renderer.py -v
"""
import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer.activation_renderer import (
    normalize_channel,
    channel_to_rgb,
    grid_dims,
    ActivationBuffer,
    LAYER_INFO,
)

LAYER_INDICES = [1, 4, 11, 18]


# ── Normalization ─────────────────────────────────────────

def test_normalize_range():
    ch = np.random.randn(14, 14).astype(np.float32)
    out = normalize_channel(ch)
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_normalize_flat_channel():
    """All-same values → should return zeros (no divide by zero)."""
    ch = np.full((7, 7), 3.14, dtype=np.float32)
    out = normalize_channel(ch)
    assert np.allclose(out, 0.0)


def test_normalize_preserves_shape():
    ch = np.random.randn(112, 112).astype(np.float32)
    out = normalize_channel(ch)
    assert out.shape == ch.shape


# ── Colormap ──────────────────────────────────────────────

def test_channel_to_rgb_shape():
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis
    ch = np.random.rand(14, 14).astype(np.float32)
    rgb = channel_to_rgb(ch, cmap)
    assert rgb.shape == (14, 14, 3)
    assert rgb.dtype == np.uint8


def test_channel_to_rgb_range():
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis
    ch = np.random.rand(28, 28).astype(np.float32)
    rgb = channel_to_rgb(ch, cmap)
    assert rgb.min() >= 0
    assert rgb.max() <= 255


# ── Grid dims ─────────────────────────────────────────────

def test_grid_dims_layer1():
    rows, cols, n = grid_dims(16, 64)
    assert rows * cols >= n
    assert n == 16


def test_grid_dims_capped():
    rows, cols, n = grid_dims(1280, 64)
    assert n == 64
    assert rows * cols >= 64


def test_grid_dims_all_layers():
    for idx in LAYER_INDICES:
        info = LAYER_INFO[idx]
        rows, cols, n = grid_dims(info["channels"], 64)
        assert rows >= 1 and cols >= 1
        assert n == min(info["channels"], 64)


# ── ActivationBuffer thread safety ───────────────────────

def test_buffer_empty_on_init():
    buf = ActivationBuffer()
    arr, meta, count = buf.get()
    assert arr is None
    assert count == 0


def test_buffer_put_get():
    buf = ActivationBuffer()
    arr = np.random.randn(16, 112, 112).astype(np.float32)
    buf.put(arr, {"frame": 1, "fps": 15.0})
    out, meta, count = buf.get()
    assert out is not None
    assert out.shape == arr.shape
    assert count == 1
    assert meta["frame"] == 1


def test_buffer_overwrites_old():
    buf = ActivationBuffer()
    arr1 = np.ones((16, 112, 112), dtype=np.float32)
    arr2 = np.zeros((16, 112, 112), dtype=np.float32)
    buf.put(arr1, {})
    buf.put(arr2, {})
    out, _, count = buf.get()
    assert count == 2
    assert np.allclose(out, 0.0)  # latest wins


# ── ZMQ message format (no engine needed) ─────────────────

def test_zmq_message_parsing():
    shape = [1, 16, 112, 112]
    arr   = np.random.randn(*shape).astype(np.float32)
    hdr   = {"name": "activation_layer_1", "shape": shape,
             "frame": 7, "inference_ms": 1.4}

    # Simulate what ZMQ sends/receives
    hdr_bytes = json.dumps(hdr).encode()
    dat_bytes = arr.tobytes()

    parsed_hdr = json.loads(hdr_bytes.decode())
    recovered  = np.frombuffer(dat_bytes, dtype=np.float32).reshape(parsed_hdr["shape"])
    recovered  = recovered.squeeze(0)

    assert recovered.shape == (16, 112, 112)
    assert np.allclose(arr.squeeze(0), recovered)


# ── Integration: all previous phases intact ───────────────

def test_phase1_intact():
    assert os.path.exists("models/pretrained/mobilenetv2.mlpackage")

def test_phase2_intact():
    assert os.path.exists("models/pretrained/mobilenetv2_instrumented.mlpackage")

def test_phase3_engine_binary():
    if not os.path.exists("engine/build/engine"):
        pytest.skip("Build engine first")
    assert os.access("engine/build/engine", os.X_OK)

def test_renderer_module_importable():
    from visualizer.activation_renderer import (
        HeatmapRenderer, ZMQReceiver, ActivationBuffer, main
    )
    assert all([HeatmapRenderer, ZMQReceiver, ActivationBuffer, main])