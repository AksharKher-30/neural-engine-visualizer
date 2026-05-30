"""
Phase 3 — pytest: ZMQ message format, float roundtrip, binary, Phase 1+2 regression.
Run: python -m pytest tests/test_engine.py -v
"""
import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENGINE_BIN = "engine/build/engine"
INSTR_PKG  = "models/pretrained/mobilenetv2_instrumented.mlpackage"
BASE_PKG   = "models/pretrained/mobilenetv2.mlpackage"
TARGET_LAYERS = [1, 4, 11, 18]

EXPECTED_SHAPES = {
    "logits":              [1, 1000],
    "activation_layer_1":  [1, 16, 112, 112],
    "activation_layer_4":  [1, 32, 28, 28],
    "activation_layer_11": [1, 96, 14, 14],
    "activation_layer_18": [1, 1280, 7, 7],
}


# ── ZMQ message format (no C++ needed) ───────────────────

def test_json_header_serialization():
    hdr = {"name": "activation_layer_1",
           "shape": [1, 16, 112, 112], "frame": 5, "inference_ms": 11.3}
    parsed = json.loads(json.dumps(hdr).encode().decode())
    assert parsed["name"]  == "activation_layer_1"
    assert parsed["shape"] == [1, 16, 112, 112]
    assert parsed["frame"] == 5


def test_float32_bytes_roundtrip():
    shape    = [1, 32, 28, 28]
    original = np.random.rand(*shape).astype(np.float32)
    recovered = np.frombuffer(original.tobytes(), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(original, recovered)


def test_all_tensor_shapes_parseable():
    for name, shape in EXPECTED_SHAPES.items():
        arr = np.random.rand(*shape).astype(np.float32)
        out = np.frombuffer(arr.tobytes(), dtype=np.float32).reshape(shape)
        assert out.shape == tuple(shape), f"Failed for {name}"


def test_zmq_bridge_importable():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "zmq_bridge", "transport/zmq_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
    assert hasattr(mod, "print_frame")


def test_engine_binary_exists():
    if not os.path.exists(ENGINE_BIN):
        pytest.skip("Build engine first — see run sequence below")
    assert os.path.isfile(ENGINE_BIN)
    assert os.access(ENGINE_BIN, os.X_OK), "Binary not executable"


# ── Integration: Phase 1 + Phase 2 regression ─────────────

def test_phase1_package_intact():
    assert os.path.exists(BASE_PKG), "Phase 1 .mlpackage missing"


def test_phase2_package_intact():
    assert os.path.exists(INSTR_PKG), "Phase 2 .mlpackage missing"


def test_phase2_model_still_predicts(tmp_path):
    """Full regression: instrumented model runs + returns all 5 outputs."""
    import coremltools as ct
    from PIL import Image
    np.random.seed(0)
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    model = ct.models.MLModel(INSTR_PKG)
    out   = model.predict({"input": img})
    assert "logits" in out
    for i in TARGET_LAYERS:
        key = f"activation_layer_{i}"
        assert key in out, f"Phase 2 regression: missing {key}"
    logits = np.array(out["logits"]).flatten()
    assert logits.shape == (1000,)
    assert logits.max() > logits.min()