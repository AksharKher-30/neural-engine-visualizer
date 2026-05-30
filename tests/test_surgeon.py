"""
Phase 2 — pytest for instrumented model.
Also integration-tests against Phase 1 base model.
Run: python -m pytest tests/test_surgeon.py -v
"""
import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_PKG  = "models/pretrained/mobilenetv2.mlpackage"
INSTR_PKG = "models/pretrained/mobilenetv2_instrumented.mlpackage"
TARGET_LAYERS = [1, 4, 11, 18]

EXPECTED_SHAPES = {
    "logits":              (1, 1000),
    "activation_layer_1":  (1, 16, 112, 112),
    "activation_layer_4":  (1, 32, 28, 28),
    "activation_layer_11": (1, 96, 14, 14),
    "activation_layer_18": (1, 1280, 7, 7),
}


@pytest.fixture(scope="module")
def base_model():
    import coremltools as ct
    if not os.path.exists(BASE_PKG):
        pytest.skip("Run phase 1 first: python models/convert.py")
    return ct.models.MLModel(BASE_PKG)


@pytest.fixture(scope="module")
def instr_model():
    import coremltools as ct
    if not os.path.exists(INSTR_PKG):
        pytest.skip("Run phase 2 first: python models/surgeon.py")
    return ct.models.MLModel(INSTR_PKG)


@pytest.fixture(scope="module")
def sample_img():
    # Deterministic — same image every run
    np.random.seed(42)
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── Phase 2 unit tests ────────────────────────────────────

def test_instrumented_pkg_exists():
    assert os.path.exists(INSTR_PKG)

def test_instrumented_model_loads(instr_model):
    assert instr_model is not None

def test_all_output_keys_present(instr_model, sample_img):
    out = instr_model.predict({"input": sample_img})
    for key in EXPECTED_SHAPES:
        assert key in out, f"Missing output: {key}"

def test_logits_shape(instr_model, sample_img):
    out = instr_model.predict({"input": sample_img})
    assert np.array(out["logits"]).flatten().shape == (1000,)

def test_activation_shapes(instr_model, sample_img):
    out = instr_model.predict({"input": sample_img})
    for layer_idx in TARGET_LAYERS:
        key = f"activation_layer_{layer_idx}"
        got = np.array(out[key]).shape
        expected = EXPECTED_SHAPES[key]
        assert got == expected, f"{key}: expected {expected}, got {got}"

def test_activations_not_all_zero(instr_model, sample_img):
    out = instr_model.predict({"input": sample_img})
    for layer_idx in TARGET_LAYERS:
        key = f"activation_layer_{layer_idx}"
        arr = np.array(out[key])
        assert arr.max() > 0, f"{key} all zeros — forward pass broken"

def test_activations_statistically_different(instr_model, sample_img):
    """Each layer must have distinct activation statistics."""
    out = instr_model.predict({"input": sample_img})
    means = [
        round(float(np.array(out[f"activation_layer_{i}"]).mean()), 4)
        for i in TARGET_LAYERS
    ]
    assert len(set(means)) > 1, f"All layer means identical: {means}"


# ── Integration test: Phase 1 ↔ Phase 2 ──────────────────

def test_logits_match_base_model(base_model, instr_model, sample_img):
    """
    INTEGRATION GATE: instrumented logits top-1 must equal base model top-1.
    Surgery must not change the forward pass computation.
    """
    base_out  = base_model.predict({"input": sample_img})
    instr_out = instr_model.predict({"input": sample_img})

    base_logits  = np.array(list(base_out.values())[0]).flatten()
    instr_logits = np.array(instr_out["logits"]).flatten()

    base_top1  = int(np.argmax(base_logits))
    instr_top1 = int(np.argmax(instr_logits))

    assert base_top1 == instr_top1, (
        f"Top-1 mismatch — surgery broke forward pass!\n"
        f"Base: class {base_top1}, Instrumented: class {instr_top1}"
    )