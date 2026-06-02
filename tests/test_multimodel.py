"""
Phase 8 — pytest for multi-model support.
Run: python -m pytest tests/test_multimodel.py -v
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_configs import MODEL_CONFIGS


# ── Model configs ─────────────────────────────────────────

def test_both_configs_exist():
    assert "mobilenetv2"    in MODEL_CONFIGS
    assert "efficientnet_b0" in MODEL_CONFIGS

def test_config_fields():
    required = ["pkg_base","pkg_instr","target_layers",
                "layer_shapes","label","params","port"]
    for name, cfg in MODEL_CONFIGS.items():
        for field in required:
            assert field in cfg, f"{name} missing field: {field}"

def test_ports_differ():
    p1 = MODEL_CONFIGS["mobilenetv2"]["port"]
    p2 = MODEL_CONFIGS["efficientnet_b0"]["port"]
    assert p1 != p2

def test_layer_shapes_consistent():
    for name, cfg in MODEL_CONFIGS.items():
        for idx in cfg["target_layers"]:
            assert idx in cfg["layer_shapes"], \
                f"{name}: layer {idx} in target_layers but not in layer_shapes"

def test_efficientnet_target_layers():
    layers = MODEL_CONFIGS["efficientnet_b0"]["target_layers"]
    assert 1 in layers
    assert 8 in layers   # final features (1280, 7, 7)


# ── EfficientNet-B0 PyTorch loading ──────────────────────

def test_efficientnet_loads():
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    assert model is not None

def test_efficientnet_output_shape():
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)

def test_efficientnet_feature_shapes():
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval()
    cfg     = MODEL_CONFIGS["efficientnet_b0"]
    x       = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            x = layer(x)
            if i in cfg["layer_shapes"]:
                expected = (1,) + cfg["layer_shapes"][i]
                assert tuple(x.shape) == expected, \
                    f"Layer {i}: expected {expected}, got {tuple(x.shape)}"


# ── Instrumented wrapper ──────────────────────────────────

def test_instrumented_efficientnet_forward():
    import torch
    from models.surgeon import InstrumentedEfficientNetB0
    model = InstrumentedEfficientNetB0([1, 3, 5, 8])
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert isinstance(out, tuple)
    assert len(out) == 5          # logits + 4 activations
    assert out[0].shape == (1, 1000)

def test_instrumented_efficientnet_layer_shapes():
    import torch
    from models.surgeon import InstrumentedEfficientNetB0
    cfg    = MODEL_CONFIGS["efficientnet_b0"]
    model  = InstrumentedEfficientNetB0(cfg["target_layers"])
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    for i, idx in enumerate(cfg["target_layers"]):
        expected = (1,) + cfg["layer_shapes"][idx]
        got      = tuple(out[i+1].shape)
        assert got == expected, f"Layer {idx}: expected {expected}, got {got}"


# ── Instrumented .mlpackage (if converted) ───────────────

def test_efficientnet_pkg_exists():
    pkg = MODEL_CONFIGS["efficientnet_b0"]["pkg_instr"]
    if not os.path.exists(pkg):
        pytest.skip("Run: python models/surgeon.py --model efficientnet_b0")
    assert os.path.exists(pkg)

def test_efficientnet_pkg_predicts():
    import coremltools as ct
    from PIL import Image
    pkg = MODEL_CONFIGS["efficientnet_b0"]["pkg_instr"]
    if not os.path.exists(pkg):
        pytest.skip("Run: python models/surgeon.py --model efficientnet_b0")
    model = ct.models.MLModel(pkg)
    img   = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out   = model.predict({"input": img})
    assert "logits" in out
    assert np.array(out["logits"]).flatten().shape == (1000,)


# ── Integration: all phases intact ───────────────────────

def test_phase1_mv2_intact():
    assert os.path.exists(MODEL_CONFIGS["mobilenetv2"]["pkg_base"])

def test_phase2_mv2_intact():
    assert os.path.exists(MODEL_CONFIGS["mobilenetv2"]["pkg_instr"])

def test_phase3_engine():
    if not os.path.exists("engine/build/engine"):
        pytest.skip("Build engine first")
    assert os.access("engine/build/engine", os.X_OK)

def test_all_previous_importable():
    from visualizer.activation_renderer import HeatmapRenderer
    from visualizer.overlay import compute_saliency
    from visualizer.hardware_profiler import ModelBenchmarker
    from visualizer.dashboard import render_activation_grid
    assert all([HeatmapRenderer, compute_saliency,
                ModelBenchmarker, render_activation_grid])