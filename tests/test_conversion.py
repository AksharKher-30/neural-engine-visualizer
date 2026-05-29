"""
Phase 1 — pytest suite for conversion pipeline
Run: pytest tests/test_conversion.py -v
"""
import os
import sys
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MLPACKAGE = "models/pretrained/mobilenetv2.mlpackage"


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pt_model():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    m.eval()
    return m


@pytest.fixture(scope="module")
def cm_model():
    import coremltools as ct
    if not os.path.exists(MLPACKAGE):
        pytest.skip(f"MLPackage missing. Run: python models/convert.py")
    return ct.models.MLModel(MLPACKAGE)


@pytest.fixture(scope="module")
def sample_image():
    """Deterministic gradient image — no external file needed."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        arr[i, :] = [i // 2, 128, 255 - i // 2]
    return Image.fromarray(arr)


# ── tests ─────────────────────────────────────────────────

def test_pytorch_loads(pt_model):
    assert pt_model is not None

def test_pytorch_output_shape(pt_model):
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = pt_model(dummy)
    assert out.shape == (1, 1000), f"Got {out.shape}, expected (1, 1000)"

def test_mlpackage_exists():
    assert os.path.exists(MLPACKAGE), \
        f"MLPackage not found at {MLPACKAGE}. Run convert.py first."

def test_coreml_loads(cm_model):
    assert cm_model is not None

def test_coreml_runs(cm_model, sample_image):
    out = cm_model.predict({"input": sample_image})
    assert out is not None
    assert len(out) > 0

def test_coreml_output_has_1000_classes(cm_model, sample_image):
    out = cm_model.predict({"input": sample_image})
    logits = list(out.values())[0].flatten()
    assert len(logits) == 1000, f"Expected 1000 classes, got {len(logits)}"

def test_prediction_parity(pt_model, cm_model, sample_image):
    """Top-5 overlap ≥ 3 between PyTorch and CoreML on same image."""
    from torchvision import transforms
    import coremltools as ct  # noqa

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tfm(sample_image).unsqueeze(0)
    with torch.no_grad():
        pt_logits = pt_model(tensor)[0]
    pt_top5 = torch.topk(pt_logits, 5).indices.numpy().tolist()

    cm_out = cm_model.predict({"input": sample_image})
    cm_logits = list(cm_out.values())[0].flatten()
    cm_top5 = np.argsort(cm_logits)[-5:][::-1].tolist()

    overlap = len(set(pt_top5) & set(cm_top5))
    assert overlap >= 3, \
        f"Top-5 overlap too low: {overlap}/5\nPyTorch: {pt_top5}\nCoreML:  {cm_top5}"