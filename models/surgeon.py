"""
Phase 2 + Phase 8 — Model Surgery: expose intermediate layer activations.
Now supports MobileNetV2 and EfficientNet-B0.

Usage:
    python models/surgeon.py --model mobilenetv2
    python models/surgeon.py --model efficientnet_b0
    python models/surgeon.py --list-layers --model efficientnet_b0
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from PIL import Image

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_configs import MODEL_CONFIGS

# ── Instrumented wrappers ─────────────────────────────────

class InstrumentedMobileNetV2(nn.Module):
    def __init__(self, layer_indices):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features      = base.features
        self.avgpool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier    = base.classifier
        self.layer_indices = sorted(layer_indices)

    def forward(self, x):
        activations = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                activations.append(x)
        logits = self.classifier(torch.flatten(self.avgpool(x), 1))
        return tuple([logits] + activations)


class InstrumentedEfficientNetB0(nn.Module):
    def __init__(self, layer_indices):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features      = base.features
        self.avgpool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier    = base.classifier
        self.layer_indices = sorted(layer_indices)

    def forward(self, x):
        activations = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                activations.append(x)
        logits = self.classifier(torch.flatten(self.avgpool(x), 1))
        return tuple([logits] + activations)


WRAPPERS = {
    "mobilenetv2":    InstrumentedMobileNetV2,
    "efficientnet_b0": InstrumentedEfficientNetB0,
}


# ── Utilities ─────────────────────────────────────────────

def list_layers(model_name: str):
    if model_name == "mobilenetv2":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    cfg     = MODEL_CONFIGS[model_name]
    targets = cfg["target_layers"]
    x       = torch.randn(1, 3, 224, 224)
    print(f"\n{model_name} feature layers:")
    print(f"{'Index':<8} {'Shape':<25} {'Type'}")
    print("-" * 60)
    with torch.no_grad():
        for i, layer in enumerate(base.features):
            x = layer(x)
            marker = " ← TARGET" if i in targets else ""
            print(f"{i:<8} {str(tuple(x.shape)):<25} {type(layer).__name__}{marker}")
    print()


def build_and_trace(model_name: str, layer_indices):
    WrapperClass = WRAPPERS[model_name]
    model = WrapperClass(layer_indices)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    return traced


def convert(traced, layer_indices, compute_units=ct.ComputeUnit.ALL):
    import numpy as np
    scale  = 1.0 / (0.226 * 255.0)
    bias   = [-0.485 / 0.226, -0.456 / 0.226, -0.406 / 0.226]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]
    for idx in sorted(layer_indices):
        outputs.append(ct.TensorType(
            name=f"activation_layer_{idx}", dtype=np.float32))
    return ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="input", shape=(1, 3, 224, 224),
            scale=scale, bias=bias,
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=outputs,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS13,
    )


def verify(mlpackage_path: str, layer_indices):
    mlmodel = ct.models.MLModel(mlpackage_path)
    img     = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out = mlmodel.predict({"input": img})
    print(f"\nInstrumented model outputs [{mlpackage_path}]:")
    for k, v in out.items():
        print(f"  {k:<40} shape: {np.array(v).shape}")
    assert "logits" in out
    for idx in layer_indices:
        key = f"activation_layer_{idx}"
        assert key in out, f"Missing: {key}"
    print("✓ All expected outputs present.")


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="mobilenetv2",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output", default=None,
                        help="Override output path")
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU"])
    parser.add_argument("--list-layers", action="store_true")
    args = parser.parse_args()

    cfg     = MODEL_CONFIGS[args.model]
    layers  = cfg["target_layers"]
    out_pkg = args.output or cfg["pkg_instr"]

    if args.list_layers:
        list_layers(args.model)
        return

    unit_map = {
        "ALL":         ct.ComputeUnit.ALL,
        "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    }

    os.makedirs(os.path.dirname(out_pkg), exist_ok=True)

    print(f"Model         : {args.model}")
    print(f"Target layers : {layers}")
    print(f"Output        : {out_pkg}")
    print("Building + tracing...")
    traced = build_and_trace(args.model, layers)
    print("Converting to CoreML...")
    mlmodel = convert(traced, layers, unit_map[args.compute_units])
    print(f"Saving → {out_pkg}")
    mlmodel.save(out_pkg)
    print("Verifying...")
    verify(out_pkg, layers)
    print("Done.")


if __name__ == "__main__":
    main()