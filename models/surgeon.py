"""
Phase 2 — Model Surgery: expose intermediate layer activations.
Wraps MobileNetV2 to return logits + activation maps at target layers.

Usage:
    python models/surgeon.py --list-layers
    python models/surgeon.py
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

DEFAULT_LAYERS = [1, 4, 11, 18]


class InstrumentedMobileNetV2(nn.Module):
    """
    Returns tuple: (logits, act_layer_1, act_layer_4, act_layer_11, act_layer_18)
    Logits must be numerically identical to base MobileNetV2.
    """
    def __init__(self, layer_indices=DEFAULT_LAYERS):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = base.classifier
        self.layer_indices = sorted(layer_indices)

    def forward(self, x):
        activations = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                activations.append(x)
        pooled = self.avgpool(x)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return tuple([logits] + activations)


def list_layers():
    """Print all feature layer output shapes — useful for picking target indices."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    print(f"\n{'Index':<8} {'Shape':<25} {'Type'}")
    print("-" * 55)
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            x = layer(x)
            marker = " ← TARGET" if i in DEFAULT_LAYERS else ""
            print(f"{i:<8} {str(tuple(x.shape)):<25} {type(layer).__name__}{marker}")
    print()


def build_and_trace(layer_indices):
    model = InstrumentedMobileNetV2(layer_indices)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    return traced


def convert(traced, layer_indices, compute_units=ct.ComputeUnit.ALL):
    scale = 1.0 / (0.226 * 255.0)
    bias = [-0.485 / 0.226, -0.456 / 0.226, -0.406 / 0.226]

    outputs = [ct.TensorType(name="logits")]
    for idx in sorted(layer_indices):
        outputs.append(ct.TensorType(name=f"activation_layer_{idx}"))

    return ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="input",
            shape=(1, 3, 224, 224),
            scale=scale,
            bias=bias,
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=outputs,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS13,
    )


def verify(mlpackage_path, layer_indices):
    """Print all output names + shapes. Assert expected keys present."""
    mlmodel = ct.models.MLModel(mlpackage_path)
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    out = mlmodel.predict({"input": img})

    print("\nInstrumented model outputs:")
    for k, v in out.items():
        arr = np.array(v)
        print(f"  {k:<35} shape: {arr.shape}")

    assert "logits" in out, "logits missing!"
    for idx in layer_indices:
        key = f"activation_layer_{idx}"
        assert key in out, f"Missing: {key}"

    print("\n✓ All expected outputs present.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="models/pretrained/mobilenetv2_instrumented.mlpackage")
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU"])
    parser.add_argument("--list-layers", action="store_true")
    args = parser.parse_args()

    if args.list_layers:
        list_layers()
        return

    unit_map = {
        "ALL":         ct.ComputeUnit.ALL,
        "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Target layers : {args.layers}")
    print("Building + tracing instrumented model...")
    traced = build_and_trace(args.layers)

    print("Converting to CoreML...")
    mlmodel = convert(traced, args.layers, unit_map[args.compute_units])

    print(f"Saving → {args.output}")
    mlmodel.save(args.output)

    print("Verifying outputs...")
    verify(args.output, args.layers)
    print("Done.")


if __name__ == "__main__":
    main()