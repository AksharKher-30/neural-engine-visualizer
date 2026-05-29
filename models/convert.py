"""
Phase 1 — Convert MobileNetV2 PyTorch → CoreML .mlpackage
Usage: python models/convert.py --output models/pretrained/mobilenetv2.mlpackage
"""
import os
import argparse
import torch
import coremltools as ct


def load_mobilenetv2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    print("Loading MobileNetV2 pretrained weights...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    print("Model loaded. Params: ~3.4M")
    return model


def trace_model(model):
    print("Tracing model with dummy input (1, 3, 224, 224)...")
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    print("Trace complete.")
    return traced


def convert_to_coreml(traced_model, compute_units):
    print(f"Converting to CoreML [compute_units={compute_units}]...")

    # ImageNet normalization baked INTO the model
    # scale = 1/(std * 255), bias = -mean/std
    scale = 1.0 / (0.226 * 255.0)
    bias = [-0.485 / 0.226, -0.456 / 0.226, -0.406 / 0.226]

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input",
            shape=(1, 3, 224, 224),
            scale=scale,
            bias=bias,
            color_layout=ct.colorlayout.RGB,
        )],
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS13,
    )
    print("Conversion complete.")
    return mlmodel


def main():
    parser = argparse.ArgumentParser(description="Convert MobileNetV2 to CoreML")
    parser.add_argument("--output", default="models/pretrained/mobilenetv2.mlpackage")
    parser.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    args = parser.parse_args()

    unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model = load_mobilenetv2()
    traced = trace_model(model)
    mlmodel = convert_to_coreml(traced, unit_map[args.compute_units])

    print(f"Saving to {args.output} ...")
    mlmodel.save(args.output)
    print(f"Done. File saved: {args.output}")


if __name__ == "__main__":
    main()