"""
Phase 1 — Verify PyTorch vs CoreML prediction parity
Usage: python models/verify.py --image data/test_images/dog.jpg
"""
import argparse
import os
import numpy as np
import torch
import coremltools as ct
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

MLPACKAGE = "models/pretrained/mobilenetv2.mlpackage"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def pytorch_top5(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top5 = torch.topk(probs, 5)
    return top5.indices.numpy().tolist(), top5.values.numpy().tolist()


def coreml_top5(mlpackage_path, image_path):
    mlmodel = ct.models.MLModel(mlpackage_path)
    # CoreML expects PIL Image — normalization is baked in via ImageType
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    out = mlmodel.predict({"input": img})
    logits = list(out.values())[0].flatten()
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top5_idx = np.argsort(probs)[-5:][::-1].tolist()
    top5_val = probs[top5_idx].tolist()
    return top5_idx, top5_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mlpackage", default=MLPACKAGE)
    args = parser.parse_args()

    assert os.path.exists(args.image), f"Image not found: {args.image}"
    assert os.path.exists(args.mlpackage), \
        f"MLPackage not found: {args.mlpackage}\nRun: python models/convert.py first"

    print(f"Image: {args.image}")

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    print("\nPyTorch inference...")
    pt_idx, pt_vals = pytorch_top5(model, args.image)
    print(f"  Top-5 indices : {pt_idx}")
    print(f"  Top-5 probs   : {[f'{v:.4f}' for v in pt_vals]}")

    print("\nCoreML inference...")
    cm_idx, cm_vals = coreml_top5(args.mlpackage, args.image)
    print(f"  Top-5 indices : {cm_idx}")
    print(f"  Top-5 probs   : {[f'{v:.4f}' for v in cm_vals]}")

    overlap = len(set(pt_idx) & set(cm_idx))
    top1_match = pt_idx[0] == cm_idx[0]
    print(f"\nTop-5 overlap : {overlap}/5")
    print(f"Top-1 match   : {'✓' if top1_match else '✗'} (PT={pt_idx[0]}, CM={cm_idx[0]})")

    # Pass if: top-1 matches AND overlap >= 2, OR overlap >= 3 regardless
    passed = (top1_match and overlap >= 2) or overlap >= 3

    if passed:
        print("✓ PASS — parity confirmed")
    else:
        print("✗ FAIL — predictions diverge too much")
        exit(1)

    print(f"\nTop-5 overlap: {overlap}/5")

    if overlap >= 3:
        print("✓ PASS — parity confirmed (≥3 overlap)")
    else:
        print("✗ FAIL — predictions diverge too much")
        exit(1)


if __name__ == "__main__":
    main()