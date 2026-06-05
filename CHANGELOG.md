# Changelog

## v1.0.0 — 2026-06-03

### Phase 1 — PyTorch → CoreML conversion pipeline
- Loaded MobileNetV2 pretrained weights via torchvision
- Traced model with torch.jit.trace, converted to CoreML .mlpackage via coremltools
- Baked ImageNet normalization into model via ct.ImageType
- Verified PyTorch ↔ CoreML prediction parity (≥3/5 top-5 overlap)

### Phase 2 — Model surgery
- Built InstrumentedMobileNetV2 wrapper returning logits + 4 activation tensors in one pass
- Target layers: 1 (edges), 4 (patterns), 11 (semantic), 18 (abstract)
- Forced float32 outputs to prevent ANE float16 heap corruption
- Integration test: instrumented logits top-1 must match base model

### Phase 3 — C++ Objective-C++ inference engine
- CoreML inference via Objective-C++ (.mm) bridge — ObjCEngine class
- OpenCV webcam capture loop in C++ at 640×480
- CVPixelBuffer for image input (handles normalization internally)
- ZMQ PUSH socket streams activation tensors to Python at 170+ fps
- Fixed float16 dtype mismatch via explicit dataType check

### Phase 4 — Activation heatmap renderer
- ZMQ PULL receiver thread + thread-safe ActivationBuffer
- Per-channel normalization → viridis colormap → NxM grid
- Matplotlib FuncAnimation at ~15fps, layer selector via CLI

### Phase 5 — Live webcam overlay
- Saliency map: abs mean across channels → upsample → cv2.applyColorMap
- Alpha blend on live webcam frame via cv2.addWeighted
- Top-3 ImageNet prediction overlay with softmax probabilities
- Interactive keyboard controls (layer, alpha, colormap, predictions, screenshot)

### Phase 6 — Hardware profiler
- Benchmarked MobileNetV2 across ALL / CPU+NE / CPU+GPU / CPU_ONLY
- 100-run latency stats: mean, std, p50, p95, throughput
- 4-panel matplotlib dashboard saved to benchmarks/plots/
- CSV export to benchmarks/results/benchmark_m3.csv
- Optional Swift MLComputePlan script for per-op ANE/CPU/GPU routing

### Phase 7 — Unified dashboard
- Single OpenCV window: webcam+overlay (left) | activation grid (right) | HUD (bottom)
- Pure OpenCV grid rendering (no matplotlib) — full-speed tiling
- Video recording via cv2.VideoWriter → outputs/demo_*.mp4
- All keyboard controls unified: layer, alpha, colormap, predictions, record, screenshot

### Phase 8 — Multi-model support
- Added EfficientNet-B0 (5.3M params, target layers 1/3/5/8)
- Centralized model metadata in models/model_configs.py
- Surgeon.py now supports both models via --model arg
- C++ engine accepts port as argv[2] — runs two instances simultaneously
- comparison.py: side-by-side dual-model view, same webcam frame, both models live