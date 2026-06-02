"""
Centralized metadata for all supported models.
"""

MODEL_CONFIGS = {
    "mobilenetv2": {
        "pkg_base":     "models/pretrained/mobilenetv2.mlpackage",
        "pkg_instr":    "models/pretrained/mobilenetv2_instrumented.mlpackage",
        "target_layers": [1, 4, 11, 18],
        "layer_shapes": {
            1:  (16,   112, 112),
            4:  (32,   28,  28),
            11: (96,   14,  14),
            18: (1280, 7,   7),
        },
        "label":  "MobileNetV2",
        "params": "3.4M",
        "port":   5555,
    },
    "efficientnet_b0": {
        "pkg_base":     "models/pretrained/efficientnet_b0.mlpackage",
        "pkg_instr":    "models/pretrained/efficientnet_b0_instrumented.mlpackage",
        "target_layers": [1, 3, 5, 8],
        "layer_shapes": {
            1: (16,   112, 112),
            3: (40,   28,  28),
            5: (112,  14,  14),
            8: (1280, 7,   7),
        },
        "label":  "EfficientNet-B0",
        "params": "5.3M",
        "port":   5556,
    },
}