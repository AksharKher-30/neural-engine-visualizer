"""
Phase 5 — Live activation overlay on webcam feed.
Blends saliency map directly onto webcam frame + shows top-3 predictions.

Usage:
    python visualizer/overlay.py
    python visualizer/overlay.py --layer 11 --alpha 0.5 --cmap hot

Keyboard controls (window must be focused):
    q → quit | 1/4/b/e → switch layer | +/- → alpha | c → colormap | s → screenshot
"""
import argparse
import json
import os
import threading
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np
import zmq

# ── Constants ─────────────────────────────────────────────

LAYER_MAP = {
    ord("1"): 1,
    ord("4"): 4,
    ord("b"): 11,
    ord("e"): 18,
}

COLORMAPS = [
    cv2.COLORMAP_JET,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_COOL,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_PLASMA,
]
COLORMAP_NAMES = ["jet", "hot", "cool", "viridis", "plasma"]

LAYER_NAMES = {1: "L1 edges", 4: "L4 patterns", 11: "L11 semantic", 18: "L18 abstract"}


# ── ImageNet labels ───────────────────────────────────────

def load_imagenet_labels(path: str = "data/imagenet_labels.txt"):
    if not os.path.exists(path):
        print(f"[Overlay] Labels not found: {path}")
        return [str(i) for i in range(1000)]
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


# ── Thread-safe multi-tensor buffer ──────────────────────

class TensorBuffer:
    """Stores the latest full frame (all tensors) from ZMQ."""
    def __init__(self):
        self._lock    = threading.Lock()
        self._tensors: Dict[str, np.ndarray] = {}
        self._meta    = {}
        self._updated = False

    def put(self, name: str, arr: np.ndarray, meta: dict):
        with self._lock:
            self._tensors[name] = arr
            self._meta = meta
            self._updated = True

    def get_all(self):
        with self._lock:
            self._updated = False
            return dict(self._tensors), dict(self._meta)

    def has_update(self) -> bool:
        with self._lock:
            return self._updated


# ── ZMQ receiver thread ───────────────────────────────────

class ZMQReceiver(threading.Thread):
    def __init__(self, buffer: TensorBuffer,
                 host: str = "localhost", port: int = 5555):
        super().__init__(daemon=True)
        self.buffer  = buffer
        self.host    = host
        self.port    = port
        self.running = True
        self._fps_times: deque = deque(maxlen=30)

    def run(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        print(f"[ZMQ] Connected to :{self.port}")

        while self.running:
            try:
                hdr_b = sock.recv()
                dat_b = sock.recv()
            except zmq.Again:
                continue
            except Exception as e:
                if self.running:
                    print(f"[ZMQ] Error: {e}")
                break

            hdr   = json.loads(hdr_b.decode())
            name  = hdr["name"]
            shape = hdr["shape"]
            arr   = (np.frombuffer(dat_b, dtype=np.float32)
                     .reshape(shape).squeeze(0))

            self._fps_times.append(time.time())
            meta = {
                "frame":        hdr.get("frame", 0),
                "inference_ms": hdr.get("inference_ms", 0.0),
                "zmq_fps":      self._zmq_fps(),
            }
            self.buffer.put(name, arr, meta)

        sock.close()
        ctx.term()

    def _zmq_fps(self) -> float:
        t = self._fps_times
        if len(t) < 2:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0] + 1e-9)

    def stop(self):
        self.running = False


# ── Saliency computation ──────────────────────────────────

def compute_saliency(arr: np.ndarray) -> np.ndarray:
    """(C,H,W) → normalized (H,W) saliency. Uses abs mean — works at all depths."""
    # abs mean: works for both shallow (signed) and deep (sparse) layers
    s = np.abs(arr).mean(axis=0)          # ← was arr.mean + ReLU
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-8:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)                  # normalize [0, 1]


def saliency_to_heatmap(saliency: np.ndarray,
                         target_h: int, target_w: int,
                         colormap_id: int) -> np.ndarray:
    """Normalized saliency → resized BGR heatmap for OpenCV blend."""
    s_uint8  = (saliency * 255).astype(np.uint8)
    resized  = cv2.resize(s_uint8, (target_w, target_h),
                          interpolation=cv2.INTER_LINEAR)
    heatmap  = cv2.applyColorMap(resized, colormap_id)   # (H,W,3) BGR
    return heatmap


def alpha_blend(frame: np.ndarray, heatmap: np.ndarray,
                alpha: float) -> np.ndarray:
    """Blend heatmap over frame: (1-alpha)*frame + alpha*heatmap."""
    return cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)


# ── Prediction overlay ────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def draw_predictions(frame: np.ndarray, logits: np.ndarray,
                     labels, n: int = 3) -> np.ndarray:
    probs   = softmax(logits.flatten())
    top_idx = np.argsort(probs)[-n:][::-1]
    out     = frame.copy()
    y0      = 30
    for rank, idx in enumerate(top_idx):
        label = labels[idx] if idx < len(labels) else str(idx)
        pct   = probs[idx] * 100
        text  = f"{rank+1}. {label[:22]:<22} {pct:5.1f}%"
        # shadow
        cv2.putText(out, text, (11, y0 + rank*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        # text
        color = (50, 255, 50) if rank == 0 else (180, 255, 180)
        cv2.putText(out, text, (10, y0 + rank*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return out


def draw_hud(frame: np.ndarray, layer: int, alpha: float,
             cmap_name: str, meta: dict, show_pred: bool) -> np.ndarray:
    """Bottom HUD bar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-36), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    hud = (f"Layer: {LAYER_NAMES.get(layer, layer)} | "
           f"alpha: {alpha:.2f} | cmap: {cmap_name} | "
           f"inf: {meta.get('inference_ms',0):.1f}ms | "
           f"zmq: {meta.get('zmq_fps',0):.0f}fps | "
           f"pred: {'ON' if show_pred else 'OFF'}")
    cv2.putText(frame, hud, (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    return frame


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",  type=int,   default=1,
                        choices=[1, 4, 11, 18])
    parser.add_argument("--alpha",  type=float, default=0.4)
    parser.add_argument("--cmap",   type=str,   default="jet",
                        choices=COLORMAP_NAMES)
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int,   default=5555)
    parser.add_argument("--width",  type=int,   default=640)
    parser.add_argument("--height", type=int,   default=480)
    args = parser.parse_args()

    labels      = load_imagenet_labels()
    buf         = TensorBuffer()
    receiver    = ZMQReceiver(buf, host=args.host, port=args.port)
    receiver.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[Overlay] Webcam failed."); return

    # Warmup
    for _ in range(5):
        cap.read(None)
        time.sleep(0.03)

    # Mutable state (changed by keyboard)
    state = {
        "layer":     args.layer,
        "alpha":     args.alpha,
        "cmap_idx":  COLORMAP_NAMES.index(args.cmap),
        "show_pred": True,
    }

    os.makedirs("outputs", exist_ok=True)
    screenshot_n = 0
    window_name  = "Neural Engine Activation Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    print("\n[Overlay] Running. Controls:")
    print("  q=quit | 1/4/b/e=layer | +/-=alpha | c=colormap | p=predictions | s=screenshot\n")

    frame_times: deque = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        tensors, meta = buf.get_all()
        layer_key = f"activation_layer_{state['layer']}"

        if layer_key in tensors:
            saliency = compute_saliency(tensors[layer_key])
            heatmap  = saliency_to_heatmap(
                saliency, h, w,
                COLORMAPS[state["cmap_idx"]]
            )
            frame = alpha_blend(frame, heatmap, state["alpha"])

        if state["show_pred"] and "logits" in tensors:
            frame = draw_predictions(frame, tensors["logits"], labels)

        frame = draw_hud(frame, state["layer"], state["alpha"],
                         COLORMAP_NAMES[state["cmap_idx"]], meta,
                         state["show_pred"])

        # Display FPS
        frame_times.append(time.time())
        if len(frame_times) > 1:
            fps = (len(frame_times)-1) / (frame_times[-1]-frame_times[0]+1e-9)
            cv2.putText(frame, f"display {fps:.0f}fps",
                        (w-130, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (120,120,120), 1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in LAYER_MAP:
            state["layer"] = LAYER_MAP[key]
            print(f"[Overlay] Layer → {state['layer']}")
        elif key == ord("+") or key == ord("="):
            state["alpha"] = min(0.95, state["alpha"] + 0.05)
        elif key == ord("-"):
            state["alpha"] = max(0.05, state["alpha"] - 0.05)
        elif key == ord("c"):
            state["cmap_idx"] = (state["cmap_idx"] + 1) % len(COLORMAPS)
            print(f"[Overlay] Colormap → {COLORMAP_NAMES[state['cmap_idx']]}")
        elif key == ord("p"):
            state["show_pred"] = not state["show_pred"]
        elif key == ord("s"):
            path = f"outputs/screenshot_{screenshot_n:04d}.png"
            cv2.imwrite(path, frame)
            print(f"[Overlay] Saved: {path}")
            screenshot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    receiver.stop()
    receiver.join(timeout=2)
    print("[Overlay] Stopped.")


if __name__ == "__main__":
    main()