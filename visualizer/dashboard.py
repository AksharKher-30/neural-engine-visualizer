"""
Phase 7 — Unified interactive dashboard.
Single OpenCV window: webcam+overlay (left) | activation grid (right) | HUD (bottom).

Usage:
    python visualizer/dashboard.py
    python visualizer/dashboard.py --layer 1 --alpha 0.4 --max-channels 64
"""
import argparse
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
import zmq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer.overlay import (
    load_imagenet_labels, softmax,
    compute_saliency, saliency_to_heatmap, alpha_blend,
    draw_predictions, COLORMAPS, COLORMAP_NAMES, LAYER_NAMES,
)

# ── Layout constants ──────────────────────────────────────

PANEL_W  = 640
PANEL_H  = 480
HUD_H    = 60
TOTAL_W  = PANEL_W * 2
TOTAL_H  = PANEL_H + HUD_H
DIVIDER  = 3   # px separator between panels

LAYER_MAP = {ord("1"): 1, ord("4"): 4, ord("b"): 11, ord("e"): 18}
LAYER_MAX_CHANNELS = {1: 16, 4: 32, 11: 64, 18: 64}

DARK_BG  = (20, 20, 30)
DARK_MID = (35, 35, 55)
TEXT_W   = (220, 220, 220)
TEXT_G   = (80, 200, 80)
TEXT_R   = (80, 80, 220)


# ── Thread-safe tensor buffer ─────────────────────────────

class TensorBuffer:
    def __init__(self):
        self._lock    = threading.Lock()
        self._tensors: Dict[str, np.ndarray] = {}
        self._meta    = {}

    def put(self, name: str, arr: np.ndarray, meta: dict):
        with self._lock:
            self._tensors[name] = arr
            self._meta = meta

    def get_all(self):
        with self._lock:
            return dict(self._tensors), dict(self._meta)


# ── ZMQ receiver thread ───────────────────────────────────

class ZMQReceiver(threading.Thread):
    def __init__(self, buffer: TensorBuffer,
                 host: str = "localhost", port: int = 5555):
        super().__init__(daemon=True)
        self.buffer  = buffer
        self.host    = host
        self.port    = port
        self.running = True
        self._fps    = deque(maxlen=30)

    def run(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        while self.running:
            try:
                hdr_b = sock.recv()
                dat_b = sock.recv()
            except zmq.Again:
                continue
            except Exception:
                break
            hdr  = json.loads(hdr_b.decode())
            arr  = (np.frombuffer(dat_b, dtype=np.float32)
                    .reshape(hdr["shape"]).squeeze(0))
            self._fps.append(time.time())
            meta = {"frame": hdr.get("frame", 0),
                    "inference_ms": hdr.get("inference_ms", 0.0),
                    "zmq_fps": self._zmq_fps()}
            self.buffer.put(hdr["name"], arr, meta)
        sock.close(); ctx.term()

    def _zmq_fps(self) -> float:
        t = self._fps
        return 0.0 if len(t) < 2 else (len(t)-1)/(t[-1]-t[0]+1e-9)

    def stop(self):
        self.running = False


# ── Activation grid renderer (pure OpenCV) ───────────────

def render_activation_grid(arr: np.ndarray,
                            panel_w: int, panel_h: int,
                            cmap_id: int,
                            max_ch: int = 64,
                            layer_idx: int = 1) -> np.ndarray:
    """(C,H,W) → BGR grid image (panel_h, panel_w, 3)."""
    C = arr.shape[0]
    n = min(C, max_ch)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Reserve top bar for layer label
    label_h  = 22
    grid_h   = panel_h - label_h
    tile_w   = panel_w // cols
    tile_h   = grid_h  // rows

    canvas = np.full((panel_h, panel_w, 3), DARK_BG, dtype=np.uint8)

    for i in range(n):
        ch = arr[i]
        lo, hi = ch.min(), ch.max()
        if hi - lo > 1e-8:
            ch_u8 = ((ch - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            ch_u8 = np.zeros_like(ch, dtype=np.uint8)
        tile = cv2.resize(ch_u8, (tile_w, tile_h),
                          interpolation=cv2.INTER_LINEAR)
        tile_bgr = cv2.applyColorMap(tile, cmap_id)
        r, c = divmod(i, cols)
        y0 = label_h + r * tile_h
        x0 = c * tile_w
        canvas[y0:y0+tile_h, x0:x0+tile_w] = tile_bgr
        # channel label
        cv2.putText(canvas, str(i), (x0+2, y0+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1)

    # Layer label bar
    shape_str = {1:"(16,112,112)", 4:"(32,28,28)",
                 11:"(96,14,14)",  18:"(1280,7,7)"}.get(layer_idx, "")
    label = (f"Layer {layer_idx}: {LAYER_NAMES.get(layer_idx,'')}  "
             f"{shape_str}  [{n}/{C} ch]")
    cv2.rectangle(canvas, (0, 0), (panel_w, label_h), DARK_MID, -1)
    cv2.putText(canvas, label, (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_W, 1)
    return canvas


# ── HUD bar ───────────────────────────────────────────────

def render_hud(total_w: int, state: dict, meta: dict,
               display_fps: float, recording: bool) -> np.ndarray:
    bar = np.full((HUD_H, total_w, 3), DARK_MID, dtype=np.uint8)
    cv2.line(bar, (0, 0), (total_w, 0), (60, 60, 90), 1)

    rec_str = "  ● REC" if recording else ""
    line1 = (f"Layer: {LAYER_NAMES.get(state['layer'], state['layer'])}  |  "
             f"alpha: {state['alpha']:.2f}  |  "
             f"cmap: {COLORMAP_NAMES[state['cmap_idx']]}  |  "
             f"inf: {meta.get('inference_ms',0):.1f}ms  |  "
             f"zmq: {meta.get('zmq_fps',0):.0f}fps  |  "
             f"display: {display_fps:.0f}fps"
             f"{rec_str}")
    line2 = ("Keys:  1/4/b/e=layer  |  +/-=alpha  |  "
             "c=colormap  |  p=predictions  |  r=record  |  s=screenshot  |  q=quit")

    col1 = TEXT_R if recording else TEXT_W
    cv2.putText(bar, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col1, 1)
    cv2.putText(bar, line2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (130, 130, 160), 1)
    return bar


# ── Waiting screen ────────────────────────────────────────

def render_waiting(panel_w: int, panel_h: int, msg: str) -> np.ndarray:
    canvas = np.full((panel_h, panel_w, 3), DARK_BG, dtype=np.uint8)
    cv2.putText(canvas, msg,
                (panel_w//2 - len(msg)*5, panel_h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 120), 1)
    return canvas


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified dashboard")
    parser.add_argument("--layer",        type=int,   default=1,
                        choices=[1, 4, 11, 18])
    parser.add_argument("--alpha",        type=float, default=0.4)
    parser.add_argument("--cmap",         default="jet",
                        choices=COLORMAP_NAMES)
    parser.add_argument("--max-channels", type=int,   default=64)
    parser.add_argument("--host",         default="localhost")
    parser.add_argument("--port",         type=int,   default=5555)
    parser.add_argument("--cam",          type=int,   default=0)
    args = parser.parse_args()

    labels   = load_imagenet_labels()
    buf      = TensorBuffer()
    receiver = ZMQReceiver(buf, host=args.host, port=args.port)
    receiver.start()

    # Webcam
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PANEL_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_H)
    if not cap.isOpened():
        print("[Dashboard] Webcam failed."); return

    for _ in range(5):    # warmup
        cap.read(None)
        time.sleep(0.03)

    os.makedirs("outputs", exist_ok=True)

    state = {
        "layer":     args.layer,
        "alpha":     args.alpha,
        "cmap_idx":  COLORMAP_NAMES.index(args.cmap),
        "show_pred": True,
    }

    recording      = False
    writer: Optional[cv2.VideoWriter] = None
    screenshot_n   = 0
    frame_times: deque = deque(maxlen=30)

    window_name = "Neural Engine Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, TOTAL_W, TOTAL_H)

    print("\n[Dashboard] Running.")
    print("  Start engine: ./engine/build/engine")
    print("  Keys: 1/4/b/e=layer | +/-=alpha | c=cmap | r=record | s=screenshot | q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.resize(frame, (PANEL_W, PANEL_H))

        tensors, meta = buf.get_all()
        layer_key     = f"activation_layer_{state['layer']}"
        cmap_id       = COLORMAPS[state["cmap_idx"]]
        max_ch        = LAYER_MAX_CHANNELS.get(state["layer"], args.max_channels)

        # ── Left panel: webcam + overlay ─────────────────
        left = frame.copy()
        if layer_key in tensors:
            saliency = compute_saliency(tensors[layer_key])
            heatmap  = saliency_to_heatmap(saliency, PANEL_H, PANEL_W, cmap_id)
            left     = alpha_blend(left, heatmap, state["alpha"])
        if state["show_pred"] and "logits" in tensors:
            left = draw_predictions(left, tensors["logits"], labels)
        if recording:
            cv2.circle(left, (PANEL_W-20, 18), 8, (0, 0, 220), -1)
            cv2.putText(left, "REC", (PANEL_W-45, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,220), 1)

        # ── Right panel: activation grid ─────────────────
        if layer_key in tensors:
            right = render_activation_grid(
                tensors[layer_key], PANEL_W, PANEL_H,
                cmap_id, max_ch, state["layer"]
            )
        else:
            right = render_waiting(PANEL_W, PANEL_H, "Waiting for engine...")

        # ── Divider ───────────────────────────────────────
        div = np.full((PANEL_H, DIVIDER, 3), (60, 60, 90), dtype=np.uint8)

        # ── HUD ───────────────────────────────────────────
        frame_times.append(time.time())
        dfps = 0.0
        if len(frame_times) > 1:
            dfps = (len(frame_times)-1) / (frame_times[-1]-frame_times[0]+1e-9)
        hud = render_hud(TOTAL_W, state, meta, dfps, recording)

        # ── Composite ─────────────────────────────────────
        top = np.hstack([left, div, right[:, :PANEL_W - DIVIDER]])
        composite = np.vstack([top, hud])

        cv2.imshow(window_name, composite)

        if recording and writer is not None:
            writer.write(composite)

        # ── Keyboard ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in LAYER_MAP:
            state["layer"] = LAYER_MAP[key]
            print(f"[Dashboard] Layer → {state['layer']}")
        elif key in (ord("+"), ord("=")):
            state["alpha"] = min(0.95, state["alpha"] + 0.05)
        elif key == ord("-"):
            state["alpha"] = max(0.05, state["alpha"] - 0.05)
        elif key == ord("c"):
            state["cmap_idx"] = (state["cmap_idx"] + 1) % len(COLORMAPS)
            print(f"[Dashboard] Colormap → {COLORMAP_NAMES[state['cmap_idx']]}")
        elif key == ord("p"):
            state["show_pred"] = not state["show_pred"]
        elif key == ord("r"):
            if not recording:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"outputs/demo_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, 20,
                                         (TOTAL_W, TOTAL_H))
                recording = True
                print(f"[Dashboard] Recording → {path}")
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                print("[Dashboard] Recording stopped.")
        elif key == ord("s"):
            path = f"outputs/screenshot_{screenshot_n:04d}.png"
            cv2.imwrite(path, composite)
            print(f"[Dashboard] Screenshot → {path}")
            screenshot_n += 1

    # cleanup
    if recording and writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    receiver.stop()
    receiver.join(timeout=2)
    print("[Dashboard] Stopped.")


if __name__ == "__main__":
    main()