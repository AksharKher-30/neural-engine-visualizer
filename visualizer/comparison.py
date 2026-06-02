"""
Phase 8 — Side-by-side dual-model comparison.
Runs two engine instances simultaneously, one per model.

Usage (3 terminals):
    Terminal 1: ./engine/build/engine models/pretrained/mobilenetv2_instrumented.mlpackage 5555
    Terminal 2: ./engine/build/engine models/pretrained/efficientnet_b0_instrumented.mlpackage 5556
    Terminal 3: python visualizer/comparison.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
import zmq

from visualizer.overlay import (
    load_imagenet_labels, softmax, compute_saliency,
    saliency_to_heatmap, alpha_blend, draw_predictions,
    COLORMAPS, COLORMAP_NAMES,
)
from models.model_configs import MODEL_CONFIGS

# ── Layout ────────────────────────────────────────────────

PANEL_W = 640
PANEL_H = 480
HUD_H   = 60
TOTAL_W = PANEL_W * 2
TOTAL_H = PANEL_H + HUD_H
DARK_BG = (20, 20, 30)
DARK_MID= (35, 35, 55)
TEXT_W  = (220, 220, 220)


# ── Thread-safe buffer ────────────────────────────────────

class TensorBuffer:
    def __init__(self):
        self._lock    = threading.Lock()
        self._tensors: Dict[str, np.ndarray] = {}
        self._meta    = {}

    def put(self, name, arr, meta):
        with self._lock:
            self._tensors[name] = arr
            self._meta = meta

    def get_all(self):
        with self._lock:
            return dict(self._tensors), dict(self._meta)


# ── ZMQ receiver ─────────────────────────────────────────

class ZMQReceiver(threading.Thread):
    def __init__(self, buf: TensorBuffer, port: int, label: str):
        super().__init__(daemon=True)
        self.buf     = buf
        self.port    = port
        self.label   = label
        self.running = True
        self._fps    = deque(maxlen=30)

    def run(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.connect(f"tcp://localhost:{self.port}")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        print(f"[{self.label}] ZMQ connected on :{self.port}")
        while self.running:
            try:
                hdr_b = sock.recv()
                dat_b = sock.recv()
            except zmq.Again:
                continue
            except Exception:
                break
            hdr = json.loads(hdr_b.decode())
            arr = (np.frombuffer(dat_b, dtype=np.float32)
                   .reshape(hdr["shape"]).squeeze(0))
            self._fps.append(time.time())
            meta = {
                "frame":        hdr.get("frame", 0),
                "inference_ms": hdr.get("inference_ms", 0.0),
                "fps": (len(self._fps)-1)/(self._fps[-1]-self._fps[0]+1e-9)
                        if len(self._fps) > 1 else 0.0,
            }
            self.buf.put(hdr["name"], arr, meta)
        sock.close(); ctx.term()

    def stop(self):
        self.running = False


# ── Panel renderer ────────────────────────────────────────

def render_model_panel(frame: np.ndarray,
                        tensors: Dict[str, np.ndarray],
                        meta: dict,
                        layer_key: str,
                        cmap_id: int,
                        alpha: float,
                        show_pred: bool,
                        labels,
                        model_label: str) -> np.ndarray:
    panel = cv2.resize(frame.copy(), (PANEL_W, PANEL_H))

    if layer_key in tensors:
        saliency = compute_saliency(tensors[layer_key])
        heatmap  = saliency_to_heatmap(saliency, PANEL_H, PANEL_W, cmap_id)
        panel    = alpha_blend(panel, heatmap, alpha)

    if show_pred and "logits" in tensors:
        panel = draw_predictions(panel, tensors["logits"], labels)

    # Model name badge
    inf_ms = meta.get("inference_ms", 0)
    badge  = f"{model_label}  {inf_ms:.1f}ms"
    cv2.rectangle(panel, (0, PANEL_H-26), (len(badge)*9+10, PANEL_H), (0,0,0), -1)
    cv2.putText(panel, badge, (5, PANEL_H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80,200,80), 1)
    return panel


def render_hud(state: dict, meta_l: dict, meta_r: dict,
               dfps: float, recording: bool) -> np.ndarray:
    bar = np.full((HUD_H, TOTAL_W, 3), DARK_MID, dtype=np.uint8)
    cv2.line(bar, (0, 0), (TOTAL_W, 0), (60, 60, 90), 1)
    rec = "  ● REC" if recording else ""
    line1 = (f"Layer: {state['layer_key']}  |  "
             f"alpha: {state['alpha']:.2f}  |  "
             f"cmap: {COLORMAP_NAMES[state['cmap_idx']]}  |  "
             f"L-fps: {meta_l.get('fps',0):.0f}  "
             f"R-fps: {meta_r.get('fps',0):.0f}  |  "
             f"display: {dfps:.0f}fps{rec}")
    line2 = ("m=toggle model | 1/4/b/e=layer | +/-=alpha | "
             "c=cmap | p=pred | r=record | s=screenshot | q=quit")
    col = (80, 80, 220) if recording else TEXT_W
    cv2.putText(bar, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    cv2.putText(bar, line2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (130, 130, 160), 1)
    return bar


# ── Main ──────────────────────────────────────────────────

def main():
    cfg_mv2 = MODEL_CONFIGS["mobilenetv2"]
    cfg_eff = MODEL_CONFIGS["efficientnet_b0"]

    labels = load_imagenet_labels()

    buf_mv2 = TensorBuffer()
    buf_eff = TensorBuffer()
    rx_mv2  = ZMQReceiver(buf_mv2, cfg_mv2["port"], "MobileNetV2")
    rx_eff  = ZMQReceiver(buf_eff, cfg_eff["port"], "EfficientNet-B0")
    rx_mv2.start()
    rx_eff.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PANEL_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_H)
    for _ in range(5):
        cap.read(None); time.sleep(0.03)

    os.makedirs("outputs", exist_ok=True)

    state = {
        "layer_key":  "activation_layer_1",
        "alpha":      0.4,
        "cmap_idx":   0,
        "show_pred":  True,
    }

    LAYER_KEYS = {
        ord("1"): "activation_layer_1",
        ord("4"): "activation_layer_4",
        ord("b"): "activation_layer_11",
        ord("e"): "activation_layer_18",
    }

    recording   = False
    writer: Optional[cv2.VideoWriter] = None
    screenshot_n = 0
    frame_times: deque = deque(maxlen=30)

    wname = "Neural Engine — Model Comparison"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, TOTAL_W, TOTAL_H)

    print("\n[Comparison] Running.")
    print("  Terminal 1: ./engine/build/engine "
          "models/pretrained/mobilenetv2_instrumented.mlpackage 5555")
    print("  Terminal 2: ./engine/build/engine "
          "models/pretrained/efficientnet_b0_instrumented.mlpackage 5556\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        t_mv2, m_mv2 = buf_mv2.get_all()
        t_eff, m_eff = buf_eff.get_all()
        cmap_id      = COLORMAPS[state["cmap_idx"]]

        left  = render_model_panel(
            frame, t_mv2, m_mv2, state["layer_key"],
            cmap_id, state["alpha"], state["show_pred"],
            labels, "MobileNetV2")

        right = render_model_panel(
            frame, t_eff, m_eff, state["layer_key"],
            cmap_id, state["alpha"], state["show_pred"],
            labels, "EfficientNet-B0")

        frame_times.append(time.time())
        dfps = ((len(frame_times)-1) /
                (frame_times[-1]-frame_times[0]+1e-9)
                if len(frame_times) > 1 else 0.0)

        div       = np.full((PANEL_H, 3, 3), (60,60,90), dtype=np.uint8)
        top       = np.hstack([left, div, right[:, :PANEL_W-3]])
        hud       = render_hud(state, m_mv2, m_eff, dfps, recording)
        composite = np.vstack([top, hud])

        cv2.imshow(wname, composite)
        if recording and writer:
            writer.write(composite)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"):
            break
        elif key in LAYER_KEYS:
            state["layer_key"] = LAYER_KEYS[key]
        elif key in (ord("+"), ord("=")):
            state["alpha"] = min(0.95, state["alpha"] + 0.05)
        elif key == ord("-"):
            state["alpha"] = max(0.05, state["alpha"] - 0.05)
        elif key == ord("c"):
            state["cmap_idx"] = (state["cmap_idx"]+1) % len(COLORMAPS)
        elif key == ord("p"):
            state["show_pred"] = not state["show_pred"]
        elif key == ord("r"):
            if not recording:
                ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                path   = f"outputs/comparison_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, 20, (TOTAL_W, TOTAL_H))
                recording = True
                print(f"[Comparison] Recording → {path}")
            else:
                recording = False
                if writer:
                    writer.release(); writer = None
                print("[Comparison] Recording stopped.")
        elif key == ord("s"):
            path = f"outputs/comparison_{screenshot_n:04d}.png"
            cv2.imwrite(path, composite)
            print(f"[Comparison] Screenshot → {path}")
            screenshot_n += 1

    if recording and writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    rx_mv2.stop(); rx_eff.stop()
    rx_mv2.join(timeout=2); rx_eff.join(timeout=2)
    print("[Comparison] Stopped.")


if __name__ == "__main__":
    main()