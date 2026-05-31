"""
Phase 4 — Real-time activation heatmap renderer.
Receives tensors from C++ engine via ZMQ, renders live as heatmap grid.

Usage:
    python visualizer/activation_renderer.py --layer 1
    python visualizer/activation_renderer.py --layer 18 --max-channels 64
"""
import argparse
import json
import math
import threading
import time
from collections import deque
from typing import Optional

import matplotlib
matplotlib.use("MacOSX")   # native macOS backend — required for interactivity
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import zmq

# ── Layer metadata ────────────────────────────────────────

LAYER_INFO = {
    1:  {"channels": 16,   "shape": (16, 112, 112),  "label": "Early edges/textures"},
    4:  {"channels": 32,   "shape": (32, 28, 28),    "label": "Corners/patterns"},
    11: {"channels": 96,   "shape": (96, 14, 14),    "label": "Semantic regions"},
    18: {"channels": 1280, "shape": (1280, 7, 7),    "label": "Abstract concepts"},
}


# ── Thread-safe buffer — stores latest activation array ──

class ActivationBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._data  = None
        self._meta  = {}
        self._count = 0

    def put(self, arr: np.ndarray, meta: dict):
        with self._lock:
            self._data  = arr
            self._meta  = meta
            self._count += 1

    def get(self):
        with self._lock:
            return self._data, self._meta.copy(), self._count


# ── ZMQ receiver thread ───────────────────────────────────

class ZMQReceiver(threading.Thread):
    def __init__(self, buffer: ActivationBuffer, layer_name: str,
                 host: str = "localhost", port: int = 5555):
        super().__init__(daemon=True)
        self.buffer     = buffer
        self.layer_name = layer_name
        self.host       = host
        self.port       = port
        self.running    = True
        self.fps_times: deque = deque(maxlen=30)

    def run(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        print(f"[Receiver] Listening for '{self.layer_name}' on :{self.port}")

        while self.running:
            try:
                hdr_bytes = sock.recv()
                dat_bytes = sock.recv()
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[Receiver] Error: {e}")
                break

            hdr  = json.loads(hdr_bytes.decode())
            name = hdr["name"]

            # Only process the layer we care about
            if name != self.layer_name:
                continue

            shape = hdr["shape"]
            arr   = np.frombuffer(dat_bytes, dtype=np.float32).reshape(shape)
            arr   = arr.squeeze(0)  # (1,C,H,W) → (C,H,W)

            self.fps_times.append(time.time())
            meta = {
                "frame":        hdr.get("frame", 0),
                "inference_ms": hdr.get("inference_ms", 0.0),
                "fps":          self._compute_fps(),
            }
            self.buffer.put(arr, meta)

        sock.close()
        ctx.term()

    def _compute_fps(self) -> float:
        t = self.fps_times
        if len(t) < 2:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0] + 1e-9)

    def stop(self):
        self.running = False


# ── Normalization + colormap ──────────────────────────────

def normalize_channel(ch: np.ndarray) -> np.ndarray:
    """Per-channel min-max normalization → [0, 1]."""
    lo, hi = ch.min(), ch.max()
    if hi - lo < 1e-8:
        return np.zeros_like(ch)
    return (ch - lo) / (hi - lo)


def channel_to_rgb(ch: np.ndarray, cmap) -> np.ndarray:
    """Normalized channel → RGB uint8 via colormap."""
    rgba = cmap(normalize_channel(ch))   # (H,W,4) float
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def grid_dims(n_channels: int, max_channels: int) -> tuple:
    n  = min(n_channels, max_channels)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols, n


# ── Heatmap renderer ──────────────────────────────────────

class HeatmapRenderer:
    def __init__(self, buffer: ActivationBuffer, layer_idx: int,
                 max_channels: int = 64, cmap_name: str = "viridis"):
        self.buffer      = buffer
        self.layer_idx   = layer_idx
        self.layer_name  = f"activation_layer_{layer_idx}"
        self.info        = LAYER_INFO[layer_idx]
        self.max_ch      = max_channels
        self.cmap        = plt.cm.get_cmap(cmap_name)
        self._last_count = -1

        C = self.info["channels"]
        _, H, W = self.info["shape"]
        self.rows, self.cols, self.n_show = grid_dims(C, max_channels)

        # Build figure
        fig_w = self.cols * 1.5
        fig_h = self.rows * 1.5 + 0.8
        self.fig, self.axes = plt.subplots(
            self.rows, self.cols,
            figsize=(max(fig_w, 8), max(fig_h, 6))
        )
        self.fig.patch.set_facecolor("#111111")
        self.axes = np.array(self.axes).flatten()

        # Placeholder image (black)
        placeholder = np.zeros((H, W, 3), dtype=np.uint8)
        self.imgs = []
        for i, ax in enumerate(self.axes):
            ax.set_facecolor("#111111")
            ax.set_xticks([])
            ax.set_yticks([])
            if i < self.n_show:
                im = ax.imshow(placeholder, aspect="auto", interpolation="nearest")
                ax.set_title(f"ch {i}", fontsize=6, color="#888888", pad=1)
                self.imgs.append(im)
            else:
                ax.set_visible(False)
                self.imgs.append(None)

        title = (f"Layer {layer_idx}  |  {self.info['label']}  |  "
                 f"shape {self.info['shape']}  |  "
                 f"showing {self.n_show}/{C} channels")
        self.fig.suptitle(title, fontsize=9, color="#cccccc",
                          fontfamily="monospace")
        self.status_text = self.fig.text(
            0.5, 0.01, "Waiting for engine...",
            ha="center", fontsize=8, color="#666666",
            fontfamily="monospace"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(self, _frame):
        arr, meta, count = self.buffer.get()

        if arr is None or count == self._last_count:
            return self.imgs  # nothing new

        self._last_count = count

        # Render each channel
        C = arr.shape[0]
        for i in range(self.n_show):
            if i >= C or self.imgs[i] is None:
                continue
            rgb = channel_to_rgb(arr[i], self.cmap)
            self.imgs[i].set_data(rgb)

        # Update status
        status = (f"frame {meta.get('frame', 0)}  |  "
                  f"inf {meta.get('inference_ms', 0):.1f}ms  |  "
                  f"recv {meta.get('fps', 0):.1f} fps")
        self.status_text.set_text(status)
        return self.imgs

    def run(self, interval_ms: int = 50):
        """Start animation — blocks main thread (required by matplotlib)."""
        ani = animation.FuncAnimation(
            self.fig, self.update,
            interval=interval_ms,
            blit=False,       # blit=True breaks suptitle update on macOS
            cache_frame_data=False,
        )
        plt.show()
        return ani


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activation heatmap renderer")
    parser.add_argument("--layer",        type=int,  default=1,
                        choices=[1, 4, 11, 18])
    parser.add_argument("--max-channels", type=int,  default=64)
    parser.add_argument("--host",         default="localhost")
    parser.add_argument("--port",         type=int,  default=5555)
    parser.add_argument("--cmap",         default="viridis",
                        choices=["viridis", "hot", "plasma", "inferno"])
    parser.add_argument("--interval",     type=int,  default=50,
                        help="Animation update interval ms")
    args = parser.parse_args()

    if args.layer not in LAYER_INFO:
        print(f"Invalid layer. Choose from: {list(LAYER_INFO.keys())}")
        return

    print(f"[Renderer] Layer {args.layer} | "
          f"max {args.max_channels} channels | cmap={args.cmap}")
    print("[Renderer] Start engine first: ./engine/build/engine")
    print("[Renderer] Close window or Ctrl+C to stop.\n")

    buf      = ActivationBuffer()
    receiver = ZMQReceiver(
        buf,
        layer_name=f"activation_layer_{args.layer}",
        host=args.host,
        port=args.port,
    )
    receiver.start()

    renderer = HeatmapRenderer(
        buf,
        layer_idx=args.layer,
        max_channels=args.max_channels,
        cmap_name=args.cmap,
    )
    renderer.run(interval_ms=args.interval)  # blocks here

    receiver.stop()
    receiver.join(timeout=2)
    print("\n[Renderer] Stopped.")


if __name__ == "__main__":
    main()