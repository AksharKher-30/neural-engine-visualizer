"""
Phase 3 — ZMQ subscriber: receives activation tensors from C++ engine.
Verifies data flowing correctly, prints per-frame stats.

Usage:
    python transport/zmq_bridge.py            # infinite
    python transport/zmq_bridge.py --frames 5 # stop after 5 frames
"""
import argparse
import json
import numpy as np
import zmq


def print_frame(tensors: dict, frame: int) -> None:
    print(f"\n── Frame {frame} {'─'*40}")
    for name, info in sorted(tensors.items()):
        a = info["arr"]
        print(f"  {name:<35} shape={info['shape']}")
        print(f"    min={a.min():.4f}  max={a.max():.4f}  "
              f"mean={a.mean():.4f}  nonzero={np.count_nonzero(a)}")
    if tensors:
        ms = next(iter(tensors.values()))["ms"]
        print(f"  inference_ms: {ms:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   default=5555, type=int)
    parser.add_argument("--frames", default=-1,   type=int)
    args = parser.parse_args()

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 8000)
    print(f"[Bridge] Connected to port {args.port}. Waiting...\n")

    current_frame  = -1
    frame_tensors  = {}
    frames_done    = 0

    try:
        while args.frames < 0 or frames_done < args.frames:
            try:
                hdr_bytes = sock.recv()
                dat_bytes = sock.recv()
            except zmq.Again:
                print("[Bridge] Timeout — is engine running?")
                break

            hdr   = json.loads(hdr_bytes.decode())
            name  = hdr["name"]
            shape = hdr["shape"]
            frame = hdr["frame"]
            ms    = hdr.get("inference_ms", 0.0)

            arr = np.frombuffer(dat_bytes, dtype=np.float32).reshape(shape)
            frame_tensors[name] = {"arr": arr, "shape": shape, "ms": ms}

            # Print when frame number changes and we have all 5 tensors
            if frame != current_frame and current_frame >= 0:
                if len(frame_tensors) >= 5:
                    print_frame(frame_tensors, current_frame)
                    frame_tensors = {}
                    frames_done += 1

            current_frame = frame

    except KeyboardInterrupt:
        print("\n[Bridge] Stopped.")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()