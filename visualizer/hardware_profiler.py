"""
Phase 6 — Hardware profiler: benchmark ANE / CPU / Metal per layer.
Runs model across 4 CoreML compute unit configs, generates dashboard + CSV.

Usage:
    python visualizer/hardware_profiler.py
    python visualizer/hardware_profiler.py --runs 50 --warmup 5 --no-plot
"""
import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import coremltools as ct
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────

PKG_BASE  = "models/pretrained/mobilenetv2.mlpackage"
PKG_INSTR = "models/pretrained/mobilenetv2_instrumented.mlpackage"
CSV_OUT   = "benchmarks/results/benchmark_m3.csv"
PLOT_OUT  = "benchmarks/plots/latency_comparison.png"
SWIFT_BIN = "engine/compute_plan"

# ── Compute unit configs ───────────────────────────────────

CONFIGS = {
    "ALL (ANE+CPU+GPU)": ct.ComputeUnit.ALL,
    "CPU + NE":          ct.ComputeUnit.CPU_AND_NE,
    "CPU + GPU":         ct.ComputeUnit.CPU_AND_GPU,
    "CPU Only":          ct.ComputeUnit.CPU_ONLY,
}

COLORS = {
    "ALL (ANE+CPU+GPU)": "#4CAF50",
    "CPU + NE":          "#2196F3",
    "CPU + GPU":         "#FF9800",
    "CPU Only":          "#F44336",
}


# ── Data classes ──────────────────────────────────────────

@dataclass
class BenchStats:
    config:     str
    model:      str
    mean_ms:    float
    std_ms:     float
    min_ms:     float
    p50_ms:     float
    p95_ms:     float
    throughput: float      # inferences / second
    raw:        List[float] = field(default_factory=list, repr=False)


# ── Test image ────────────────────────────────────────────

def make_test_image(seed: int = 42) -> Image.Image:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── Benchmarker ───────────────────────────────────────────

class ModelBenchmarker:
    def __init__(self, n_warmup: int = 10, n_runs: int = 100):
        self.n_warmup = n_warmup
        self.n_runs   = n_runs
        self.img      = make_test_image()

    def benchmark(self, pkg_path: str, config_name: str,
                  compute_unit) -> BenchStats:
        model_label = (os.path.basename(pkg_path)
                       .replace(".mlpackage", "")
                       .replace("mobilenetv2", "MobileNetV2"))

        print(f"  Loading [{config_name}] {model_label} ...")
        model = ct.models.MLModel(pkg_path, compute_units=compute_unit)

        print(f"  Warming up ({self.n_warmup} runs) ...")
        for _ in range(self.n_warmup):
            model.predict({"input": self.img})

        print(f"  Benchmarking ({self.n_runs} runs) ...")
        times = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter()
            model.predict({"input": self.img})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)   # → ms

        raw = np.array(times)
        return BenchStats(
            config=config_name,
            model=model_label,
            mean_ms=float(raw.mean()),
            std_ms=float(raw.std()),
            min_ms=float(raw.min()),
            p50_ms=float(np.percentile(raw, 50)),
            p95_ms=float(np.percentile(raw, 95)),
            throughput=1000.0 / float(raw.mean()),
            raw=times,
        )

    def run_all(self, packages: Dict[str, str]) -> List[BenchStats]:
        results = []
        for pkg_label, pkg_path in packages.items():
            if not os.path.exists(pkg_path):
                print(f"  SKIP — not found: {pkg_path}")
                continue
            print(f"\n── {pkg_label} ──")
            for cfg_name, compute_unit in CONFIGS.items():
                stat = self.benchmark(pkg_path, cfg_name, compute_unit)
                stat.model = pkg_label
                print(f"    {cfg_name:<22} "
                      f"mean={stat.mean_ms:.2f}ms  "
                      f"p95={stat.p95_ms:.2f}ms  "
                      f"fps={stat.throughput:.1f}")
                results.append(stat)
        return results


# ── CSV export ────────────────────────────────────────────

def save_csv(results: List[BenchStats], path: str = CSV_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "config", "mean_ms", "std_ms",
            "min_ms", "p50_ms", "p95_ms", "throughput_fps"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model":          r.model,
                "config":         r.config,
                "mean_ms":        f"{r.mean_ms:.4f}",
                "std_ms":         f"{r.std_ms:.4f}",
                "min_ms":         f"{r.min_ms:.4f}",
                "p50_ms":         f"{r.p50_ms:.4f}",
                "p95_ms":         f"{r.p95_ms:.4f}",
                "throughput_fps": f"{r.throughput:.2f}",
            })
    print(f"\n[Profiler] CSV saved: {path}")


# ── Swift compute plan (optional) ─────────────────────────

def run_swift_compute_plan(pkg_path: str) -> Optional[dict]:
    """Run compiled Swift compute_plan binary → parse JSON."""
    if not os.path.exists(SWIFT_BIN):
        return None
    try:
        result = subprocess.run(
            [SWIFT_BIN, pkg_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception as e:
        print(f"[Profiler] Swift compute plan unavailable: {e}")
        return None


# ── Dashboard ─────────────────────────────────────────────

DARK_BG  = "#1a1a2e"
DARK_AX  = "#16213e"
TEXT_COL = "#e0e0e0"
GRID_COL = "#2a2a4a"


def setup_dark_ax(ax):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.7)


def plot_dashboard(results: List[BenchStats],
                   compute_plan: Optional[dict] = None,
                   save_path: str = PLOT_OUT):
    models  = list(dict.fromkeys(r.model for r in results))
    configs = list(CONFIGS.keys())
    n_cfg   = len(configs)
    n_mod   = len(models)

    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.4, wspace=0.35,
                            left=0.07, right=0.97,
                            top=0.90, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])   # latency bar
    ax2 = fig.add_subplot(gs[0, 1])   # throughput
    ax3 = fig.add_subplot(gs[0, 2])   # speedup
    ax4 = fig.add_subplot(gs[1, 0])   # p50 vs p95
    ax5 = fig.add_subplot(gs[1, 1])   # raw distribution (first model ALL)
    ax6 = fig.add_subplot(gs[1, 2])   # summary table

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        setup_dark_ax(ax)

    # ── Panel 1: Mean latency ─────────────────────────────
    x      = np.arange(n_cfg)
    width  = 0.35
    offset = np.linspace(-width/2*(n_mod-1), width/2*(n_mod-1), n_mod)

    for mi, model in enumerate(models):
        means = [next((r.mean_ms for r in results
                       if r.model == model and r.config == c), 0)
                 for c in configs]
        stds  = [next((r.std_ms for r in results
                       if r.model == model and r.config == c), 0)
                 for c in configs]
        bars = ax1.bar(x + offset[mi], means, width*0.85,
                       label=model, yerr=stds, capsize=3,
                       color=[COLORS[c] for c in configs],
                       alpha=0.85 if mi == 0 else 0.5,
                       error_kw={"ecolor": TEXT_COL, "elinewidth": 1})

    ax1.set_title("Mean Latency (ms) ± std", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=7)
    ax1.legend(fontsize=7, facecolor=DARK_AX, labelcolor=TEXT_COL)

    # ── Panel 2: Throughput ───────────────────────────────
    for mi, model in enumerate(models):
        tputs = [next((r.throughput for r in results
                       if r.model == model and r.config == c), 0)
                 for c in configs]
        ax2.barh([f"{c}\n({model[:8]})" for c in configs],
                 tputs,
                 color=[COLORS[c] for c in configs],
                 alpha=0.85 if mi == 0 else 0.55,
                 height=0.35, left=mi*5)

    ax2.set_title("Throughput (inferences/sec)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("fps")

    # ── Panel 3: Speedup over CPU Only ────────────────────
    for mi, model in enumerate(models):
        cpu_ms = next((r.mean_ms for r in results
                       if r.model == model and r.config == "CPU Only"), 1.0)
        speedups = [cpu_ms / max(next((r.mean_ms for r in results
                    if r.model == model and r.config == c), cpu_ms), 0.01)
                    for c in configs]
        ax3.bar(x + offset[mi], speedups, width*0.85,
                color=[COLORS[c] for c in configs],
                alpha=0.85 if mi == 0 else 0.55,
                label=model)

    ax3.axhline(1.0, color="#ffffff", linewidth=1, linestyle="--", alpha=0.5)
    ax3.set_title("Speedup over CPU Only", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Speedup (×)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=7)
    ax3.text(0.02, 0.97, "CPU baseline", transform=ax3.transAxes,
             color="#aaaaaa", fontsize=7, va="top")

    # ── Panel 4: p50 vs p95 (tail latency) ───────────────
    model0    = models[0]
    p50s = [next((r.p50_ms for r in results
                  if r.model == model0 and r.config == c), 0) for c in configs]
    p95s = [next((r.p95_ms for r in results
                  if r.model == model0 and r.config == c), 0) for c in configs]
    bw = 0.3
    ax4.bar(x - bw/2, p50s, bw, label="p50 (median)",
            color=[COLORS[c] for c in configs], alpha=0.9)
    ax4.bar(x + bw/2, p95s, bw, label="p95 (tail)",
            color=[COLORS[c] for c in configs], alpha=0.45, hatch="//")
    ax4.set_title("p50 vs p95 Latency (tail jitter)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Latency (ms)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=7)
    ax4.legend(fontsize=7, facecolor=DARK_AX, labelcolor=TEXT_COL)

    # ── Panel 5: Raw latency distribution (violin) ────────
    all_stat = next((r for r in results
                     if r.model == model0 and r.config == "ALL (ANE+CPU+GPU)"), None)
    cpu_stat = next((r for r in results
                     if r.model == model0 and r.config == "CPU Only"), None)
    if all_stat and cpu_stat and all_stat.raw and cpu_stat.raw:
        vp = ax5.violinplot([all_stat.raw, cpu_stat.raw],
                             showmedians=True, showextrema=True)
        for pc, col in zip(vp["bodies"], ["#4CAF50", "#F44336"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax5.set_xticks([1, 2])
        ax5.set_xticklabels(["ALL (ANE)", "CPU Only"], fontsize=9)
        ax5.set_title("Latency Distribution (100 runs)", fontsize=10, fontweight="bold")
        ax5.set_ylabel("Latency (ms)")
    else:
        ax5.text(0.5, 0.5, "No raw data", ha="center", va="center",
                 color=TEXT_COL, transform=ax5.transAxes)
        ax5.set_title("Latency Distribution", fontsize=10, fontweight="bold")

    # ── Panel 6: Summary table ────────────────────────────
    ax6.axis("off")
    best = min(results, key=lambda r: r.mean_ms)
    cpu  = next((r for r in results
                 if r.config == "CPU Only" and r.model == model0), None)

    lines = [
        "── PROFILER SUMMARY ──────────────────",
        f"  System   : Apple M3 (ANE + Metal GPU)",
        f"  Model    : MobileNetV2 (3.4M params)",
        f"  Runs     : {len(results[0].raw) if results[0].raw else 'N/A'} per config",
        "",
        f"  BEST CONFIG:  {best.config}",
        f"  Best mean:    {best.mean_ms:.2f} ms",
        f"  Best fps:     {best.throughput:.1f} inf/sec",
        "",
    ]
    if cpu:
        speedup = cpu.mean_ms / max(best.mean_ms, 0.01)
        lines += [
            f"  ANE speedup:  {speedup:.1f}× over CPU",
            f"  CPU mean:     {cpu.mean_ms:.2f} ms",
        ]
    lines += ["", "  Configs benchmarked:"]
    for r in [r for r in results if r.model == model0]:
        lines.append(f"    {r.config:<22} {r.mean_ms:6.2f}ms")

    for i, line in enumerate(lines):
        col = "#4CAF50" if "BEST" in line or "speedup" in line else TEXT_COL
        ax6.text(0.02, 0.97 - i * 0.057, line,
                 transform=ax6.transAxes, fontsize=7.5,
                 color=col, fontfamily="monospace", va="top")

    fig.suptitle(
        "Neural Engine Visualizer — Hardware Profiler  |  Apple M3",
        fontsize=13, color=TEXT_COL, fontweight="bold", y=0.96
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    print(f"[Profiler] Plot saved: {save_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hardware profiler")
    parser.add_argument("--runs",    type=int,  default=100)
    parser.add_argument("--warmup",  type=int,  default=10)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-instr",action="store_true",
                        help="Skip instrumented model (faster)")
    args = parser.parse_args()

    packages = {"Base model": PKG_BASE}
    if not args.no_instr:
        packages["Instrumented"] = PKG_INSTR

    print(f"\n[Profiler] Benchmarking {args.runs} runs × "
          f"{len(packages)} models × {len(CONFIGS)} configs")
    print(f"[Profiler] This will take ~"
          f"{args.runs * len(packages) * len(CONFIGS) * 5 // 1000 + 2} minutes.\n")

    benchmarker = ModelBenchmarker(n_warmup=args.warmup, n_runs=args.runs)
    results     = benchmarker.run_all(packages)

    save_csv(results)

    # Optional Swift compute plan
    plan_data = run_swift_compute_plan(PKG_BASE)
    if plan_data:
        ops     = plan_data.get("ops", [])
        ane_ct  = sum(1 for o in ops if o.get("device") == "ANE")
        cpu_ct  = sum(1 for o in ops if o.get("device") == "CPU")
        gpu_ct  = sum(1 for o in ops if o.get("device") == "GPU")
        total   = len(ops)
        print(f"\n[ComputePlan] {total} ops:  "
              f"ANE={ane_ct} ({100*ane_ct//max(total,1)}%)  "
              f"CPU={cpu_ct} ({100*cpu_ct//max(total,1)}%)  "
              f"GPU={gpu_ct} ({100*gpu_ct//max(total,1)}%)")

    if not args.no_plot:
        plot_dashboard(results, compute_plan=plan_data)

    print("\n[Profiler] Done.")


if __name__ == "__main__":
    main()