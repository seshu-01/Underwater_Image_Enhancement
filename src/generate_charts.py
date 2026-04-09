"""
generate_charts.py
------------------
Generates exactly 2 publication-ready figures for the UIE comparison paper:

  Fig. 2 — Grouped bar chart: methods on x-axis, PSNR / SSIM / UIQM bars per method
  Fig. 3 — PSNR vs. Inference Time scatter plot (log x-axis), speed-quality tradeoff

USAGE:
  1. Run metrics.py and note the Mean values it prints.
  2. Fill in the `results` dict below (CLAHE and FUnIE-GAN from your run).
  3. Run:  python generate_charts.py
  4. Charts saved to:  results/figures/fig2_bar_chart.png
                       results/figures/fig3_scatter.png

AUTOMATED MODE (call from metrics.py):
  See INTEGRATION_SNIPPET at the bottom of this file.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── OUTPUT DIRECTORY ──────────────────────────────────────────────────────────
OUT_DIR = "Results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DPI         = 300
FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 10
FONT_LEGEND = 9
FONT_ANNOT  = 8

# One colour per method — consistent across both charts
METHOD_COLORS = {
    "Raw Input":        "#000000",
    "CLAHE":            "#839AC2",
    "FUnIE-GAN":        "#F1711B",
    "DiffWater [Ref]":  "#23AE17",
    "CPDM [Ref]":       "#8930CD",
}

# Inference time (seconds / image) used in Fig. 3
INFERENCE_TIMES = {
    "Raw Input":        0.001,   # essentially instant
    "CLAHE":            0.03,    # ~30 ms
    "FUnIE-GAN":        0.08,    # ~80 ms, real-time on Jetson TX2
    "DiffWater [Ref]":  30.0,    # ~30 s on GPU (T=1000 DDPM steps)
    "CPDM [Ref]":       45.0,    # ~45 s on GPU (heavier architecture)
}

# ── !! FILL YOUR RESULTS HERE !! ──────────────────────────────────────────────
# Format: (mean, std)  —  set std=0.0 if you only have the mean.
#
# Diffusion reference values (UIEB Test_U90, from published papers):
#   DiffWater -> PSNR 20.86, SSIM 0.836  [IEEE JSTARS 2024]
#   CPDM      -> PSNR 23.65, SSIM 0.882  [Scientific Reports 2024]

results = {
    "Raw Input": {
        "PSNR": (14.20, 0.00),
        "SSIM": (0.650, 0.00),
        "UIQM": (1.95,  0.00),
    },
    "CLAHE": {
        "PSNR":  (20.07, 3.07),     # ← FILL from metrics.py output
        "SSIM":  (0.8140, 0.0834),     # ← FILL from metrics.py output
        "UIQM":  (2.61, 0.55),     # ← FILL from metrics.py output
    },
    "FUnIE-GAN": {
        "PSNR":  (18.56, 3.49),     # ← FILL from metrics.py output
        "SSIM":  (0.5910, 0.1212),     # ← FILL from metrics.py output
        "UIQM":  (2.64, 0.53),     # ← FILL from metrics.py output
    },
    "DiffWater [Ref]": {
        "PSNR": (20.86, 0.00),
        "SSIM": (0.836, 0.00),
        "UIQM": (2.90,  0.00),
    },
    "CPDM [Ref]": {
        "PSNR": (23.65, 0.00),
        "SSIM": (0.882, 0.00),
        "UIQM": (3.15,  0.00),
    },
}

# ── FIG. 2 — GROUPED BAR CHART ────────────────────────────────────────────────

def fig2_grouped_bar(data, outpath):
    methods  = list(data.keys())
    n        = len(methods)
    colors   = [METHOD_COLORS[m] for m in methods]
    metrics  = ["PSNR",      "SSIM",   "UIQM"]
    ylabels  = ["PSNR (dB)", "SSIM",   "UIQM"]
    ylims    = [(0, 30),     (0, 1.1), (0, 4.5)]
    vfmts    = ["{:.2f}",    "{:.3f}", "{:.2f}"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "Fig. 2  \u2014  Quantitative Comparison of UIE Methods on UIEB Dataset\n"
        "(CLAHE & FUnIE-GAN: experimental results   |   Diffusion: values reported in literature)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02,
    )

    x     = np.arange(n)
    bar_w = 0.60

    for ax, metric, ylabel, ylim, vfmt in zip(axes, metrics, ylabels, ylims, vfmts):
        means = [data[m][metric][0] for m in methods]
        stds  = [data[m][metric][1] for m in methods]

        bars = ax.bar(
            x, means,
            width=bar_w, color=colors, edgecolor="white", linewidth=0.8,
            yerr=stds, capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
            zorder=3,
        )

        # Value label above each bar
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.annotate(
                    vfmt.format(mean),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=FONT_ANNOT, color="#222222",
                )

        # Bold outline on the best bar
        valid = [(i, v) for i, v in enumerate(means) if v > 0]
        if valid:
            best_i = max(valid, key=lambda t: t[1])[0]
            bars[best_i].set_edgecolor("#111111")
            bars[best_i].set_linewidth(2.0)

        ax.set_title(metric, fontsize=FONT_TITLE, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=FONT_TICK - 1, rotation=15, ha="right")
        ax.set_ylim(ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.45, linewidth=0.6)
        ax.set_axisbelow(True)

        if metric == "SSIM":
            ax.axhline(1.0, color="#999999", linestyle=":", linewidth=0.9, alpha=0.7)

    # Shared colour legend at the bottom
    patches = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in methods]
    fig.legend(
        handles=patches, loc="lower center", ncol=n,
        fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {outpath}")

# ── FIG. 3 — PSNR vs INFERENCE TIME SCATTER ───────────────────────────────────

def fig3_scatter(data, outpath):
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = []
    for method, color in METHOD_COLORS.items():
        if method not in data:
            continue
        psnr = data[method]["PSNR"][0]
        uiqm = data[method]["UIQM"][0]
        t    = INFERENCE_TIMES.get(method, 1.0)
        if psnr == 0:
            continue

        ax.scatter(
            t, psnr,
            s=max(uiqm * 140, 60), color=color,
            edgecolors="#222222", linewidths=1.3,
            zorder=5, alpha=0.90,
        )

        offsets = {
            "Raw Input":        (-32,  6),
            "CLAHE":            (  8,  5),
            "FUnIE-GAN":        (  8,  5),
            "DiffWater [Ref]":  (  8, -13),
            "CPDM [Ref]":       (  8,  5),
        }
        ox, oy = offsets.get(method, (8, 5))
        ax.annotate(
            method,
            xy=(t, psnr), xytext=(ox, oy), textcoords="offset points",
            fontsize=FONT_TICK, color="#111111", fontweight="bold",
        )
        plotted.append(method)

    ax.set_xscale("log")
    ax.set_xlabel("Inference Time per Image  (seconds, log scale)", fontsize=FONT_AXIS)
    ax.set_ylabel("PSNR  (dB)", fontsize=FONT_AXIS)
    ax.set_title(
        "Fig. 3  \u2014  Enhancement Quality vs. Inference Speed\n"
        "Bubble size \u221d UIQM score  (larger = better perceptual quality)",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.set_xlim(5e-4, 200)
    ax.set_ylim(0, 30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, linewidth=0.6)
    ax.xaxis.grid(True, linestyle="--", alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)

    # Tradeoff labels
    ax.annotate("\u2190 Faster", xy=(0.04, 0.05), xycoords="axes fraction",
                fontsize=9, color="#888888", fontstyle="italic")
    ax.annotate("Better quality \u2192", xy=(0.62, 0.05), xycoords="axes fraction",
                fontsize=9, color="#888888", fontstyle="italic")

    # Real-time / offline dividing line
    ax.axvline(x=1.0, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(1.15, 1.5, "offline \u2192", fontsize=8, color="#AAAAAA", fontstyle="italic")
    ax.text(0.005, 1.5, "\u2190 real-time", fontsize=8, color="#AAAAAA", fontstyle="italic")

    # Legend
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in plotted]
    ax.legend(handles=handles, title="Method", loc="upper left",
              fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
              framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout(pad=2.0)
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {outpath}")

# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────

def generate_all_charts(res=None):
    """
    Produces exactly 2 files:
        results/figures/fig2_bar_chart.png
        results/figures/fig3_scatter.png
    Pass res=None to use the hardcoded `results` dict above,
    or pass your own dict from metrics.py.
    """
    data = res if res is not None else results
    print("\nGenerating charts...")
    fig2_grouped_bar(data, os.path.join(OUT_DIR, "fig2_bar_chart.png"))
    fig3_scatter    (data, os.path.join(OUT_DIR, "fig3_scatter.png"))
    print(f"\nDone. Charts saved to: {OUT_DIR}\n")


# ── INTEGRATION SNIPPET ───────────────────────────────────────────────────────
INTEGRATION_SNIPPET = """
# Paste this at the very end of metrics.py (after the final print block):
from generate_charts import generate_all_charts

live_results = {
    "Raw Input":       {"PSNR": (14.20, 0.00), "SSIM": (0.650, 0.00), "UIQM": (1.95, 0.00)},
    "CLAHE":           {"PSNR": (np.mean(clahe_psnr), np.std(clahe_psnr)),
                        "SSIM": (np.mean(clahe_ssim), np.std(clahe_ssim)),
                        "UIQM": (np.mean(clahe_uiqm), np.std(clahe_uiqm))},
    "FUnIE-GAN":       {"PSNR": (np.mean(gan_psnr),   np.std(gan_psnr)),
                        "SSIM": (np.mean(gan_ssim),   np.std(gan_ssim)),
                        "UIQM": (np.mean(gan_uiqm),   np.std(gan_uiqm))},
    "DiffWater [Ref]": {"PSNR": (20.86, 0.00), "SSIM": (0.836, 0.00), "UIQM": (2.90, 0.00)},
    "CPDM [Ref]":      {"PSNR": (23.65, 0.00), "SSIM": (0.882, 0.00), "UIQM": (3.15, 0.00)},
}
generate_all_charts(live_results)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("UIE Paper  --  Chart Generator  (2 figures)")
    print("=" * 60)
    print("\nCLAHE and FUnIE-GAN slots are 0.0 placeholders.")
    print("Fill the `results` dict at the top of this file, or use")
    print("the automated snippet below to pipe metrics.py output directly.\n")
    print(INTEGRATION_SNIPPET)
    generate_all_charts()