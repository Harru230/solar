#!/usr/bin/env python3
"""Generates publication-ready pipeline overview diagram for the 4-page Overleaf report."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_PATH = Path("/home/adamp/solar_release/report/figures/pipeline_diagram.png")

fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=260)
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 4.2)
ax.axis("off")

def draw_box(x, y, w, h, title, body, fc="#eef4ff", ec="#2b5b84", lw=1.6, title_color="#102a43"):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h / 2 - 0.14, body, ha="center", va="center", fontsize=7.8, color="#243b53", linespacing=1.32)

def draw_arrow(x0, y0, x1, y1, color="#334e68"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.6, color=color))

# Stage 1: Input & CLAHE
draw_box(
    0.2, 1.35, 1.85, 1.65,
    "1. Input & CLAHE",
    "GONG H-alpha\n(2048x2048)\nCh0: Linear norm\nCh1: CLAHE (2.0, 8x8)\nCh2: CLAHE (4.0, 16x16)",
    fc="#f0f4f8", ec="#486581"
)

# Stage 2: 6-Family Detector Pool (30 models)
draw_box(
    2.40, 2.25, 2.45, 1.65,
    "2A. 6-Family Mask R-CNN",
    "ResNet50-FPN v2 (30 ckpts)\nRes: 1536 / 1792 / 2048\nAnchors: 3 & 5 ratios; BCE+Dice\n4-Flip TTA -> WMF (IoU>0.40)",
    fc="#e6fffa", ec="#0d9488", title_color="#115e59"
)

# Stage 2B: Full-Disk Semantic UNet
draw_box(
    2.40, 0.30, 2.45, 1.55,
    "2B. Full-Disk Semantic UNet",
    "5-Fold ResNet50 (1024x1024)\n+ ResNeXt101 (1280x1280)\n3ch: fg, core, edge\n(frPyObjects exact GT)",
    fc="#fff7ed", ec="#ea580c", title_color="#9a3412"
)

# Stage 3: Boundary & Context Refiners
draw_box(
    5.25, 1.25, 2.35, 1.85,
    "3. ROI Boundary Refiners",
    "6-ch ROI crop (H-alpha + WMF\n+ Semantic Prior)\nLocal ResUNet + WideRefV2\nTargets: exact frPyObjects\nBCE + Dice + Edge + IoU head",
    fc="#eff6ff", ec="#2563eb", title_color="#1e40af"
)

# Stage 4: Ranker
draw_box(
    8.00, 1.35, 1.65, 1.65,
    "4. Instance Scorer",
    "5-Fold Cross-Fit\nLightGBM (27 feats)\nTarget: Hybrid IoU\n(raw + refined GT)\nNested tau opt.",
    fc="#f5f3ff", ec="#7c3aed", title_color="#5b21b6"
)

# Stage 5: Disjoint Canvas
draw_box(
    10.00, 1.35, 1.30, 1.65,
    "5. Disjoint Output",
    "Mask NMS (0.03)\nGreedy paint\nOcc >= 50%\n2048x2048\nCOCO RLE",
    fc="#ecfdf5", ec="#059669", title_color="#065f46"
)

draw_arrow(2.07, 2.35, 2.38, 2.95)
draw_arrow(2.07, 1.95, 2.38, 1.15)
draw_arrow(4.87, 3.00, 5.23, 2.40)
draw_arrow(4.87, 1.15, 5.23, 1.90)
draw_arrow(7.62, 2.18, 7.98, 2.18)
draw_arrow(4.87, 3.30, 8.82, 3.30)
ax.plot([8.82, 8.82], [3.30, 3.04], color="#334e68", lw=1.6)
ax.annotate("", xy=(8.82, 3.00), xytext=(8.82, 3.08), arrowprops=dict(arrowstyle="-|>,head_width=0.20,head_length=0.28", lw=1.6, color="#334e68"))
draw_arrow(9.67, 2.18, 9.98, 2.18)

plt.tight_layout(pad=0.3)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=260, bbox_inches="tight")
print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
