#!/usr/bin/env python3
"""
regenerate_shap_plots.py

Regenerates the SHAP waterfall plots directly from an already-saved
shap_contributions.csv (from plot_shap.py) -- no SHAP recomputation, no
model loading, no data loading. Same exact figure design (chevron-arrow
bars, positive-only normalized-to-100%, one line per feature), with two
adjustments:
    - title and the "[n = X patients]" badge both moved down slightly,
      giving more clearance from the top edge of the figure (some
      plots, e.g. Brain, had the title sitting close enough to the top
      that it looked cropped)
    - figure is now more RECTANGULAR (wider relative to its height)
      instead of approaching square for larger feature counts

USAGE
    python regenerate_shap_plots.py \
        --contributions-csv /workspace/shap_plots/shap_contributions.csv \
        --output-dir /workspace/shap_plots_v2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLOR_PALETTE = [
    "#1e5c1e", "#2a6d2a", "#367e36", "#449044", "#53a253",
    "#63b463", "#74c674", "#86d886", "#99eb99",
]
SOLID_GREEN = "#5AAC75"


def make_waterfall_plot(feature_names, values_pct, class_name, output_path, n_patients):
    """
    Same design as plot_shap.py's make_waterfall_plot -- chevron-arrow
    bars, positive-only normalized-to-100%, real-measured-text-width
    label placement -- with title/badge moved down for more top
    clearance, and a wider, more rectangular aspect ratio.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    n = len(feature_names)
    bar_color = SOLID_GREEN

    feature_names = list(reversed(feature_names))
    values_pct = list(reversed(values_pct))

    cumulative = np.concatenate([[0], np.cumsum(values_pct)])
    total = cumulative[-1]

    # wider, more rectangular figure -- more width per feature-row of
    # height than before (was 10 wide x ~0.65*n+1.8 tall, approaching
    # square for larger n; now scales height more slowly and starts
    # from a wider base, staying visibly landscape-oriented regardless
    # of feature count)
    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.55 * n + 1.7)))

    bar_height = 0.62
    point_width = max(1.2, total * 0.015)

    text_objects = []
    for i, (name, val) in enumerate(zip(feature_names, values_pct)):
        x_start, x_end = cumulative[i], cumulative[i] + val
        y = i
        vertices = [
            (x_start, y + bar_height / 2),
            (x_end, y + bar_height / 2),
            (x_end + point_width, y),
            (x_end, y - bar_height / 2),
            (x_start, y - bar_height / 2),
        ]
        ax.add_patch(Polygon(vertices, closed=True, facecolor=bar_color,
                              edgecolor="white", linewidth=1.2))
        label = f"+{val:.2f}"
        txt = ax.text(x_start + val / 2, y, label, ha="center", va="center",
                       fontsize=10, fontweight="bold", color="white")
        text_objects.append((txt, i, x_start, x_end, val))

    ax.set_ylim(-0.7, n - 0.5)
    ax.set_xlim(0, total * 1.12)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for txt, i, x_start, x_end, val in text_objects:
        bbox = txt.get_window_extent(renderer=renderer)
        (x0_data, _), (x1_data, _) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
        label_width_data = x1_data - x0_data
        bar_width_data = x_end - x_start
        if label_width_data > bar_width_data * 0.92:
            txt.set_position((x_end + point_width + total * 0.01, i))
            txt.set_ha("left")
            txt.set_color("#0a2e0a")

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"1 = {name.upper()}" for name in feature_names], fontsize=10)
    ax.set_xlabel("Contribution (%)", fontsize=10)

    # title and patient count combined into ONE line, directly next to
    # each other -- an axes-level title (not a separately-positioned
    # figure-level suptitle/text pair) so it sits close to the plot
    # naturally and works correctly with tight_layout()
    ax.set_title(f"{class_name} \u2014 Key Genomic Drivers   [n = {n_patients} patients]",
                 fontsize=15, fontweight="bold", pad=14)

    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Regenerate SHAP waterfall plots from an already-saved contributions CSV")
    parser.add_argument("--contributions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.contributions_csv)
    required_cols = {"tumor_origin", "feature", "normalized_pct", "n_patients"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{args.contributions_csv} is missing expected columns "
                          f"(need {required_cols}, has {set(df.columns)})")

    for class_name, group in df.groupby("tumor_origin"):
        # normalized_pct was saved already sorted biggest-first per class
        # (that's how plot_shap.py wrote it) -- preserve that order here
        group = group.sort_values("normalized_pct", ascending=False)
        feature_names = group["feature"].tolist()
        values_pct = group["normalized_pct"].tolist()
        n_patients = int(group["n_patients"].iloc[0])

        safe_name = str(class_name).replace("/", "-").replace(" ", "_")
        plot_path = output_dir / f"{safe_name}_waterfall.png"
        make_waterfall_plot(feature_names, values_pct, class_name, plot_path, n_patients)
        print(f"{class_name}: {n_patients} patients, saved {plot_path}")


if __name__ == "__main__":
    main()