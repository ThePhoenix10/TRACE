import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator


BLUE_PALETTE = [
    "#1e5c8e",
    "#2a6d9d",
    "#367eac",
    "#4490ba",
    "#53a2c8",
    "#63b4d6",
    "#74c6e4",
    "#86d8f1",
    "#99ebff",
]


def get_bar_colors(palette, n):
    if n <= len(palette):
        return palette[:n]

    indices = np.linspace(0, len(palette) - 1, n)
    indices = np.round(indices).astype(int)

    return [palette[i] for i in indices]


def make_waterfall_plot(
    feature_names,
    values_pct,
    class_name,
    output_path,
    n_patients,
):
    n = len(feature_names)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    bar_colors = get_bar_colors(BLUE_PALETTE, n)

    feature_names = list(reversed(feature_names))
    values_pct = list(reversed(values_pct))
    bar_colors = list(reversed(bar_colors))

    cumulative = np.concatenate([[0], np.cumsum(values_pct)])
    total = cumulative[-1]

    fig, ax = plt.subplots(
        figsize=(12.1, max(3.2, 0.55 * n + 1.7))
    )

    bar_height = 0.62
    point_width = max(1.2, total * 0.015)

    text_objects = []

    for i, (name, val, bar_color) in enumerate(
        zip(feature_names, values_pct, bar_colors)
    ):
        x_start = cumulative[i]
        x_end = cumulative[i] + val
        y = i

        vertices = [
            (x_start, y + bar_height / 2),
            (x_end, y + bar_height / 2),
            (x_end + point_width, y),
            (x_end, y - bar_height / 2),
            (x_start, y - bar_height / 2),
        ]

        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=bar_color,
                edgecolor="white",
                linewidth=1.2,
            )
        )

        txt = ax.text(
            x_start + val / 2,
            y,
            f"+{val:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

        text_objects.append(
            (txt, i, x_start, x_end)
        )

    ax.set_ylim(-0.7, n - 0.5)
    ax.set_xlim(0, total * 1.12)

    # X-axis ticks in increments of 10
    ax.xaxis.set_major_locator(
        MultipleLocator(10)
    )

    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for txt, i, x_start, x_end in text_objects:
        bbox = txt.get_window_extent(renderer=renderer)

        (x0_data, _), (x1_data, _) = (
            inv.transform((bbox.x0, bbox.y0)),
            inv.transform((bbox.x1, bbox.y1)),
        )

        label_width_data = x1_data - x0_data
        bar_width_data = x_end - x_start

        if label_width_data > bar_width_data * 0.92:
            txt.set_position(
                (
                    x_end + point_width + total * 0.01,
                    i,
                )
            )

            txt.set_ha("left")
            txt.set_color("#0a2e4f")
            txt.set_fontweight("bold")

    ax.set_yticks(range(n))

    ax.set_yticklabels(
        [f"1 = {name.upper()}" for name in feature_names],
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Contribution (%)",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(
        f"{class_name} — Key Genomic Drivers   [n = {n_patients} patients]",
        fontsize=18,
        fontweight="bold",
        pad=24,
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(13)

    ax.grid(
        True,
        axis="both",
        alpha=0.3,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(1.3)

    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.3)

    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_linewidth(1.3)

    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_linewidth(1.3)

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    plt.savefig(
        output_path,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Replot saved SHAP contributions using the blue palette."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_columns = {
        "tumor_origin",
        "feature",
        "normalized_pct",
        "n_patients",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name, group in df.groupby("tumor_origin"):
        group = group.sort_values(
            "normalized_pct",
            ascending=False,
        )

        features = group["feature"].tolist()
        values = group["normalized_pct"].tolist()
        n_patients = int(
            group["n_patients"].iloc[0]
        )

        safe_name = (
            class_name
            .replace("/", "-")
            .replace(" ", "_")
        )

        output_path = (
            output_dir
            / f"{safe_name}_waterfall.png"
        )

        make_waterfall_plot(
            features,
            values,
            class_name,
            output_path,
            n_patients,
        )

        print(
            f"Saved: {output_path}"
        )

    print(
        f"\nDone. Plots saved to {output_dir}"
    )


if __name__ == "__main__":
    main()