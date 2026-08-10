#!/usr/bin/env python3
"""
plot_confusion_matrix.py

Aggregates all 5 cross-validation folds' saved confusion matrix CSVs
(from fusion.py's save_confusion_matrix()) into one combined confusion
matrix -- summing raw counts across folds, since each fold's test set
covers a different set of patients, so summing gives the total count
for every (true, predicted) pair across the ENTIRE cohort -- and
renders it using the purple color palette.

USAGE
python plot_confusion_matrix.py \
  --results-dir /workspace/fusion_results_abmil \
  --label stack_fused \
  --n-folds 5 \
  --output /workspace/fusion_results_abmil/combined_confusion_matrix.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm, LinearSegmentedColormap


PURPLE_CM_COLORS = [
    "#F0E8F8",
    "#D1B8E8",
    "#B188D8",
    "#935AC8",
    "#7D3DB8",
    "#652A9D",
    "#581E8E",
    "#3A145E",
    "#1D0A2E",
]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot a combined confusion matrix across CV folds"
    )

    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory with fold*_confusion_matrix.csv files",
    )

    parser.add_argument(
        "--label",
        required=True,
        help="e.g. stack_fused, stack_mil, multiply_fused",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    combined_cm = None
    n_found = 0

    for fold_idx in range(args.n_folds):
        csv_path = (
            results_dir
            / f"fold{fold_idx}_{args.label}_confusion_matrix.csv"
        )

        if not csv_path.exists():
            print(
                f"WARNING: {csv_path} not found -- skipping this fold"
            )
            continue

        fold_cm = pd.read_csv(
            csv_path,
            index_col=0,
        )

        n_found += 1

        if combined_cm is None:
            combined_cm = fold_cm.copy()
        else:
            combined_cm = combined_cm.add(
                fold_cm,
                fill_value=0,
            )

    if combined_cm is None:
        raise FileNotFoundError(
            f"No confusion matrix CSVs found for label "
            f"'{args.label}' in {results_dir}"
        )

    combined_cm = (
        combined_cm
        .fillna(0)
        .astype(int)
    )

    print(
        f"Aggregated {n_found}/{args.n_folds} folds' confusion matrices"
    )

    print(
        f"Combined matrix: {combined_cm.shape[0]} classes, "
        f"{combined_cm.values.sum()} total patients"
    )

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_purples",
        PURPLE_CM_COLORS,
    )

    cm_values = combined_cm.values

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    im = ax.imshow(
        cm_values,
        interpolation="nearest",
        cmap=custom_cmap,
        norm=LogNorm(
            vmin=max(
                cm_values[cm_values > 0].min(),
                0.5,
            ),
            vmax=cm_values.max(),
        ),
    )

    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.ax.tick_params(
        labelsize=11
    )

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    class_names = combined_cm.index.tolist()
    tick_marks = range(len(class_names))

    ax.set_xticks(
        list(tick_marks)
    )

    ax.set_yticks(
        list(tick_marks)
    )

    # Tumor-origin labels
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_yticklabels(
        class_names,
        fontsize=13,
        fontweight="bold",
    )

    for tick in ax.get_xticklabels():
        tick.set_fontsize(13)
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontsize(13)
        tick.set_fontweight("bold")

    # Axis labels
    ax.set_ylabel(
        "True Label",
        fontsize=14,
        fontweight="bold",
        labelpad=12,
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=14,
        fontweight="bold",
        labelpad=12,
    )

    # Main title
    ax.set_title(
        "Confusion Matrix - TRACE",
        fontsize=25,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        f"Saved combined confusion matrix to {output_path}"
    )

    combined_csv_path = output_path.with_suffix(
        ".csv"
    )

    combined_cm.to_csv(
        combined_csv_path
    )

    print(
        f"Saved combined confusion matrix raw counts to "
        f"{combined_csv_path}"
    )


if __name__ == "__main__":
    main()