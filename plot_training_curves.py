#!/usr/bin/env python3
"""
plot_training_curves.py

Plots per-fold training/validation loss and accuracy curves, one line
per fold, using the project's blue/purple/green/red/orange palette.

Supports BOTH model types this project trains:

xgboost  -- reads <model>_fold<N>_training_curve.csv
            columns:
            round,
            train_mlogloss,
            val_mlogloss,
            train_accuracy,
            val_accuracy

mil      -- reads <model>_fold<N>_history.csv
            columns:
            epoch,
            train_loss,
            val_loss,
            train_top1_accuracy,
            val_top1_accuracy

Produces 4 figures per model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


COLOR_PALETTE = [
    "#1f77b4",  # blue
    "#9467bd",  # purple
    "#2ca02c",  # green
    "#d62728",  # red
    "#ff7f0e",  # orange
]


MODEL_TYPE_SPECS = {
    "xgboost": {
        "file_suffix": "training_curve.csv",
        "x_column": "round",
        "xlabel": "Boosting Round",
        "columns": {
            "train_loss": "train_mlogloss",
            "val_loss": "val_mlogloss",
            "train_accuracy": "train_accuracy",
            "val_accuracy": "val_accuracy",
        },
    },

    "mil": {
        "file_suffix": "history.csv",
        "x_column": "epoch",
        "xlabel": "Epoch",
        "columns": {
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "train_accuracy": "train_top1_accuracy",
            "val_accuracy": "val_top1_accuracy",
        },
    },
}


def plot_fold_curves(
    fold_curves,
    xlabel,
    ylabel,
    title,
    outpath,
):
    """
    fold_curves:
    list of (fold_label, list_of_values) tuples.

    Each fold uses its own color from COLOR_PALETTE.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (fold_label, curve) in enumerate(fold_curves):
        ax.plot(
            range(1, len(curve) + 1),
            curve,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            linewidth=3.2,
            alpha=0.95,
            label=fold_label,
        )

    ax.set_xlabel(
        xlabel,
        fontsize=14,
        fontweight="bold",
        labelpad=12,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=14,
        fontweight="bold",
        labelpad=16,
    )

    ax.set_title(
        title,
        fontsize=30,
        fontweight="bold",
        pad=20,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=12)
    )

    legend = ax.legend(
        fontsize=15,
        frameon=True,
        shadow=True,
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()

    plt.savefig(
        outpath,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {outpath}")


def load_fold_curves(
    model_dir,
    model_name,
    n_folds,
    column,
    file_suffix,
):
    model_dir = Path(model_dir)

    curves = []

    for fold_idx in range(n_folds):

        curve_path = (
            model_dir
            / f"{model_name}_fold{fold_idx}_{file_suffix}"
        )

        if not curve_path.exists():
            print(
                f"WARNING: {curve_path} not found -- "
                f"skipping fold {fold_idx}"
            )
            continue

        df = pd.read_csv(curve_path)

        if column not in df.columns:
            print(
                f"WARNING: column '{column}' not found in "
                f"{curve_path} "
                f"(has: {list(df.columns)}) -- "
                f"skipping fold {fold_idx}"
            )
            continue

        curves.append(
            (
                f"Fold {fold_idx + 1}",
                df[column].tolist(),
            )
        )

    return curves


def generate_plots_for_one_model(
    model_dir,
    model_name,
    model_type,
    output_dir,
    n_folds,
):
    spec = MODEL_TYPE_SPECS[model_type]

    if model_name.lower() == "xgboost":
        display_name = "XGBoost"
        filename_prefix = ""

    elif model_name.lower() == "abmil":
        display_name = "ABMIL"
        filename_prefix = "abmil_"

    elif model_name.lower() == "clam":
        display_name = "CLAM"
        filename_prefix = "clam_"

    elif model_name.lower() == "transmil":
        display_name = "TransMIL"
        filename_prefix = "transmil_"

    else:
        display_name = model_name
        filename_prefix = f"{model_name}_"

    plot_specs = [
        (
            "train_loss",
            "Training Log Loss"
            if model_type == "xgboost"
            else "Training Loss",
            f"{display_name} Training Loss per Fold",
            f"{filename_prefix}train_loss_curve.png",
        ),

        (
            "train_accuracy",
            "Training Accuracy",
            f"{display_name} Training Accuracy per Fold",
            f"{filename_prefix}train_accuracy_curve.png",
        ),

        (
            "val_loss",
            "Validation Log Loss"
            if model_type == "xgboost"
            else "Validation Loss",
            f"{display_name} Validation Loss per Fold",
            f"{filename_prefix}val_loss_curve.png",
        ),

        (
            "val_accuracy",
            "Validation Accuracy",
            f"{display_name} Validation Accuracy per Fold",
            f"{filename_prefix}val_accuracy_curve.png",
        ),
    ]

    for metric_key, ylabel, title, filename in plot_specs:

        column = spec["columns"][metric_key]

        curves = load_fold_curves(
            model_dir=model_dir,
            model_name=model_name,
            n_folds=n_folds,
            column=column,
            file_suffix=spec["file_suffix"],
        )

        if not curves:
            print(
                f"No fold curves found for "
                f"{model_name}/{metric_key} -- "
                f"skipping this plot"
            )
            continue

        plot_fold_curves(
            curves,
            spec["xlabel"],
            ylabel,
            title,
            output_dir / filename,
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot per-fold training/validation curves "
            "(XGBoost or MIL)"
        )
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=[
            "xgboost",
            "mil",
        ],
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Directory containing this model's "
            "fold curve/history files"
        ),
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "xgboost, or one of "
            "abmil/clam/transmil for MIL"
        ),
    )

    parser.add_argument(
        "--mil-base-dir",
        default=None,
        help=(
            "Base directory containing "
            "mil_models_grid_<architecture>/ "
            "subfolders"
        ),
    )

    parser.add_argument(
        "--mil-architectures",
        nargs="+",
        default=[
            "abmil",
            "clam",
            "transmil",
        ],
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.model_type == "mil"
        and args.mil_base_dir
    ):

        for arch in args.mil_architectures:

            model_dir = (
                Path(args.mil_base_dir)
                / f"mil_models_grid_{arch}"
            )

            print(f"\n=== {arch} ===")

            generate_plots_for_one_model(
                model_dir=model_dir,
                model_name=arch,
                model_type="mil",
                output_dir=output_dir,
                n_folds=args.n_folds,
            )

    else:

        if not args.model_dir or not args.model_name:

            parser.error(
                "--model-dir and --model-name "
                "are required unless using "
                "--mil-base-dir "
                "(MIL multi-architecture mode)"
            )

        generate_plots_for_one_model(
            model_dir=Path(args.model_dir),
            model_name=args.model_name,
            model_type=args.model_type,
            output_dir=output_dir,
            n_folds=args.n_folds,
        )


if __name__ == "__main__":
    main()