import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator


_this_dir = str(Path(__file__).resolve().parent)
_removed = []
for entry in ("", ".", _this_dir):
    if entry in sys.path:
        sys.path.remove(entry)
        _removed.append(entry)
import shap
for entry in _removed:
    sys.path.insert(0, entry)

from models.genomics.tcga_genomic_classifier import load_matrix, load_site_fold_assignment


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


def compute_shap_by_class(args):


    X, tumor_origins_all, case_ids_all, feature_names = load_matrix(args.matrix_prefix)
    folds = load_site_fold_assignment(Path(args.site_fold_csv), case_ids_all)
    label_encoder = joblib.load(Path(args.model_dir) / "label_encoder.joblib")
    class_names = list(label_encoder.classes_)

    vectors_by_class = {c: [] for c in class_names}

    for fold_idx in range(args.n_folds):
        model_path = Path(args.model_dir) / f"{args.model_name}_fold{fold_idx}_model.joblib"
        if not model_path.exists():
            print(f"WARNING: {model_path} not found -- skipping fold {fold_idx}")
            continue
        artifact = joblib.load(model_path)
        model, scaler = artifact["model"], artifact["scaler"]


        booster = model.get_booster()
        booster.set_param({"device": "cpu"})
        booster.set_param({"eval_metric": "mlogloss"})

        test_mask = folds == fold_idx
        X_test = X[test_mask]
        origins_test = tumor_origins_all[test_mask]
        n_test = X_test.shape[0]
        print(f"Fold {fold_idx}: computing SHAP for {n_test} test patients (CPU, in batches of "
              f"{args.shap_batch_size})...")

        X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
        explainer = shap.TreeExplainer(booster)

        for batch_start in range(0, n_test, args.shap_batch_size):
            batch_end = min(batch_start + args.shap_batch_size, n_test)
            X_batch = X_test_scaled[batch_start:batch_end]
            X_batch_dense = np.asarray(
                X_batch.toarray() if hasattr(X_batch, "toarray") else X_batch, dtype=np.float32,
            )
            shap_values = explainer.shap_values(X_batch_dense)

            for i in range(batch_end - batch_start):
                true_class = origins_test[batch_start + i]
                if true_class not in vectors_by_class:
                    continue
                class_idx = list(class_names).index(true_class)
                vectors_by_class[true_class].append(shap_values[i, :, class_idx])

            print(f"  ...{batch_end}/{n_test} patients done")

    return vectors_by_class, feature_names, class_names


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
    parser = argparse.ArgumentParser(description="Positive-only, normalized SHAP waterfall plots per tumor origin")
    parser.add_argument("--matrix-prefix", required=True)
    parser.add_argument("--site-fold-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default="xgboost")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=8, help="Number of top positive features to show per class")
    parser.add_argument("--shap-batch-size", type=int, default=100,
                         help="Patients processed per SHAP batch -- keeps peak memory bounded "
                              "regardless of feature count. Lower this further if you still hit "
                              "an out-of-memory error.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_by_class, feature_names, class_names = compute_shap_by_class(args)

    all_rows = []
    for class_name in class_names:
        vectors = vectors_by_class[class_name]
        if not vectors:
            print(f"{class_name}: no test patients found across any fold -- skipping")
            continue

        mean_contributions = np.mean(np.stack(vectors), axis=0)

        positive_mask = mean_contributions > 0
        pos_features = np.array(feature_names)[positive_mask]
        pos_values = mean_contributions[positive_mask]

        if len(pos_values) == 0:
            print(f"{class_name}: no positive-contributing features found -- skipping")
            continue

        order = np.argsort(-pos_values)[:args.top_n]
        top_features = pos_features[order]
        top_values = pos_values[order]

        normalized_pct = top_values / top_values.sum() * 100

        safe_name = class_name.replace("/", "-").replace(" ", "_")
        plot_path = output_dir / f"{safe_name}_waterfall.png"
        make_waterfall_plot(top_features, normalized_pct, class_name, plot_path, len(vectors))
        print(f"{class_name}: {len(vectors)} patients, saved {plot_path}")

        for feat, raw_val, pct in zip(top_features, top_values, normalized_pct):
            all_rows.append({"tumor_origin": class_name, "feature": feat,
                              "mean_raw_shap": raw_val, "normalized_pct": pct, "n_patients": len(vectors)})

    pd.DataFrame(all_rows).to_csv(output_dir / "shap_contributions.csv", index=False)
    print(f"\nSaved underlying numbers to {output_dir / 'shap_contributions.csv'}")


if __name__ == "__main__":
    main()