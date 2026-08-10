import argparse
import itertools
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from models.genomics.tcga_genomic_classifier import (
    build_sklearn_model, compute_ranking_metrics, fit_sklearn_model,
    get_full_width_scores, load_matrix, load_site_fold_assignment,
)

MODALITY_PREFIXES = {


    "mutation": ("som_mut", "som_mat", "mut"),
    "cna": ("cna",),
    "signature": ("sig",),
}


def identify_modality_columns(feature_names, modality):
    prefixes = MODALITY_PREFIXES[modality]
    return np.array([any(col.startswith(p) for p in prefixes) for col in feature_names])


def run_one_fold(fold_idx, X, y, folds, n_classes, label_encoder, best_params):


    train_mask = folds != fold_idx
    test_mask = folds == fold_idx
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = build_sklearn_model(dict(best_params), n_classes)
    X_fit, X_es_val, y_fit, y_es_val = train_test_split_stratified(X_train, y_train)
    model, _ = fit_sklearn_model(model, X_fit, y_fit, X_es_val, y_es_val)

    preds = model.predict(X_test)
    scores = get_full_width_scores(model, X_test, n_classes)
    metrics = compute_ranking_metrics(y_test, scores, preds, n_classes)
    return metrics


def train_test_split_stratified(X, y):
    from sklearn.model_selection import train_test_split
    try:
        return train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    except ValueError:
        return train_test_split(X, y, test_size=0.1, random_state=0)


def run_ablation(modality_name, keep_mask_or_none, X_full, y, folds, n_folds,
                  n_classes, label_encoder, classifier_dir, output_dir):


    X = X_full[:, keep_mask_or_none] if keep_mask_or_none is not None else X_full
    n_removed = X_full.shape[1] - X.shape[1]
    print(f"[{modality_name}] using {X.shape[1]} / {X_full.shape[1]} features "
          f"({n_removed} removed)")

    fold_rows = []
    for fold_idx in range(n_folds):
        params_path = Path(classifier_dir) / f"xgboost_fold{fold_idx}_best_params.json"
        if not params_path.exists():
            print(f"  WARNING: {params_path} not found -- skipping fold {fold_idx}")
            continue
        best_params = json.loads(params_path.read_text())

        metrics = run_one_fold(fold_idx, X, y, folds, n_classes, label_encoder, best_params)
        print(f"  fold {fold_idx}: top1={metrics['top1_accuracy']:.4f}, "
              f"weighted_F1={metrics['weighted_f1']:.4f}")
        fold_rows.append({"fold": fold_idx, **metrics})

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        safe_name = modality_name.replace(" ", "_")
        fold_df.to_csv(output_dir / f"ablation_{safe_name}_folds.csv", index=False)
    return fold_df


PANEL_COLORS = [
    "#1e5c8e",
    "#4490ba",
    "#74c6e4",
]

LABEL_MAP = {
    "mutation removed": "No Som. Mut.",
    "cna removed": "No CNA",
    "signature removed": "No SBS",
    "mutation+cna removed": "No Som. Mut. + CNA",
    "mutation+signature removed": "No Som. Mut. + SBS",
    "cna+signature removed": "No CNA + SBS",
}

METRIC_SPECS = [
    ("top1_drop_from_baseline", "Top-1 Accuracy Δ (%)"),
    ("top3_drop_from_baseline", "Top-3 Accuracy Δ (%)"),
    ("weighted_f1_drop_from_baseline", "Weighted F1 Δ (%)"),
]


def plot_ablation_figure(summary_csv, output):
    df = pd.read_csv(summary_csv)

    df = df[
        df["condition"] != "baseline (all features)"
    ].copy()

    df["label"] = (
        df["condition"]
        .map(LABEL_MAP)
        .fillna(df["condition"])
    )

    for col, _ in METRIC_SPECS:
        df[f"{col}_delta_pct"] = -df[col] * 100

    fixed_order = [
        "mutation removed",
        "cna removed",
        "signature removed",
        "mutation+cna removed",
        "mutation+signature removed",
        "cna+signature removed",
    ]

    order_index = {
        cond: i
        for i, cond in enumerate(fixed_order)
    }

    df["_order"] = df["condition"].map(order_index)
    df = df.sort_values("_order")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 6.2),
    )

    fig.suptitle(
        "Ablation Study — Mutation Classifier Δ vs Baseline by Metric",
        fontsize=30,
        fontweight="bold",
        y=1.00,
    )

    for ax, (col, panel_title), color in zip(
        axes,
        METRIC_SPECS,
        PANEL_COLORS,
    ):
        values = df[f"{col}_delta_pct"].tolist()
        labels = df["label"].tolist()
        y_pos = range(len(labels))

        bars = ax.barh(
            y_pos,
            values,
            color=color,
            height=0.6,
            zorder=3,
        )

        for bar, val in zip(bars, values):
            offset = 0.55

            if val < 0:
                label_x = val - offset
                ha = "right"
            else:
                label_x = val + offset
                ha = "left"

            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%",
                va="center",
                ha=ha,
                fontsize=14,
                fontweight="bold",
                color="black",
                clip_on=False,
            )

        ax.set_yticks(
            list(y_pos)
        )

        ax.set_yticklabels(
            labels,
            fontsize=17,
            fontweight="bold",
        )

        ax.yaxis.tick_left()

        ax.tick_params(
            axis="y",
            labelleft=True,
            labelright=False,
            labelsize=17,
            pad=9,
        )

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(17)

        ax.invert_yaxis()

        ax.axvline(
            0,
            color="black",
            linewidth=1.2,
            zorder=4,
        )

        ax.set_title(
            panel_title,
            fontsize=18,
            fontweight="bold",
            pad=14,
        )

        ax.set_xlabel(
            "Δ vs Baseline (%)",
            fontsize=16,
            fontweight="bold",
            labelpad=12,
        )

        ax.tick_params(
            axis="x",
            labelsize=15,
        )

        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(15)

        ax.grid(
            True,
            axis="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.3,
            zorder=0,
        )

        ax.set_axisbelow(True)

        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(1.3)

        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_linewidth(1.3)

        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(1.3)

        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(1.3)

        min_val = min(values)

        left_pad = max(
            8.0,
            abs(min_val) * 0.35,
        )

        ax.set_xlim(
            min_val - left_pad,
            0,
        )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    fig.subplots_adjust(
        wspace=0.85
    )

    output_path = Path(
        output
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
        f"Saved figure to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Feature-ablation study: remove one modality at a time, reusing saved per-fold hyperparameters")
    parser.add_argument("--matrix-prefix", required=True)
    parser.add_argument("--site-fold-csv", required=True)
    parser.add_argument("--classifier-dir", required=True,
                         help="Directory with the ORIGINAL (full-feature) nested-CV results -- "
                              "reads xgboost_fold<N>_best_params.json from here")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-class-size", type=int, default=2)
    parser.add_argument("--sanity-fold", type=int, default=0,
                         help="Which fold to use for the initial full-feature sanity check")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading matrix...")
    X, tumor_origins, case_ids, feature_names = load_matrix(args.matrix_prefix)
    print(f"Loaded matrix: {X.shape[0]} patients x {X.shape[1]} features")

    folds = load_site_fold_assignment(Path(args.site_fold_csv), case_ids)

    class_counts = pd.Series(tumor_origins).value_counts()
    small_classes = class_counts[class_counts < args.min_class_size].index.tolist()
    keep_mask = (folds != -1)
    if small_classes:
        print(f"Dropping {len(small_classes)} tumor origin(s) with fewer than "
              f"{args.min_class_size} patients: {small_classes}")
        keep_mask = keep_mask & (~pd.Series(tumor_origins).isin(small_classes).to_numpy())

    X, tumor_origins, case_ids, folds = X[keep_mask], tumor_origins[keep_mask], case_ids[keep_mask], folds[keep_mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tumor_origins)
    n_classes = len(label_encoder.classes_)
    print(f"Final dataset: {X.shape[0]} patients, {n_classes} tumor origins, {args.n_folds} CV folds\n")


    for modality in MODALITY_PREFIXES:
        mask = identify_modality_columns(feature_names, modality)
        print(f"Modality '{modality}': {mask.sum()} matching columns identified")
    print()


    print(f"=== SANITY CHECK: fold {args.sanity_fold}, full feature set, saved hyperparameters ===")
    sanity_params_path = Path(args.classifier_dir) / f"xgboost_fold{args.sanity_fold}_best_params.json"
    if not sanity_params_path.exists():
        print(f"ERROR: {sanity_params_path} not found -- cannot run sanity check. Stopping.")
        return
    best_params = json.loads(sanity_params_path.read_text())
    sanity_metrics = run_one_fold(args.sanity_fold, X, y, folds, n_classes, label_encoder, best_params)

    sanity_lines = [
        f"Sanity check -- fold {args.sanity_fold}, full feature set ({X.shape[1]} features), "
        f"reused saved hyperparameters: {best_params}",
        f"  top1_accuracy: {sanity_metrics['top1_accuracy']:.4f}",
        f"  top3_accuracy: {sanity_metrics['top3_accuracy']:.4f}",
        f"  weighted_f1:   {sanity_metrics['weighted_f1']:.4f}",
    ]

    original_report_path = Path(args.classifier_dir) / f"xgboost_fold{args.sanity_fold}_classification_report.txt"
    if original_report_path.exists():
        original_weighted_f1 = None
        for line in original_report_path.read_text().splitlines():
            if line.startswith("Weighted F1:"):
                original_weighted_f1 = float(line.split(":")[1].strip())
        if original_weighted_f1 is not None:
            diff = sanity_metrics["weighted_f1"] - original_weighted_f1
            sanity_lines.append(f"  Original run's weighted_F1 for this fold: {original_weighted_f1:.4f}")
            sanity_lines.append(f"  Difference: {diff:+.4f}")
            if abs(diff) > 0.03:
                sanity_lines.append(
                    "  WARNING: this differs from the original run by more than 0.03 -- the "
                    "reuse-hyperparameters mechanism may not be reproducing the original setup "
                    "correctly. Investigate before trusting the ablation results below."
                )
            else:
                sanity_lines.append("  OK -- closely matches the original run. Proceeding with ablations.")
    else:
        sanity_lines.append(f"  (no original report found at {original_report_path} to compare against)")

    print("\n".join(sanity_lines))
    (output_dir / "sanity_check.txt").write_text("\n".join(sanity_lines))
    print()


    print("=== Baseline: reading existing per-fold results (no re-run needed) ===")
    baseline_rows = []
    for fold_idx in range(args.n_folds):
        report_path = Path(args.classifier_dir) / f"xgboost_fold{fold_idx}_classification_report.txt"
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found -- cannot read baseline for fold {fold_idx}")
            continue
        weighted_f1 = top1 = top3 = top5 = macro_f1 = None
        for line in report_path.read_text().splitlines():
            if line.startswith("Weighted F1:"):
                weighted_f1 = float(line.split(":")[1].strip())
            elif line.startswith("Macro F1:"):
                macro_f1 = float(line.split(":")[1].strip())
            elif line.startswith("Top-1 accuracy:"):
                top1 = float(line.split(":")[1].strip())
            elif line.startswith("Top-3 accuracy:"):
                top3 = float(line.split(":")[1].strip())
            elif line.startswith("Top-5 accuracy:"):
                top5 = float(line.split(":")[1].strip())
        baseline_rows.append({"fold": fold_idx, "top1_accuracy": top1, "top3_accuracy": top3,
                               "top5_accuracy": top5, "weighted_f1": weighted_f1, "macro_f1": macro_f1})
        print(f"  fold {fold_idx}: top1={top1}, weighted_F1={weighted_f1} (from existing report)")

    baseline_fold_df = pd.DataFrame(baseline_rows)
    baseline_fold_df.to_csv(output_dir / "ablation_baseline_all_features_folds.csv", index=False)
    if baseline_fold_df["weighted_f1"].isna().any():
        print("  WARNING: some folds' baseline metrics couldn't be read from the existing reports "
              "-- if that classification_report.txt format differs from what's expected here, "
              "the drop-from-baseline numbers below may be incomplete.")
    print()

    summary_rows = [{
        "condition": "baseline (all features)",
        "n_features": X.shape[1],
        "top1_mean": baseline_fold_df["top1_accuracy"].mean(), "top1_std": baseline_fold_df["top1_accuracy"].std(),
        "top3_mean": baseline_fold_df["top3_accuracy"].mean(), "top3_std": baseline_fold_df["top3_accuracy"].std(),
        "weighted_f1_mean": baseline_fold_df["weighted_f1"].mean(), "weighted_f1_std": baseline_fold_df["weighted_f1"].std(),
        "top1_drop_from_baseline": 0.0, "top3_drop_from_baseline": 0.0, "weighted_f1_drop_from_baseline": 0.0,
        "note": f"{args.n_folds}-fold mean",
    }]

    baseline_top1_mean = baseline_fold_df["top1_accuracy"].mean()
    baseline_top3_mean = baseline_fold_df["top3_accuracy"].mean()
    baseline_f1_mean = baseline_fold_df["weighted_f1"].mean()


    modality_names = list(MODALITY_PREFIXES.keys())
    combos = list(itertools.combinations(modality_names, 1)) + list(itertools.combinations(modality_names, 2))

    for combo in combos:
        condition_label = "+".join(combo) + " removed"
        combo_tag = "no_" + "_".join(combo)
        print(f"=== Ablation: {condition_label} ===")

        remove_mask = np.zeros(len(feature_names), dtype=bool)
        for modality in combo:
            remove_mask |= identify_modality_columns(feature_names, modality)
        keep_mask_ablation = ~remove_mask

        fold_df = run_ablation(combo_tag, keep_mask_ablation, X, y, folds, args.n_folds,
                                n_classes, label_encoder, args.classifier_dir, output_dir)
        if not fold_df.empty:
            ablated_top1_mean = fold_df["top1_accuracy"].mean()
            ablated_top3_mean = fold_df["top3_accuracy"].mean()
            ablated_f1_mean = fold_df["weighted_f1"].mean()


            per_fold_drop = baseline_fold_df[["fold", "top1_accuracy", "top3_accuracy", "weighted_f1"]].merge(
                fold_df[["fold", "top1_accuracy", "top3_accuracy", "weighted_f1"]], on="fold",
                suffixes=("_baseline", "_ablated")
            )
            per_fold_drop["top1_drop"] = per_fold_drop["top1_accuracy_baseline"] - per_fold_drop["top1_accuracy_ablated"]
            per_fold_drop["top3_drop"] = per_fold_drop["top3_accuracy_baseline"] - per_fold_drop["top3_accuracy_ablated"]
            per_fold_drop["weighted_f1_drop"] = per_fold_drop["weighted_f1_baseline"] - per_fold_drop["weighted_f1_ablated"]
            per_fold_drop.to_csv(output_dir / f"ablation_{combo_tag}_drop_per_fold.csv", index=False)

            summary_rows.append({
                "condition": condition_label,
                "n_features": int(keep_mask_ablation.sum()),
                "top1_mean": ablated_top1_mean, "top1_std": fold_df["top1_accuracy"].std(),
                "top3_mean": ablated_top3_mean, "top3_std": fold_df["top3_accuracy"].std(),
                "weighted_f1_mean": ablated_f1_mean, "weighted_f1_std": fold_df["weighted_f1"].std(),
                "top1_drop_from_baseline": baseline_top1_mean - ablated_top1_mean,
                "top3_drop_from_baseline": baseline_top3_mean - ablated_top3_mean,
                "weighted_f1_drop_from_baseline": baseline_f1_mean - ablated_f1_mean,
                "note": f"{args.n_folds}-fold mean",
            })
        print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    print("=== SUMMARY (including drop from baseline) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {output_dir}/ablation_summary.csv")
    print(f"Per-fold drop CSVs saved as {output_dir}/ablation_no_<modality>_drop_per_fold.csv")

    plot_ablation_figure(
        output_dir / "ablation_summary.csv",
        output_dir / "ablation_figure.png",
    )


if __name__ == "__main__":
    main()