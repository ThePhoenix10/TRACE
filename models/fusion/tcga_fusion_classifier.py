import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.special import softmax as scipy_softmax
from sklearn.preprocessing import LabelEncoder

from models.histology.tcga_histology_classifier import (
    BagDataset, build_model as build_mil_model, load_barcode_to_fold,
    load_barcode_to_tumor_origin, load_case_id_subset,
)
from models.genomics.tcga_genomic_classifier import (
    compute_ranking_metrics, get_full_width_scores,
    load_matrix, load_site_fold_assignment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fuse_mil_mutation")


def get_mil_fold_probabilities(args, fold_idx):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    case_id_subset = load_case_id_subset(Path(args.patient_subset_file)) if args.patient_subset_file else None
    barcode_to_tumor_origin = load_barcode_to_tumor_origin(
        Path(args.patient_cohort_csv), Path(args.case_id_barcode_map), case_id_subset
    )
    barcode_to_fold = load_barcode_to_fold(Path(args.site_fold_csv), Path(args.case_id_barcode_map))

    embeddings_dir = Path(args.embeddings_dir)
    h5_paths, tumor_origins, barcodes = [], [], []
    for h5_path in sorted(embeddings_dir.glob("*.h5")):
        barcode = h5_path.stem
        tumor_origin = barcode_to_tumor_origin.get(barcode)
        if tumor_origin is not None and barcode in barcode_to_fold:
            h5_paths.append(h5_path)
            tumor_origins.append(tumor_origin)
            barcodes.append(barcode)

    class_counts = pd.Series(tumor_origins).value_counts()
    small_classes = class_counts[class_counts < args.min_class_size].index.tolist()
    if small_classes:
        keep_mask = ~pd.Series(tumor_origins).isin(small_classes).to_numpy()
        h5_paths = [p for p, keep in zip(h5_paths, keep_mask) if keep]
        tumor_origins = [c for c, keep in zip(tumor_origins, keep_mask) if keep]
        barcodes = [b for b, keep in zip(barcodes, keep_mask) if keep]

    mil_label_encoder = LabelEncoder()
    y = mil_label_encoder.fit_transform(tumor_origins)
    n_classes = len(mil_label_encoder.classes_)
    folds = np.array([barcode_to_fold[b] for b in barcodes])
    dataset = BagDataset(h5_paths, y)

    idx_test = np.where(folds == fold_idx)[0]

    params_path = Path(args.mil_model_dir) / f"{args.mil_model_name}_fold{fold_idx}_best_params.json"
    checkpoint_path = Path(args.mil_model_dir) / f"{args.mil_model_name}_fold{fold_idx}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No MIL checkpoint at {checkpoint_path} -- has this fold finished training?")
    hidden_dim = json.loads(params_path.read_text())["hidden_dim"] if params_path.exists() else 256

    model = build_mil_model(args.mil_model_name, n_classes, hidden_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_probs = {}
    with torch.no_grad():
        for idx in idx_test:
            bag, _ = dataset[idx]
            bag = bag.to(device)
            if args.mil_model_name == "clam":
                logits, _, _ = model(bag)
            else:
                logits, _ = model(bag)
            probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
            all_probs[barcodes[idx]] = probs


    case_id_barcode_map = pd.read_csv(args.case_id_barcode_map)
    barcode_to_case_id = dict(zip(case_id_barcode_map["barcode"], case_id_barcode_map["case_id"]))
    probs_by_case_id = {
        barcode_to_case_id[b]: p for b, p in all_probs.items() if b in barcode_to_case_id
    }

    return probs_by_case_id, list(mil_label_encoder.classes_)


def get_mutation_fold_probabilities(args, fold_idx):


    X, tumor_origins_all, case_ids_all, _ = load_matrix(args.mutation_matrix_prefix)
    folds = load_site_fold_assignment(Path(args.site_fold_csv), case_ids_all)
    test_mask = folds == fold_idx
    X_test = X[test_mask]
    case_ids_test = case_ids_all[test_mask]

    label_encoder = joblib.load(Path(args.mutation_model_dir) / "label_encoder.joblib")
    n_classes = len(label_encoder.classes_)

    checkpoint_path = Path(args.mutation_model_dir) / f"{args.mutation_model_name}_fold{fold_idx}_model.joblib"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No mutation model checkpoint at {checkpoint_path} -- "
                                 f"has this fold finished training?")
    artifact = joblib.load(checkpoint_path)


    model = artifact["model"]
    scaler = artifact["scaler"]
    X_test_input = scaler.transform(X_test) if scaler is not None else X_test
    raw_scores = get_full_width_scores(model, X_test_input, n_classes)
    if not hasattr(model, "predict_proba"):


        raw_scores = scipy_softmax(raw_scores, axis=1)

    probs_by_case_id = {cid: raw_scores[i] for i, cid in enumerate(case_ids_test)}
    return probs_by_case_id, list(label_encoder.classes_)


def align_to_common_classes(probs_by_case_id, model_classes, common_classes):


    class_to_common_idx = {c: i for i, c in enumerate(common_classes)}
    model_idx_to_common_idx = {
        i: class_to_common_idx[c] for i, c in enumerate(model_classes) if c in class_to_common_idx
    }

    aligned = {}
    for case_id, probs in probs_by_case_id.items():
        vec = np.zeros(len(common_classes))
        for model_idx, common_idx in model_idx_to_common_idx.items():
            vec[common_idx] = probs[model_idx]
        aligned[case_id] = vec
    return aligned


def save_confusion_matrix_counts(y_true, preds, common_classes, output_dir, fold_idx, label):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, preds, labels=np.arange(len(common_classes)))
    cm_df = pd.DataFrame(cm, index=common_classes, columns=common_classes)
    cm_df.index.name = "true"
    cm_df.columns.name = "predicted"

    csv_path = output_dir / f"fold{fold_idx}_{label}_confusion_matrix.csv"
    cm_df.to_csv(csv_path)
    log.info(f"Saved {label} confusion matrix counts to {csv_path}")


def load_aligned_fold_data(args, fold_idx, common_classes=None):


    mil_probs, mil_classes = get_mil_fold_probabilities(args, fold_idx)
    mut_probs, mut_classes = get_mutation_fold_probabilities(args, fold_idx)

    common_case_ids = sorted(set(mil_probs) & set(mut_probs))
    if common_classes is None:
        common_classes = sorted(set(mil_classes) | set(mut_classes))

    mil_aligned = align_to_common_classes(mil_probs, mil_classes, common_classes)
    mut_aligned = align_to_common_classes(mut_probs, mut_classes, common_classes)

    patient_cohort = pd.read_csv(args.patient_cohort_csv)
    case_id_to_true_label = dict(zip(patient_cohort["case_id"], patient_cohort["tumor_origin"]))
    class_to_idx = {c: i for i, c in enumerate(common_classes)}

    case_ids, mil_rows, mut_rows, y_true = [], [], [], []
    for cid in common_case_ids:
        true_label = case_id_to_true_label.get(cid)
        if true_label is None or true_label not in class_to_idx:
            continue
        case_ids.append(cid)
        mil_rows.append(mil_aligned[cid])
        mut_rows.append(mut_aligned[cid])
        y_true.append(class_to_idx[true_label])

    return (case_ids, np.array(mil_rows), np.array(mut_rows), np.array(y_true), common_classes)


def compute_fusion_scores(method, mil_matrix, mut_matrix):


    if method == "mean":
        return (mil_matrix + mut_matrix) / 2
    elif method == "max":
        return np.maximum(mil_matrix, mut_matrix)
    elif method == "multiply":
        product = mil_matrix * mut_matrix
        row_sums = product.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return product / row_sums
    raise ValueError(f"compute_fusion_scores doesn't handle method={method} -- use the dedicated path instead")


def compute_vote_predictions(mil_matrix, mut_matrix):


    mil_preds = mil_matrix.argmax(axis=1)
    mut_preds = mut_matrix.argmax(axis=1)
    mil_conf = mil_matrix.max(axis=1)
    mut_conf = mut_matrix.max(axis=1)

    final_preds = np.where(mil_preds == mut_preds, mil_preds,
                            np.where(mil_conf >= mut_conf, mil_preds, mut_preds))
    return final_preds


def run_stacking_fusion(args, test_fold_idx, dropout_prob=0.0):


    from sklearn.linear_model import LogisticRegression


    _, _, _, _, common_classes = load_aligned_fold_data(args, test_fold_idx)

    train_mil, train_mut, train_y = [], [], []
    for fold_idx in range(args.n_folds):
        if fold_idx == test_fold_idx:
            continue
        _, mil_matrix, mut_matrix, y_true, _ = load_aligned_fold_data(args, fold_idx, common_classes)
        train_mil.append(mil_matrix)
        train_mut.append(mut_matrix)
        train_y.append(y_true)

    train_mil = np.concatenate(train_mil, axis=0)
    train_mut = np.concatenate(train_mut, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    if dropout_prob > 0:
        rng = np.random.default_rng(0)
        drop_mask = rng.random(len(train_y)) < dropout_prob
        drop_which = rng.integers(0, 2, size=len(train_y))
        train_mil = train_mil.copy()
        train_mut = train_mut.copy()
        train_mil[drop_mask & (drop_which == 0)] = 0
        train_mut[drop_mask & (drop_which == 1)] = 0
        log.info(f"Modality dropout: zeroed one modality's features on "
                 f"{drop_mask.sum()}/{len(train_y)} training rows")

    train_X = np.concatenate([train_mil, train_mut], axis=1)
    log.info(f"Stacking meta-classifier: {train_X.shape[0]} training rows "
             f"(from {args.n_folds - 1} other folds), {train_X.shape[1]} features")

    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta_model.fit(train_X, train_y)

    case_ids, test_mil, test_mut, test_y, _ = load_aligned_fold_data(args, test_fold_idx, common_classes)
    test_X = np.concatenate([test_mil, test_mut], axis=1)
    test_scores = meta_model.predict_proba(test_X)


    full_scores = np.zeros((test_scores.shape[0], len(common_classes)))
    for i, cls_idx in enumerate(meta_model.classes_):
        full_scores[:, cls_idx] = test_scores[:, i]

    return case_ids, full_scores, test_y, common_classes


def process_one_fold(args, fold_idx, output_dir):


    method_tag = args.fusion_method

    if args.fusion_method == "stack":
        method_tag = f"stack_dropout{args.dropout_prob:.2f}" if args.dropout_prob > 0 else "stack"
        log.info(f"[{method_tag}] Training stacking meta-classifier on folds other than {fold_idx}...")
        case_ids, fused_scores, y_true, common_classes = run_stacking_fusion(
            args, fold_idx, dropout_prob=args.dropout_prob
        )


        _, mil_scores, mut_scores, _, _ = load_aligned_fold_data(args, fold_idx, common_classes)
        only_mil = only_mut = set()

    else:
        log.info(f"Loading MIL ({args.mil_model_name}) fold {fold_idx} probabilities...")
        mil_probs, mil_classes = get_mil_fold_probabilities(args, fold_idx)
        log.info(f"MIL: {len(mil_probs)} test patients, {len(mil_classes)} classes")

        log.info(f"Loading mutation ({args.mutation_model_name}) fold {fold_idx} probabilities...")
        mut_probs, mut_classes = get_mutation_fold_probabilities(args, fold_idx)
        log.info(f"Mutation: {len(mut_probs)} test patients, {len(mut_classes)} classes")

        common_case_ids = sorted(set(mil_probs) & set(mut_probs))
        only_mil = set(mil_probs) - set(mut_probs)
        only_mut = set(mut_probs) - set(mil_probs)
        log.info(f"Patients in BOTH (fusable): {len(common_case_ids)}")
        if only_mil:
            log.warning(f"{len(only_mil)} patients have a MIL prediction but no mutation prediction -- excluded")
        if only_mut:
            log.warning(f"{len(only_mut)} patients have a mutation prediction but no MIL prediction -- excluded")

        if not common_case_ids:
            log.error(f"Fold {fold_idx}: no patients present in both test sets -- nothing to fuse. Skipping fold.")
            return None, None

        common_classes = sorted(set(mil_classes) | set(mut_classes))
        if set(mil_classes) != set(mut_classes):
            log.warning(f"MIL and mutation classifiers were trained on DIFFERENT class sets "
                        f"(MIL: {len(mil_classes)}, mutation: {len(mut_classes)}, union: {len(common_classes)}) "
                        f"-- likely a --min-class-size mismatch between the two pipelines.")

        mil_aligned = align_to_common_classes(mil_probs, mil_classes, common_classes)
        mut_aligned = align_to_common_classes(mut_probs, mut_classes, common_classes)

        patient_cohort = pd.read_csv(args.patient_cohort_csv)
        case_id_to_true_label = dict(zip(patient_cohort["case_id"], patient_cohort["tumor_origin"]))
        class_to_idx = {c: i for i, c in enumerate(common_classes)}

        case_ids, y_true, mil_scores, mut_scores = [], [], [], []
        for cid in common_case_ids:
            true_label = case_id_to_true_label.get(cid)
            if true_label is None or true_label not in class_to_idx:
                continue
            case_ids.append(cid)
            y_true.append(class_to_idx[true_label])
            mil_scores.append(mil_aligned[cid])
            mut_scores.append(mut_aligned[cid])

        y_true = np.array(y_true)
        mil_scores = np.array(mil_scores)
        mut_scores = np.array(mut_scores)

        if args.fusion_method == "vote":
            fused_preds_direct = compute_vote_predictions(mil_scores, mut_scores)
            fused_scores = None
        else:
            fused_scores = compute_fusion_scores(args.fusion_method, mil_scores, mut_scores)

    n_classes = len(common_classes)
    mil_preds = mil_scores.argmax(axis=1)
    mut_preds = mut_scores.argmax(axis=1)
    fused_preds = fused_preds_direct if args.fusion_method == "vote" else fused_scores.argmax(axis=1)

    mil_metrics = compute_ranking_metrics(y_true, mil_scores, mil_preds, n_classes)
    mut_metrics = compute_ranking_metrics(y_true, mut_scores, mut_preds, n_classes)
    fused_metrics = compute_ranking_metrics(
        y_true, fused_scores if fused_scores is not None else mil_scores, fused_preds, n_classes
    )


    log.info(f"Fold {fold_idx} [{method_tag}] -- MIL only:       top1={mil_metrics['top1_accuracy']:.4f}, "
             f"weighted_F1={mil_metrics['weighted_f1']:.4f}")
    log.info(f"Fold {fold_idx} [{method_tag}] -- Mutation only:  top1={mut_metrics['top1_accuracy']:.4f}, "
             f"weighted_F1={mut_metrics['weighted_f1']:.4f}")
    log.info(f"Fold {fold_idx} [{method_tag}] -- FUSED:          top1={fused_metrics['top1_accuracy']:.4f}, "
             f"weighted_F1={fused_metrics['weighted_f1']:.4f}")

    save_confusion_matrix_counts(
        y_true,
        fused_preds,
        common_classes,
        output_dir,
        fold_idx,
        f"{method_tag}_fused",
    )

    report_path = output_dir / f"fold{fold_idx}_{method_tag}_fusion_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Fold {fold_idx} -- fusion method: {method_tag}\n")
        f.write(f"MIL ({args.mil_model_name}) + Mutation ({args.mutation_model_name})\n")
        if only_mil or only_mut:
            f.write(f"MIL-only excluded: {len(only_mil)}; mutation-only excluded: {len(only_mut)}\n")
        f.write(f"Patients evaluated: {len(y_true)}, common classes: {n_classes}\n\n")
        for name, m in [("MIL only", mil_metrics), ("Mutation only", mut_metrics), ("FUSED", fused_metrics)]:
            f.write(f"{name}:\n")
            f.write(f"  top-1 accuracy: {m['top1_accuracy']:.4f}\n")
            f.write(f"  top-3 accuracy: {m['top3_accuracy']:.4f}\n")
            f.write(f"  top-5 accuracy: {m['top5_accuracy']:.4f}\n")
            f.write(f"  weighted F1:    {m['weighted_f1']:.4f}\n")
            f.write(f"  macro F1:       {m['macro_f1']:.4f}\n\n")
    log.info(f"Saved report to {report_path}")

    fold_row = {
        "fold": fold_idx, "fusion_method": method_tag, "n_patients": len(y_true), "n_classes": n_classes,
        **{f"mil_{k}": v for k, v in mil_metrics.items()},
        **{f"mutation_{k}": v for k, v in mut_metrics.items()},
        **{f"fused_{k}": v for k, v in fused_metrics.items()},
    }
    return method_tag, fold_row


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


def generate_combined_confusion_matrix(
    results_dir,
    label,
    n_folds,
    output,
):
    results_dir = Path(results_dir)
    combined_cm = None
    n_found = 0

    for fold_idx in range(n_folds):
        csv_path = (
            results_dir
            / f"fold{fold_idx}_{label}_confusion_matrix.csv"
        )

        if not csv_path.exists():
            log.warning(f"{csv_path} not found -- skipping this fold")
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
            f"'{label}' in {results_dir}"
        )

    combined_cm = (
        combined_cm
        .fillna(0)
        .astype(int)
    )

    log.info(
        f"Aggregated {n_found}/{n_folds} folds' confusion matrices"
    )

    log.info(
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

    positive_values = cm_values[cm_values > 0]
    if positive_values.size == 0:
        plt.close(fig)
        raise ValueError("Combined confusion matrix contains no positive counts")

    im = ax.imshow(
        cm_values,
        interpolation="nearest",
        cmap=custom_cmap,
        norm=LogNorm(
            vmin=max(
                positive_values.min(),
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

    ax.set_title(
        "Confusion Matrix - TRACE",
        fontsize=25,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    output_path = Path(output)

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

    log.info(
        f"Saved combined confusion matrix to {output_path}"
    )

    combined_csv_path = output_path.with_suffix(
        ".csv"
    )

    combined_cm.to_csv(
        combined_csv_path
    )

    log.info(
        f"Saved combined confusion matrix raw counts to "
        f"{combined_csv_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fuse MIL and mutation classifier predictions, per fold")
    parser.add_argument("--fold", type=int, default=None,
                         help="Run just this one fold. If OMITTED, runs every fold from 0 to "
                              "--n-folds-1 in sequence within this one invocation, and saves a "
                              "mean+/-std summary across all of them at the end.")
    parser.add_argument("--fusion-method", required=True,
                         choices=["mean", "max", "multiply", "vote", "stack"],
                         help="mean/max/multiply/vote need only this fold's saved predictions. "
                              "stack trains a small meta-classifier on every OTHER fold's "
                              "predictions (leakage-free) -- slower, needs all 5 folds' "
                              "MIL+mutation predictions available.")
    parser.add_argument("--dropout-prob", type=float, default=0.0,
                         help="For --fusion-method stack only: probability of zeroing one modality's "
                              "features per TRAINING row (never applied to the evaluated test fold), "
                              "so the meta-classifier doesn't over-rely on either modality")
    parser.add_argument("--n-folds", type=int, default=5, help="Needed by --fusion-method stack and by --fold omission")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--patient-cohort-csv", required=True)
    parser.add_argument("--case-id-barcode-map", required=True)
    parser.add_argument("--site-fold-csv", required=True)
    parser.add_argument("--patient-subset-file", default=None)
    parser.add_argument("--mil-model-dir", required=True)
    parser.add_argument("--mil-model-name", required=True, choices=["abmil", "clam", "transmil"])
    parser.add_argument("--mutation-matrix-prefix", required=True)
    parser.add_argument("--mutation-model-dir", required=True)
    parser.add_argument("--mutation-model-name", required=True)
    parser.add_argument("--min-class-size", type=int, default=3,
                         help="MUST match the value used for the original MIL training run")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_indices = [args.fold] if args.fold is not None else list(range(args.n_folds))
    method_tag = None
    fold_rows = []

    for fold_idx in fold_indices:
        tag, row = process_one_fold(args, fold_idx, output_dir)
        if row is None:
            continue
        method_tag = tag
        fold_rows.append(row)

    if not fold_rows:
        log.error("No folds produced a result -- nothing to summarize.")
        return


    csv_path = output_dir / f"fusion_cv_folds_{method_tag}.csv"
    new_rows_df = pd.DataFrame(fold_rows)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["fold"].isin(new_rows_df["fold"])]
        combined = pd.concat([existing, new_rows_df], ignore_index=True).sort_values("fold")
    else:
        combined = new_rows_df
    combined.to_csv(csv_path, index=False)
    log.info(f"Updated {csv_path} ({len(combined)} fold(s) total on disk)")


    metric_cols = [c for c in combined.columns if c.startswith(("mil_", "mutation_", "fused_"))]
    summary = {"fusion_method": method_tag, "n_folds": len(combined)}
    for col in metric_cols:
        summary[f"{col}_mean"] = combined[col].mean()
        summary[f"{col}_std"] = combined[col].std()

    summary_path = output_dir / f"fusion_summary_{method_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"=== [{method_tag}] SUMMARY across {len(combined)} fold(s) ===")
    log.info(f"  MIL only:      top1={summary['mil_top1_accuracy_mean']:.4f}+/-{summary['mil_top1_accuracy_std']:.4f}, "
             f"weighted_F1={summary['mil_weighted_f1_mean']:.4f}+/-{summary['mil_weighted_f1_std']:.4f}")
    log.info(f"  Mutation only: top1={summary['mutation_top1_accuracy_mean']:.4f}+/-{summary['mutation_top1_accuracy_std']:.4f}, "
             f"weighted_F1={summary['mutation_weighted_f1_mean']:.4f}+/-{summary['mutation_weighted_f1_std']:.4f}")
    log.info(f"  FUSED:         top1={summary['fused_top1_accuracy_mean']:.4f}+/-{summary['fused_top1_accuracy_std']:.4f}, "
             f"weighted_F1={summary['fused_weighted_f1_mean']:.4f}+/-{summary['fused_weighted_f1_std']:.4f}")
    log.info(f"Saved summary to {summary_path}")

    combined_cm_output = output_dir / f"combined_confusion_matrix_{method_tag}.png"
    generate_combined_confusion_matrix(
        results_dir=output_dir,
        label=f"{method_tag}_fused",
        n_folds=args.n_folds,
        output=combined_cm_output,
    )


if __name__ == "__main__":
    main()