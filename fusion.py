#!/usr/bin/env python3
"""
fuse_mil_mutation.py

Late-fusion of one MIL (histology) fold checkpoint with one mutation
classifier fold checkpoint, for the SAME outer fold -- combines each
model's predicted probability distribution (simple average) into a
single fused prediction per patient, and reports MIL-only,
mutation-only, AND fused performance side by side so the actual benefit
of fusion is directly visible, not just asserted.

Run once per fold as each becomes available (--fold 0, then --fold 1,
etc.) -- results accumulate into fusion_cv_folds.csv, one row per fold,
resumable/appendable the same way as every other CV script in this
project. You don't need all 5 folds done on both sides to start -- just
run whichever folds are ready on both the MIL and mutation side.

THREE ALIGNMENT PROBLEMS THIS SCRIPT HANDLES EXPLICITLY
    1. Different patient identifiers: MIL's .h5 files are keyed by
       barcode, the mutation classifier's fused table is keyed by
       case_id. Bridged via case_id_barcode_map.csv. Only patients
       present in BOTH pipelines' test sets for this fold get fused --
       anyone missing from either side is reported and excluded, not
       silently dropped.
    2. Different label encodings: MIL's label_encoder and the mutation
       classifier's label_encoder were each fit independently and may
       not use the same class->index mapping (or even the exact same
       class SET, if --min-class-size differed between the two runs).
       This script aligns both models' probability vectors by CLASS
       NAME onto a shared class list (the union of both), never by raw
       index -- a class either model didn't see gets a 0 probability
       column for that model rather than misaligning indices.
    3. Different score types: XGBoost/LightGBM/logreg/MLP produce real
       calibrated-ish probabilities (predict_proba / softmax), but
       linear_svm/sgd only have decision_function (arbitrary-scale
       scores, not probabilities). Averaging those directly with real
       probabilities would be methodologically wrong -- this script
       applies softmax to decision_function scores and to MLP's raw
       logits first, so every model's output is a genuine 0-1
       probability distribution before fusion.

USAGE
    python fuse_mil_mutation.py \
        --fold 0 \
        --embeddings-dir /workspace/matched_cohort_uni/embeddings \
        --patient-cohort-csv /workspace/splits/patient_cohort.csv \
        --case-id-barcode-map /workspace/splits/case_id_barcode_map.csv \
        --site-fold-csv /workspace/splits/site_stratified_5fold.csv \
        --patient-subset-file /workspace/mutation_tables/fused/patient_fused_matrix_rows.txt \
        --mil-model-dir /workspace/mil_models_abmil \
        --mil-model-name abmil \
        --mutation-matrix-prefix /workspace/mutation_tables/fused/patient_fused_matrix \
        --mutation-model-dir /workspace/mutation_tables/fused/classifier_cv \
        --mutation-model-name xgboost \
        --output-dir /workspace/fusion_results

REQUIREMENTS
    pip install --break-system-packages torch h5py scikit-learn pandas \
        scipy joblib xgboost lightgbm

OUTPUT (under --output-dir)
    fold<N>_fusion_report.txt   MIL-only / mutation-only / fused metrics side by side
    fusion_cv_folds.csv         one row per fold processed so far (appended across runs)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax as scipy_softmax
from sklearn.preprocessing import LabelEncoder

from train_mil_models import (
    BagDataset, build_model as build_mil_model, load_barcode_to_fold,
    load_barcode_to_tumor_origin, load_case_id_subset,
)
from mutClassifier import (
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
    """
    Reconstructs the MIL fold's exact test set (same logic as
    confusion_matrix.py), loads that fold's checkpoint, runs inference,
    and returns {case_id: probability_vector}, plus the MIL class name
    list (in the order that vector's columns correspond to).
    Bridges barcode -> case_id via case_id_barcode_map.csv, since MIL's
    own data is barcode-keyed but the mutation side is case_id-keyed.
    """
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

    # bridge barcode -> case_id for the returned dict's keys
    case_id_barcode_map = pd.read_csv(args.case_id_barcode_map)
    barcode_to_case_id = dict(zip(case_id_barcode_map["barcode"], case_id_barcode_map["case_id"]))
    probs_by_case_id = {
        barcode_to_case_id[b]: p for b, p in all_probs.items() if b in barcode_to_case_id
    }

    return probs_by_case_id, list(mil_label_encoder.classes_)


def get_mutation_fold_probabilities(args, fold_idx):
    """
    Loads the fused mutation matrix, restricts to this fold's test
    patients (via site_stratified_5fold.csv), loads that fold's saved
    classifier artifact, and returns {case_id: probability_vector} in
    genuine 0-1 probability space (softmax applied if the underlying
    model only has decision_function or raw logits -- see module
    docstring point 3), plus the mutation classifier's class name list.
    """
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

    # mutClassifier.py is XGBoost-only now (MLP/logreg/linear_svm/sgd/
    # lightgbm support was removed) -- every saved artifact is the plain
    # sklearn-style {"model":..., "scaler":...} shape, never an MLP
    # checkpoint (which would have a "model_state_dict" key instead).
    model = artifact["model"]
    scaler = artifact["scaler"]
    X_test_input = scaler.transform(X_test) if scaler is not None else X_test
    raw_scores = get_full_width_scores(model, X_test_input, n_classes)
    if not hasattr(model, "predict_proba"):
        # decision_function output isn't a probability distribution --
        # softmax converts it to one so fusion-by-averaging is valid
        raw_scores = scipy_softmax(raw_scores, axis=1)

    probs_by_case_id = {cid: raw_scores[i] for i, cid in enumerate(case_ids_test)}
    return probs_by_case_id, list(label_encoder.classes_)


def align_to_common_classes(probs_by_case_id, model_classes, common_classes):
    """
    Remaps each patient's probability vector from model_classes' column
    order onto common_classes' column order -- any class the model
    didn't have a column for (absent from ITS training set) gets 0
    probability in the aligned vector, not a misaligned value from a
    different class.
    """
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


def save_confusion_matrix(y_true, preds, common_classes, output_dir, fold_idx, label, metrics):
    """
    Saves a confusion matrix (raw counts CSV + heatmap PNG) for one
    prediction set (MIL-only, mutation-only, or fused). Reuses the same
    sklearn confusion_matrix + seaborn heatmap pattern as
    confusionMatrix.py, so figures across the project look consistent.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, preds, labels=np.arange(len(common_classes)))
    cm_df = pd.DataFrame(cm, index=common_classes, columns=common_classes)
    cm_df.index.name = "true"
    cm_df.columns.name = "predicted"

    csv_path = output_dir / f"fold{fold_idx}_{label}_confusion_matrix.csv"
    cm_df.to_csv(csv_path)
    log.info(f"Saved {label} confusion matrix (raw counts) to {csv_path}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Count"})
        ax.set_title(f"Fold {fold_idx} -- {label} -- Confusion Matrix "
                     f"(top1={metrics['top1_accuracy']:.3f}, weighted_F1={metrics['weighted_f1']:.3f})")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        png_path = output_dir / f"fold{fold_idx}_{label}_confusion_matrix.png"
        plt.savefig(png_path, dpi=600)
        plt.close(fig)
        log.info(f"Saved {label} confusion matrix heatmap to {png_path}")
    except ImportError:
        log.warning("matplotlib/seaborn not installed -- skipped PNG heatmap "
                    "(pip install --break-system-packages matplotlib seaborn), CSV still saved above")


def load_aligned_fold_data(args, fold_idx, common_classes=None):
    """
    Loads MIL + mutation probabilities for one fold, aligns both to a
    shared class list, and returns (case_ids, mil_matrix, mut_matrix,
    y_true, common_classes) as plain numpy arrays ready for fusion math.
    If common_classes is given, aligns to THAT list (needed so every
    fold used by stacking shares identical columns); otherwise computes
    it fresh from this fold's own two models' class sets.
    """
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
    """
    Combines two (n_patients, n_classes) probability matrices into one
    fused score matrix, for the methods that operate directly on
    probability vectors (mean/max/multiply). vote and stack are handled
    separately (see main()) since they don't reduce to a single
    per-class score combination rule.
    """
    if method == "mean":
        return (mil_matrix + mut_matrix) / 2
    elif method == "max":
        return np.maximum(mil_matrix, mut_matrix)
    elif method == "multiply":
        product = mil_matrix * mut_matrix
        row_sums = product.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid divide-by-zero if both models gave all-zero rows
        return product / row_sums  # renormalize back to a valid probability distribution
    raise ValueError(f"compute_fusion_scores doesn't handle method={method} -- use the dedicated path instead")


def compute_vote_predictions(mil_matrix, mut_matrix):
    """
    Voting-based late fusion with 2 voters: if MIL and mutation agree on
    the argmax class, use it. If they disagree, break the tie using
    whichever modality is more CONFIDENT in its own prediction (higher
    own max probability) -- a defensible rule with only 2 voters, since
    a literal majority vote is undefined for a 2-way tie.
    """
    mil_preds = mil_matrix.argmax(axis=1)
    mut_preds = mut_matrix.argmax(axis=1)
    mil_conf = mil_matrix.max(axis=1)
    mut_conf = mut_matrix.max(axis=1)

    final_preds = np.where(mil_preds == mut_preds, mil_preds,
                            np.where(mil_conf >= mut_conf, mil_preds, mut_preds))
    return final_preds


def run_stacking_fusion(args, test_fold_idx, dropout_prob=0.0):
    """
    Trains a small LogisticRegression meta-classifier on [mil_probs,
    mut_probs] concatenated as features, using every OTHER fold's data
    as training data -- test_fold_idx's data is NEVER used for training,
    same nested-CV principle used everywhere else in this project. This
    is what makes stacking leakage-free rather than just fitting on the
    same data being evaluated.

    dropout_prob, if > 0, randomly zeroes out one modality's entire
    feature block (all of MIL's columns, or all of mutation's columns)
    for that fraction of TRAINING rows -- forces the meta-classifier to
    not over-rely on whichever modality happens to be stronger, so it
    still performs reasonably on patients where one modality is missing
    or unreliable. Never applied to the actual test fold being evaluated.
    """
    from sklearn.linear_model import LogisticRegression

    # first pass: establish the common class list from the TEST fold itself
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
        drop_which = rng.integers(0, 2, size=len(train_y))  # 0 = drop MIL, 1 = drop mutation
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

    # predict_proba's columns follow meta_model.classes_ order, which may
    # be a SUBSET of common_classes if some class was entirely absent
    # from the training folds -- remap back to the full common_classes
    # width so downstream code (confusion matrix, metrics) sees a
    # consistent column count across every fusion method
    full_scores = np.zeros((test_scores.shape[0], len(common_classes)))
    for i, cls_idx in enumerate(meta_model.classes_):
        full_scores[:, cls_idx] = test_scores[:, i]

    return case_ids, full_scores, test_y, common_classes


def process_one_fold(args, fold_idx, output_dir):
    """
    Runs one fold's fusion (whichever --fusion-method was chosen) end to
    end: loads/aligns predictions, computes fused scores, saves per-fold
    confusion matrices + report, and returns (method_tag, fold_row) for
    the caller to accumulate into an all-folds summary.
    """
    method_tag = args.fusion_method

    if args.fusion_method == "stack":
        method_tag = f"stack_dropout{args.dropout_prob:.2f}" if args.dropout_prob > 0 else "stack"
        log.info(f"[{method_tag}] Training stacking meta-classifier on folds other than {fold_idx}...")
        case_ids, fused_scores, y_true, common_classes = run_stacking_fusion(
            args, fold_idx, dropout_prob=args.dropout_prob
        )
        # still load this fold's raw MIL/mutation scores too, purely for
        # the MIL-only / mutation-only comparison rows in the report
        _, mil_scores, mut_scores, _, _ = load_aligned_fold_data(args, fold_idx, common_classes)
        only_mil = only_mut = set()  # already resolved inside load_aligned_fold_data's inner join

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
            fused_scores = None  # vote produces hard predictions directly, no probability matrix
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
    # NOTE: for "vote", fused_scores is None (no probability matrix, only
    # hard predictions), so top-3/top-5 accuracy in fused_metrics is NOT
    # meaningful for vote -- only top-1/weighted_f1/macro_f1 (which use
    # fused_preds directly, not the score matrix) are valid for that method

    log.info(f"Fold {fold_idx} [{method_tag}] -- MIL only:       top1={mil_metrics['top1_accuracy']:.4f}, "
             f"weighted_F1={mil_metrics['weighted_f1']:.4f}")
    log.info(f"Fold {fold_idx} [{method_tag}] -- Mutation only:  top1={mut_metrics['top1_accuracy']:.4f}, "
             f"weighted_F1={mut_metrics['weighted_f1']:.4f}")
    log.info(f"Fold {fold_idx} [{method_tag}] -- FUSED:          top1={fused_metrics['top1_accuracy']:.4f}, "
             f"weighted_F1={fused_metrics['weighted_f1']:.4f}")

    save_confusion_matrix(y_true, mil_preds, common_classes, output_dir, fold_idx, f"{method_tag}_mil", mil_metrics)
    save_confusion_matrix(y_true, mut_preds, common_classes, output_dir, fold_idx,
                           f"{method_tag}_mutation", mut_metrics)
    save_confusion_matrix(y_true, fused_preds, common_classes, output_dir, fold_idx,
                           f"{method_tag}_fused", fused_metrics)

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

    # method-specific CSV filename -- lets multiple fusion methods run in
    # separate terminals concurrently without one overwriting another's
    # summary file. Accumulates one row per fold, replacing a fold's row
    # if that fold is rerun, so this stays correct across repeated runs
    # (e.g. running --fold 0 alone earlier, then all folds later).
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

    # mean +/- std summary across every fold NOW on disk for this method
    # (not just the folds processed in THIS run -- so running --fold 4
    # after an earlier --fold 0..3 run still produces a correct 5-fold summary)
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


if __name__ == "__main__":
    main()