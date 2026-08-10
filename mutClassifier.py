#!/usr/bin/env python3
"""
mutClassifier.py

Tunes XGBoost for tumor-origin prediction from the sparse mutation
matrix, using Optuna for hyperparameter search, nested 5-fold
site-stratified CV, and automatically generates per-fold training/
validation loss and accuracy curve plots.

XGBoost-only: this used to compare six model types (logreg, linear_svm,
sgd, xgboost, lightgbm, mlp) -- XGBoost was consistently the strongest
performer, so every other model type has been removed for a leaner,
single-purpose script. If you want any of the others back, they'd need
to be reintroduced manually (sample_params, build_sklearn_model,
fit_sklearn_model, fit_one_split all previously had per-model branches).

TUNING + EVALUATION: NESTED SITE-STRATIFIED 5-FOLD CV
    Hyperparameter tuning happens SEPARATELY for each of the 5 outer
    folds (from --site-fold-csv, e.g. site_stratified_5fold.csv), using
    ONLY that fold's four training folds as the tuning pool -- the outer
    test fold is never touched until final evaluation. This is nested
    CV: Optuna's trial budget (--n-trials) runs once PER outer fold,
    each trial evaluated on a stratified train/validation split carved
    out of that fold's training pool. The optimization target is plain
    VALIDATION ACCURACY.

    Because the outer folds are grouped by Tissue Source Site (the
    submitting institution), no site's patients ever appear in both the
    training pool (tuning + final fit) and the test fold for that
    iteration -- this controls for site/scanner/staining batch effects,
    a known confound in TCGA-based studies. The reported result is the
    MEAN +/- STD across the 5 outer folds' held-out test performance.

    A trial that fails (e.g. a rare class breaks the inner
    stratification) is caught and scored as a failure rather than
    crashing the whole study.

TRAINING/VALIDATION CURVES (generated automatically)
    Every fold's per-round train/val log loss and accuracy (captured
    directly from XGBoost's own eval history during fitting -- not
    re-derived) is saved to <model>_fold<N>_training_curve.csv, and
    after all folds finish, 4 plots are generated automatically:
        train_loss_curve.png / train_accuracy_curve.png
        val_loss_curve.png / val_accuracy_curve.png
    each with one line per fold, using the project's blue color palette.

USAGE
    python mutClassifier.py \
        --matrix-prefix /workspace/mutation_tables/fused/patient_fused_matrix \
        --site-fold-csv /workspace/splits/site_stratified_5fold.csv \
        --output-dir /workspace/mutation_tables/fused/classifier_cv \
        --n-trials 30

REQUIREMENTS
    pip install --break-system-packages scikit-learn joblib optuna \
        xgboost matplotlib pandas

OUTPUT (under --output-dir)
    model_comparison.csv          mean +/- std CV accuracy/macro-F1
    xgboost_model.joblib           model artifact from the BEST-performing fold
                                    (representative checkpoint, same as
                                    xgboost_fold<N>_model.joblib for whichever
                                    N had the best weighted F1)
    xgboost_fold<N>_model.joblib    EVERY fold's own model, saved separately
                                    (N = 0..n_folds-1) -- load any of these
                                    directly with joblib.load() for graphing/
                                    analysis without retraining anything
    xgboost_fold<N>_best_params.json  best hyperparameters found FOR THIS FOLD
                                    specifically (tuned only on that fold's
                                    training pool -- may differ fold to fold)
    xgboost_fold<N>_training_curve.csv  per-round train/val loss+accuracy
    xgboost_cv_folds.csv           per-fold accuracy/macro-F1, plus that
                                    fold's own tuning validation accuracy
    xgboost_fold<N>_classification_report.txt  per-class metrics for fold N
    optuna_xgboost_outerfold<N>.db  resumable Optuna study storage, ONE PER
                                    OUTER FOLD (nested CV tunes separately per
                                    fold) -- rerunning with the same
                                    --output-dir continues each fold's search
                                    instead of restarting it
    train_loss_curve.png / train_accuracy_curve.png
    val_loss_curve.png / val_accuracy_curve.png  one line per fold, blue palette
    label_encoder.joblib
"""

# MUST happen before numpy/scipy/sklearn import -- these libraries read
# OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS at import time to
# decide how many CPU threads their underlying BLAS backend is allowed to
# use, and default to grabbing EVERY core on the machine if unset. This
# was directly observed causing a single logreg fit to spawn 129 threads
# on a 128-core pod -- effectively starving every other concurrently
# running process (including GPU-bound MIL training, which still needs
# real CPU cycles to load each tile bag) of any CPU time at all, even
# though logreg itself has no real need for that many threads on a
# problem this size. Capped at 4 -- fast enough for any single model fit
# here, while leaving real CPU headroom for whatever else is running
# alongside this script. Override by setting these yourself in the shell
# BEFORE running this script, if you want a different cap.
import os
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "4")

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mutClassifier")



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def load_matrix(matrix_prefix: str):
    """
    Loads a table given its path prefix -- the same convention used by
    fuse_tables.py: <prefix>.npz, <prefix>_rows.txt, <prefix>_columns.txt.
    This means the same classifier can point at the mutation-only table,
    the signatures-only table, or a fused table combining both (or more,
    as you add feature types) -- just by changing --matrix-prefix, no
    code changes needed. Examples:
        --matrix-prefix /workspace/mutation_tables/patient_mutation_matrix
        --matrix-prefix /workspace/mutation_tables/patient_signature_matrix
        --matrix-prefix /workspace/mutation_tables/fused/patient_fused_matrix

    Row parsing is ORDER-AGNOSTIC: final_extract_muts.py/mutSigs.py write
    "tumor_origin<TAB>case_id" (older convention), while fuse_tables.py's
    fused output writes "case_id<TAB>tumor_origin" (case_id first, the
    newer convention). GDC case_id values are always UUIDs
    (8-4-4-4-12 hex), a format tumor origins never match -- this reliably
    tells the two fields apart regardless of which one comes first,
    so this loader works correctly against either table format without
    needing every table's format migrated to match.
    """
    npz_path = Path(f"{matrix_prefix}.npz")
    rows_path = Path(f"{matrix_prefix}_rows.txt")
    columns_path = Path(f"{matrix_prefix}_columns.txt")

    for p in (npz_path, rows_path, columns_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    sparse_matrix = scipy.sparse.load_npz(npz_path)

    rows_raw = rows_path.read_text().splitlines()
    tumor_origins, case_ids = [], []
    for line in rows_raw:
        field_a, field_b = line.split("\t")
        if UUID_PATTERN.match(field_a):
            case_id, tumor_origin = field_a, field_b
        else:
            tumor_origin, case_id = field_a, field_b
        tumor_origins.append(tumor_origin)
        case_ids.append(case_id)

    columns = columns_path.read_text().splitlines()
    return sparse_matrix, np.array(tumor_origins), np.array(case_ids), columns


def load_site_fold_assignment(site_fold_csv: Path, case_ids: np.ndarray) -> np.ndarray:
    """
    Loads site_stratified_5fold.csv (from make_splits.py) and returns a
    fold index array aligned to the matrix's own case_id order (same
    order as the sparse matrix's rows). Rows whose case_id isn't found in
    the fold assignment file get fold = -1 and are excluded by the caller
    -- this can happen if the fold file was built from a slightly
    different patient set (e.g. before/after an empty-MAF cleanup pass).
    """
    fold_df = pd.read_csv(site_fold_csv)
    case_id_to_fold = dict(zip(fold_df["case_id"], fold_df["fold"]))

    folds = np.array([case_id_to_fold.get(cid, -1) for cid in case_ids])
    n_missing = (folds == -1).sum()
    if n_missing > 0:
        log.warning(f"{n_missing} / {len(case_ids)} patients in the matrix have no fold "
                     f"assignment in {site_fold_csv} -- excluded from CV")
    return folds


# ---------------------------------------------------------------------------
# Availability checks -- skip gracefully if a library isn't installed
# ---------------------------------------------------------------------------

def check_model_available() -> bool:
    try:
        import xgboost  # noqa
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# sklearn-style models: param sampling + model construction, in one place
# ---------------------------------------------------------------------------

def sample_params(trial) -> dict:
    return {
        # upper bound widened since early stopping (see
        # fit_sklearn_model) now finds the actual natural stopping
        # point per trial -- Optuna doesn't need to guess the exact
        # right count, just needs room to go high enough when a low
        # learning_rate calls for more rounds
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _gpu_available() -> bool:
    """
    Reuses torch's CUDA detection (already a dependency for the MLP)
    rather than adding a new one -- if torch sees a GPU, XGBoost/LightGBM
    are told to try using it too.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_sklearn_model(params: dict, n_classes: int):
    from xgboost import XGBClassifier
    # XGBoost's official pip wheels bundle CUDA support (1.6+), so
    # device="cuda" reliably works if a GPU is actually present --
    # no special install needed
    device = "cuda" if _gpu_available() else "cpu"
    return XGBClassifier(
        objective="multi:softmax", num_class=n_classes, tree_method="hist",
        device=device, eval_metric="mlogloss", n_jobs=-1, random_state=42, **params,
    )


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def fit_sklearn_model(model, X_tr, y_tr, X_val=None, y_val=None, early_stopping_rounds=50):
    """
    Fits the XGBoost model. When a validation set is given, uses early
    stopping with restoration of the BEST boosting round rather than
    just whatever round training happened to stop at -- combined with
    sample_params' wide n_estimators upper bound, this lets Optuna
    search a generously large range without needing to guess the exact
    right count itself: early stopping finds the natural stopping point
    per trial. Patience defaults to 50 rounds specifically to NOT be
    narrow -- a low patience (e.g. 10-15) risks stopping before the
    model has actually converged, especially early in an Optuna search
    when other hyperparameters (e.g. a low learning_rate) mean progress
    per round is slow.

    Returns (model, eval_history) -- eval_history is a list of
    {round, train_mlogloss, val_mlogloss, train_accuracy, val_accuracy}
    dicts when a validation set was given (the per-round training
    curve), or None otherwise.
    """
    eval_history = None
    if X_val is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds, eval_metric=["merror", "mlogloss"])
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        results = model.evals_result()
        train_key, val_key = list(results.keys())  # "validation_0" (train), "validation_1" (val)
        train_loss = results[train_key]["mlogloss"]
        val_loss = results[val_key]["mlogloss"]
        train_acc = [1.0 - e for e in results[train_key]["merror"]]
        val_acc = [1.0 - e for e in results[val_key]["merror"]]
        eval_history = [
            {"round": i + 1, "train_mlogloss": tr_l, "val_mlogloss": va_l,
             "train_accuracy": tr_a, "val_accuracy": va_a}
            for i, (tr_l, va_l, tr_a, va_a) in enumerate(zip(train_loss, val_loss, train_acc, val_acc))
        ]
    else:
        model.fit(X_tr, y_tr)
    return model, eval_history


def make_objective(X_train, y_train, n_classes):
    def objective(trial):
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=trial.number,
            )
        except ValueError:
            return 0.0

        # XGBoost is scale-invariant -- MaxAbsScaler doesn't change its
        # splits/predictions at all, so no scaling is applied here.
        #
        # NOTE: XGBoost input stays as plain scipy sparse here even when
        # a GPU is used for training (device="cuda" in build_sklearn_model)
        # -- a GPU-resident CuPy sparse CSR was tried, but this XGBoost
        # build's DMatrix dispatcher explicitly rejects it
        # ("cupyx CSR is not supported yet"). A dense CuPy array would
        # work but risks blowing GPU memory at this column count
        # (214K+ columns -- a dense float32 copy of even one training
        # split would be several GB, wasteful for data this sparse). So
        # XGBoost keeps the CPU<->GPU copy-per-predict overhead (visible
        # as the "mismatched devices" warning) as the accepted tradeoff.
        params = sample_params(trial)
        model = build_sklearn_model(params, n_classes)
        model, _ = fit_sklearn_model(model, X_tr, y_tr, X_val, y_val)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    return objective


# ---------------------------------------------------------------------------
# Fit + evaluation -- one split (used per-fold by the CV loop below)
# ---------------------------------------------------------------------------

def get_full_width_scores(model, X_scaled, n_classes):
    """
    Returns an (n_samples, n_classes) ranking-score matrix for computing
    top-k accuracy. Handles one wrinkle: sklearn/XGBoost models only
    ever learn the classes present in y_train for that fold --
    model.classes_ can be a SUBSET of the full 0..n_classes-1 range if a
    class was entirely absent from this fold's training data. Raw
    scores would then be misaligned with the true label space. This
    remaps scores into the full n_classes-wide array, filling any class
    the model never saw during training with a very low score --
    correctly ranking it last, since the model genuinely never predicts
    it, rather than silently misaligning columns.

    (The predict_proba/decision_function fallback below is generic --
    XGBoost always has predict_proba, so the decision_function branch
    never actually triggers here, but keeping it costs nothing and
    matches the module docstring's note that other model types could be
    reintroduced later.)
    """
    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(X_scaled)
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_scaled)
        if raw_scores.ndim == 1:  # binary case returns a 1D array -- shouldn't happen at 32 classes, but be safe
            raw_scores = np.column_stack([-raw_scores, raw_scores])
    else:
        raise ValueError(f"Model {type(model)} has neither predict_proba nor decision_function")

    model_classes = getattr(model, "classes_", np.arange(n_classes))
    if raw_scores.shape[1] == n_classes and list(model_classes) == list(range(n_classes)):
        return raw_scores  # already full-width and in order, nothing to remap

    full_scores = np.full((raw_scores.shape[0], n_classes), -1e9, dtype=np.float64)
    for i, cls in enumerate(model_classes):
        full_scores[:, int(cls)] = raw_scores[:, i]
    return full_scores


def compute_ranking_metrics(y_true, scores, preds, n_classes, k_values=(1, 3, 5)):
    """
    Computes top-1/top-3/top-5 accuracy and weighted F1 from a full-width
    (n_samples, n_classes) score matrix. top-1 accuracy is identical to
    plain accuracy_score on preds (kept as its own computation here so
    this function is self-contained and doesn't depend on preds matching
    argmax(scores) exactly, e.g. for models with custom tie-breaking).
    """
    results = {}
    ranked = np.argsort(-scores, axis=1)  # each row: class indices, best first
    for k in k_values:
        topk = ranked[:, :k]
        correct = (topk == y_true[:, None]).any(axis=1)
        results[f"top{k}_accuracy"] = correct.mean()

    results["weighted_f1"] = f1_score(y_true, preds, labels=np.arange(n_classes),
                                        average="weighted", zero_division=0)
    results["macro_f1"] = f1_score(y_true, preds, labels=np.arange(n_classes),
                                     average="macro", zero_division=0)
    return results


def fit_one_split(best_params, X_train, y_train, X_test, y_test, n_classes, label_encoder):
    # NOTE: XGBoost input stays as plain scipy sparse even when GPU is
    # used for training -- see the matching note in make_objective for
    # why a GPU-resident CuPy sparse CSR isn't used here (unsupported by
    # this XGBoost build's DMatrix dispatcher; dense would risk GPU OOM
    # at 214K+ columns).
    model = build_sklearn_model(dict(best_params), n_classes)
    # carve a validation slice out of the TRAINING data only for early
    # stopping -- X_test (the outer test fold) is never touched here,
    # it's reserved purely for final evaluation below
    X_fit, X_es_val, y_fit, y_es_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=0
    )
    model, eval_history = fit_sklearn_model(model, X_fit, y_fit, X_es_val, y_es_val)

    preds = model.predict(X_test)
    scores = get_full_width_scores(model, X_test, n_classes)

    artifact = {"model": model, "scaler": None}

    metrics = compute_ranking_metrics(y_test, scores, preds, n_classes)
    report = classification_report(
        y_test, preds, labels=np.arange(n_classes), target_names=label_encoder.classes_, zero_division=0
    )
    return metrics, report, artifact, eval_history


def fit_and_evaluate_nested_cv(X, y, folds, n_folds, n_classes,
                                label_encoder, args, output_dir, n_trials):
    """
    NESTED, site-grouped K-fold CV. For each outer fold:
      1. The outer test fold (site-grouped, held out) is set aside and
         NEVER touched until step 5.
      2. Hyperparameters are tuned via Optuna using ONLY the other four
         folds as the tuning pool (make_objective further splits THIS
         pool internally for its own train/val trials -- so the outer
         test fold's patients, and by extension their sites, never
         influence which hyperparameters get chosen for this fold).
      3. Best hyperparameters for this fold are locked in and saved
         separately (<model>_fold<N>_best_params.json).
      4. The model is retrained on the FULL four-fold training pool with
         those hyperparameters.
      5. Evaluated ONCE on the untouched outer test fold.

    This fixes the leakage in the earlier (non-nested) design, where a
    single global 80/20 tuning split was drawn from the full dataset
    BEFORE cross-validation -- meaning patients (and their sites) that
    later served as an outer test fold could still influence which
    hyperparameters were chosen, making the reported 5-fold performance
    somewhat optimistic. Nested CV costs ~5x the tuning budget per model
    (a full Optuna search runs once per outer fold instead of once total)
    -- worth it for the more honest number, but budget accordingly.

    Fully resumable per fold: if a fold's model, params, and report all
    already exist on disk, that fold is skipped entirely (not just the
    tuning step) -- delete a fold's files to force just that fold to rerun.
    """
    model_name = "xgboost"
    fold_results = []
    best_fold_metric = -1
    best_artifact = None
    best_fold_idx = None

    for fold_idx in range(n_folds):
        test_mask = folds == fold_idx
        train_mask = (folds != fold_idx) & (folds != -1)

        X_train_outer, X_test = X[train_mask], X[test_mask]
        y_train_outer, y_test = y[train_mask], y[test_mask]

        fold_model_path = output_dir / f"{model_name}_fold{fold_idx}_model.joblib"
        fold_params_path = output_dir / f"{model_name}_fold{fold_idx}_best_params.json"
        fold_report_path = output_dir / f"{model_name}_fold{fold_idx}_classification_report.txt"

        if fold_model_path.exists() and fold_params_path.exists() and fold_report_path.exists():
            log.info(f"  [{model_name}] fold {fold_idx + 1}/{n_folds}: already complete -- skipping "
                     f"(delete this fold's files to force a rerun)")
            with open(fold_report_path) as f:
                report_text = f.read()
            metrics = {}
            for line in report_text.splitlines():
                for key, label in [("top1_accuracy", "Top-1 accuracy"), ("top3_accuracy", "Top-3 accuracy"),
                                    ("top5_accuracy", "Top-5 accuracy"), ("weighted_f1", "Weighted F1"),
                                    ("macro_f1", "Macro F1")]:
                    if line.startswith(f"{label}:"):
                        metrics[key] = float(line.split(":")[1].strip())
            fold_results.append({"fold": fold_idx, **metrics})
            if metrics.get("weighted_f1", -1) > best_fold_metric:
                best_fold_metric = metrics["weighted_f1"]
                best_artifact = joblib.load(fold_model_path)
                best_fold_idx = fold_idx
            continue

        log.info(f"  [{model_name}] fold {fold_idx + 1}/{n_folds}: "
                 f"outer train={X_train_outer.shape[0]}, outer test={X_test.shape[0]}")

        # --- INNER tuning: pool is ONLY this fold's 4 training folds --
        # the outer test fold above is never passed in here, so it can't
        # influence hyperparameter selection for this fold
        log.info(f"  [{model_name}] fold {fold_idx + 1}: tuning ({n_trials} trials, "
                 f"pool = outer-train only)...")
        storage_path = output_dir / f"optuna_{model_name}_outerfold{fold_idx}.db"
        study = optuna.create_study(
            direction="maximize", storage=f"sqlite:///{storage_path}",
            study_name=f"{model_name}_fold{fold_idx}", load_if_exists=True,
        )
        objective = make_objective(X_train_outer, y_train_outer, n_classes)
        remaining = max(0, n_trials - len(study.trials))
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        best_params = study.best_params

        with open(fold_params_path, "w") as f:
            json.dump(best_params, f, indent=2)

        # --- retrain on the FULL outer-train pool with those params, evaluate ONCE on the held-out outer test fold ---
        metrics, report, artifact, eval_history = fit_one_split(
            best_params, X_train_outer, y_train_outer, X_test, y_test,
            n_classes, label_encoder,
        )
        log.info(f"  [{model_name}] fold {fold_idx + 1}: "
                 f"top1={metrics['top1_accuracy']:.4f}, top3={metrics['top3_accuracy']:.4f}, "
                 f"top5={metrics['top5_accuracy']:.4f}, weighted_F1={metrics['weighted_f1']:.4f}, "
                 f"macro_F1={metrics['macro_f1']:.4f} (tuning val acc: {study.best_value:.4f})")

        fold_results.append({"fold": fold_idx, "tuning_val_accuracy": study.best_value, **metrics})

        if eval_history:
            curve_path = output_dir / f"{model_name}_fold{fold_idx}_training_curve.csv"
            pd.DataFrame(eval_history).to_csv(curve_path, index=False)
            log.info(f"  [{model_name}] fold {fold_idx + 1}: saved training curve "
                     f"({len(eval_history)} rounds) to {curve_path}")

        with open(fold_report_path, "w") as f:
            f.write(f"Model: {model_name}, fold: {fold_idx}\n")
            f.write(f"Hyperparameters tuned on: outer-train folds only (nested CV)\n")
            f.write(f"Tuning validation accuracy: {study.best_value:.4f}\n")
            f.write(f"Top-1 accuracy: {metrics['top1_accuracy']:.4f}\n")
            f.write(f"Top-3 accuracy: {metrics['top3_accuracy']:.4f}\n")
            f.write(f"Top-5 accuracy: {metrics['top5_accuracy']:.4f}\n")
            f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n\n")
            f.write(report)

        joblib.dump(artifact, fold_model_path)

        if metrics["weighted_f1"] > best_fold_metric:
            best_fold_metric = metrics["weighted_f1"]
            best_artifact = artifact
            best_fold_idx = fold_idx

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(output_dir / f"{model_name}_cv_folds.csv", index=False)

    summary = {}
    for metric_name in ["top1_accuracy", "top3_accuracy", "top5_accuracy", "weighted_f1", "macro_f1"]:
        summary[f"{metric_name}_mean"] = fold_df[metric_name].mean()
        summary[f"{metric_name}_std"] = fold_df[metric_name].std()

    log.info(f"[{model_name}] NESTED CV RESULT: "
             f"top1={summary['top1_accuracy_mean']:.4f}+/-{summary['top1_accuracy_std']:.4f}, "
             f"top3={summary['top3_accuracy_mean']:.4f}+/-{summary['top3_accuracy_std']:.4f}, "
             f"top5={summary['top5_accuracy_mean']:.4f}+/-{summary['top5_accuracy_std']:.4f}, "
             f"weighted_F1={summary['weighted_f1_mean']:.4f}+/-{summary['weighted_f1_std']:.4f} "
             f"(best fold: {best_fold_idx})")

    return summary, best_artifact, best_fold_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_final_model_training(X, y, n_classes, label_encoder, args, output_dir):
    """
    Trains ONE model on the FULL dataset (every patient), with
    hyperparameters found by a FRESH Optuna search run directly on the
    full dataset -- not borrowed from any of the 5 CV folds. Reuses
    make_objective() exactly as the per-fold nested CV does, just
    passing every patient as the tuning pool instead of one fold's
    4-fold training pool. Study saved as optuna_xgboost_final.db,
    distinct from the 5 per-fold study files.

    A small internal split is still carved out purely for early
    stopping -- not a test set, matching fit_one_split's existing
    early-stopping pattern exactly.

    Saves xgboost_final_model.joblib and xgboost_final_params.json
    (chosen hyperparameters + the tuning validation accuracy they
    achieved, for transparency).
    """
    model_name = "xgboost"
    final_model_path = output_dir / f"{model_name}_final_model.joblib"
    final_params_path = output_dir / f"{model_name}_final_params.json"
    if final_model_path.exists() and final_params_path.exists():
        log.info(f"[{model_name}] final model already complete -- skipping "
                 f"(delete {final_model_path.name} and optuna_{model_name}_final.db to force a rerun)")
        return

    log.info(f"[{model_name}] final model: running FRESH Optuna search ({args.n_trials} trials) on "
             f"the full dataset ({X.shape[0]} patients) -- not reusing any of the 5 CV folds' hyperparameters")
    storage_path = output_dir / f"optuna_{model_name}_final.db"
    study = optuna.create_study(
        direction="maximize", storage=f"sqlite:///{storage_path}",
        study_name=f"{model_name}_final", load_if_exists=True,
    )
    objective = make_objective(X, y, n_classes)
    remaining = max(0, args.n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    best_params = study.best_params
    log.info(f"[{model_name}] final model: fresh tuning done -- best_params={best_params}, "
             f"tuning_val_accuracy={study.best_value:.4f}")

    model = build_sklearn_model(dict(best_params), n_classes)
    X_fit, X_es_val, y_fit, y_es_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=0
    )
    model, eval_history = fit_sklearn_model(model, X_fit, y_fit, X_es_val, y_es_val)

    artifact = {"model": model, "scaler": None}
    joblib.dump(artifact, final_model_path)

    if eval_history:
        history_path = output_dir / f"{model_name}_final_training_curve.csv"
        pd.DataFrame(eval_history).to_csv(history_path, index=False)
        log.info(f"[{model_name}] saved training curve ({len(eval_history)} rounds) to {history_path}")

    with open(final_params_path, "w") as f:
        json.dump({**best_params, "tuning_val_accuracy": study.best_value,
                   "n_total_patients": int(X.shape[0]), "n_trials": args.n_trials}, f, indent=2)

    log.info(f"[{model_name}] final model saved to {final_model_path} "
             f"(trained on all {X.shape[0]} patients, hyperparameters from a fresh "
             f"{args.n_trials}-trial Optuna search)")


# ---------------------------------------------------------------------------
# Training curve plots -- one line per fold, distinct colors
# ---------------------------------------------------------------------------

CURVE_COLOR_PALETTE = [
    "#1f77b4",  # blue
    "#9467bd",  # purple
    "#2ca02c",  # green
    "#d62728",  # red
    "#ff7f0e",  # orange
]


def plot_fold_curves(fold_curves, xlabel, ylabel, title, outpath):
    """
    fold_curves: list of (fold_label, list_of_values) tuples, one per
    fold. Each fold's line uses its own distinct color from
    CURVE_COLOR_PALETTE (blue/purple/green/red/orange), cycling if there
    are ever more than 5 folds.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (fold_label, curve) in enumerate(fold_curves):
        ax.plot(
            range(1, len(curve) + 1), curve,
            color=CURVE_COLOR_PALETTE[i % len(CURVE_COLOR_PALETTE)], linewidth=1.8, alpha=0.9,
            label=fold_label,
        )
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved training curve plot: {outpath}")


def generate_all_training_curve_plots(output_dir, n_folds):
    """
    Reads every completed fold's xgboost_fold<N>_training_curve.csv and
    produces 4 plots: train/val loss, train/val accuracy, each with one
    line per fold. Called automatically at the end of the nested CV run
    -- skips gracefully (with a log message) if fewer than 2 folds have
    a curve file, since a meaningful per-fold comparison needs more than one line.
    """
    model_name = "xgboost"
    fold_data = {}
    for fold_idx in range(n_folds):
        curve_path = output_dir / f"{model_name}_fold{fold_idx}_training_curve.csv"
        if curve_path.exists():
            fold_data[fold_idx] = pd.read_csv(curve_path)

    if len(fold_data) < 2:
        log.warning(f"Only {len(fold_data)} fold training curve(s) found -- skipping "
                     f"training curve plots (need at least 2 folds for a meaningful comparison)")
        return

    specs = [
        ("train_mlogloss", "Boosting Round", "Training Log Loss",
         "XGBoost Training Loss per Fold", "train_loss_curve.png"),
        ("train_accuracy", "Boosting Round", "Training Accuracy",
         "XGBoost Training Accuracy per Fold", "train_accuracy_curve.png"),
        ("val_mlogloss", "Boosting Round", "Validation Log Loss",
         "XGBoost Validation Loss per Fold", "val_loss_curve.png"),
        ("val_accuracy", "Boosting Round", "Validation Accuracy",
         "XGBoost Validation Accuracy per Fold", "val_accuracy_curve.png"),
    ]
    for column, xlabel, ylabel, title, filename in specs:
        fold_curves = [(f"Fold {i + 1}", df[column].tolist()) for i, df in fold_data.items()]
        plot_fold_curves(fold_curves, xlabel, ylabel, title, output_dir / filename)


def main():
    model_name = "xgboost"
    parser = argparse.ArgumentParser(description="Tune XGBoost for tumor-origin prediction")
    parser.add_argument("--matrix-prefix", required=True,
                         help="Path prefix for the feature table to train on -- same convention as "
                              "fuse_tables.py's --output-prefix. Points at <prefix>.npz / "
                              "<prefix>_rows.txt / <prefix>_columns.txt. Use this to run the exact "
                              "same classifier against the mutation-only table, the signatures-only "
                              "table, or a fused table combining multiple feature types.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site-fold-csv", required=True,
                         help="Path to site_stratified_5fold.csv from make_splits.py. Hyperparameters "
                              "are tuned once (via an inner stratified split, same as before), but final "
                              "evaluation now runs full site-stratified K-fold CV using these fold "
                              "assignments, reporting mean +/- std across folds instead of a single "
                              "train/test split.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (must match the fold file)")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials PER OUTER FOLD")
    parser.add_argument("--min-class-size", type=int, default=2)
    parser.add_argument("--final-model", action="store_true",
                         help="Train ONE final model on the FULL dataset (all patients, no held-out "
                              "test fold), instead of the nested-CV loop. This is the model you'd "
                              "actually deploy -- nested CV's job was to evaluate the approach, not to "
                              "produce N separate deployable models (each fold's model only saw ~80% "
                              "of the data). Hyperparameters come from a FRESH Optuna search "
                              "(--n-trials trials) run directly on the full dataset -- not borrowed "
                              "from any of the 5 CV folds. Saved as optuna_xgboost_final.db, distinct "
                              "from the 5 per-fold study files.")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not check_model_available():
        log.error("xgboost is not installed (pip install --break-system-packages xgboost)")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading matrix...")
    X, tumor_origins, case_ids, columns = load_matrix(args.matrix_prefix)
    log.info(f"Loaded matrix: {X.shape[0]} patients x {X.shape[1]} features")

    log.info(f"Loading site-stratified fold assignment from {args.site_fold_csv}...")
    folds = load_site_fold_assignment(Path(args.site_fold_csv), case_ids)

    class_counts = pd.Series(tumor_origins).value_counts()
    small_classes = class_counts[class_counts < args.min_class_size].index.tolist()
    keep_mask = (folds != -1)
    if small_classes:
        log.warning(f"Dropping {len(small_classes)} tumor origin(s) with fewer than "
                     f"{args.min_class_size} patients: {small_classes}")
        keep_mask = keep_mask & (~pd.Series(tumor_origins).isin(small_classes).to_numpy())

    X, tumor_origins, case_ids, folds = X[keep_mask], tumor_origins[keep_mask], case_ids[keep_mask], folds[keep_mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tumor_origins)
    n_classes = len(label_encoder.classes_)
    log.info(f"Final dataset: {X.shape[0]} patients, {n_classes} tumor origins, "
             f"{args.n_folds} CV folds")

    if args.final_model:
        try:
            run_final_model_training(X, y, n_classes, label_encoder, args, output_dir)
        except Exception as e:
            log.error(f"{model_name} failed during final-model training: {e}")
            log.error("Full traceback:", exc_info=True)
        log.info("Done (final-model mode).")
        return

    log.info(f"=== Running nested site-grouped CV for {model_name} "
             f"({args.n_trials} tuning trials PER outer fold) ===")

    try:
        summary, best_artifact, best_fold_idx = fit_and_evaluate_nested_cv(
            X, y, folds, args.n_folds, n_classes,
            label_encoder, args, output_dir, args.n_trials,
        )
    except Exception as e:
        log.error(f"{model_name} failed during nested CV: {e}")
        log.error("Full traceback:", exc_info=True)
        return

    results_df = pd.DataFrame([{
        "model": model_name,
        "top1_accuracy_mean": summary["top1_accuracy_mean"], "top1_accuracy_std": summary["top1_accuracy_std"],
        "top3_accuracy_mean": summary["top3_accuracy_mean"], "top3_accuracy_std": summary["top3_accuracy_std"],
        "top5_accuracy_mean": summary["top5_accuracy_mean"], "top5_accuracy_std": summary["top5_accuracy_std"],
        "weighted_f1_mean": summary["weighted_f1_mean"], "weighted_f1_std": summary["weighted_f1_std"],
        "macro_f1_mean": summary["macro_f1_mean"], "macro_f1_std": summary["macro_f1_std"],
        "n_trials_per_fold": args.n_trials, "best_fold": best_fold_idx,
    }])
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    log.info("\n" + results_df.to_string(index=False))

    if best_artifact is not None:
        joblib.dump(best_artifact, output_dir / f"{model_name}_model.joblib")

    best_row = results_df.iloc[0]
    log.info(f"{model_name}: "
             f"top1={best_row['top1_accuracy_mean']:.4f}+/-{best_row['top1_accuracy_std']:.4f}, "
             f"top3={best_row['top3_accuracy_mean']:.4f}+/-{best_row['top3_accuracy_std']:.4f}, "
             f"top5={best_row['top5_accuracy_mean']:.4f}+/-{best_row['top5_accuracy_std']:.4f}, "
             f"weighted_F1={best_row['weighted_f1_mean']:.4f}+/-{best_row['weighted_f1_std']:.4f}")

    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    log.info("Generating training curve plots (one line per fold)...")
    generate_all_training_curve_plots(output_dir, args.n_folds)

    log.info("Done.")


if __name__ == "__main__":
    main()