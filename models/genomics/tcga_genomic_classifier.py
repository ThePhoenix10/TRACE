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


UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def load_matrix(matrix_prefix: str):


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


    fold_df = pd.read_csv(site_fold_csv)
    case_id_to_fold = dict(zip(fold_df["case_id"], fold_df["fold"]))

    folds = np.array([case_id_to_fold.get(cid, -1) for cid in case_ids])
    n_missing = (folds == -1).sum()
    if n_missing > 0:
        log.warning(f"{n_missing} / {len(case_ids)} patients in the matrix have no fold "
                     f"assignment in {site_fold_csv} -- excluded from CV")
    return folds


def check_model_available() -> bool:
    try:
        import xgboost
        return True
    except ImportError:
        return False


def sample_params(trial) -> dict:
    return {


        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _gpu_available() -> bool:


    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_sklearn_model(params: dict, n_classes: int):
    from xgboost import XGBClassifier


    device = "cuda" if _gpu_available() else "cpu"
    return XGBClassifier(
        objective="multi:softmax", num_class=n_classes, tree_method="hist",
        device=device, eval_metric="mlogloss", n_jobs=-1, random_state=42, **params,
    )


def fit_sklearn_model(model, X_tr, y_tr, X_val=None, y_val=None, early_stopping_rounds=50):


    eval_history = None
    if X_val is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds, eval_metric=["merror", "mlogloss"])
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        results = model.evals_result()
        train_key, val_key = list(results.keys())
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


        params = sample_params(trial)
        model = build_sklearn_model(params, n_classes)
        model, _ = fit_sklearn_model(model, X_tr, y_tr, X_val, y_val)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    return objective


def get_full_width_scores(model, X_scaled, n_classes):


    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(X_scaled)
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_scaled)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])
    else:
        raise ValueError(f"Model {type(model)} has neither predict_proba nor decision_function")

    model_classes = getattr(model, "classes_", np.arange(n_classes))
    if raw_scores.shape[1] == n_classes and list(model_classes) == list(range(n_classes)):
        return raw_scores

    full_scores = np.full((raw_scores.shape[0], n_classes), -1e9, dtype=np.float64)
    for i, cls in enumerate(model_classes):
        full_scores[:, int(cls)] = raw_scores[:, i]
    return full_scores


def compute_ranking_metrics(y_true, scores, preds, n_classes, k_values=(1, 3, 5)):


    results = {}
    ranked = np.argsort(-scores, axis=1)
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


    model = build_sklearn_model(dict(best_params), n_classes)


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


def run_final_model_training(X, y, n_classes, label_encoder, args, output_dir):


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


CURVE_COLOR_PALETTE = [
    "#1f77b4",
    "#9467bd",
    "#2ca02c",
    "#d62728",
    "#ff7f0e",
]


def plot_fold_curves(
    fold_curves,
    xlabel,
    ylabel,
    title,
    outpath,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (fold_label, curve) in enumerate(fold_curves):
        ax.plot(
            range(1, len(curve) + 1),
            curve,
            color=CURVE_COLOR_PALETTE[i % len(CURVE_COLOR_PALETTE)],
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

    log.info(f"Saved training curve plot: {outpath}")


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
            log.warning(
                f"{curve_path} not found -- skipping fold {fold_idx}"
            )
            continue

        df = pd.read_csv(curve_path)

        if column not in df.columns:
            log.warning(
                f"Column '{column}' not found in {curve_path} "
                f"(has: {list(df.columns)}) -- skipping fold {fold_idx}"
            )
            continue

        curves.append(
            (
                f"Fold {fold_idx + 1}",
                df[column].tolist(),
            )
        )

    return curves


def generate_all_training_curve_plots(output_dir, n_folds):
    model_name = "xgboost"
    file_suffix = "training_curve.csv"

    plot_specs = [
        (
            "train_mlogloss",
            "Training Log Loss",
            "XGBoost Training Loss per Fold",
            "train_loss_curve.png",
        ),
        (
            "train_accuracy",
            "Training Accuracy",
            "XGBoost Training Accuracy per Fold",
            "train_accuracy_curve.png",
        ),
        (
            "val_mlogloss",
            "Validation Log Loss",
            "XGBoost Validation Loss per Fold",
            "val_loss_curve.png",
        ),
        (
            "val_accuracy",
            "Validation Accuracy",
            "XGBoost Validation Accuracy per Fold",
            "val_accuracy_curve.png",
        ),
    ]

    for column, ylabel, title, filename in plot_specs:
        curves = load_fold_curves(
            model_dir=output_dir,
            model_name=model_name,
            n_folds=n_folds,
            column=column,
            file_suffix=file_suffix,
        )

        if not curves:
            log.warning(
                f"No fold curves found for {model_name}/{column} -- "
                f"skipping this plot"
            )
            continue

        plot_fold_curves(
            curves,
            "Boosting Round",
            ylabel,
            title,
            output_dir / filename,
        )

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