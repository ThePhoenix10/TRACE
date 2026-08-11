import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics import f1_score

from models.histology.tcga_histology_classifier import BagDataset, build_model
from models.genomics.tcga_genomic_classifier import get_full_width_scores
from models.fusion.tcga_fusion_classifier import (
    align_to_common_classes,
    get_mil_fold_probabilities,
    get_mutation_fold_probabilities,
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def read_rows(path: Path):
    case_ids = []
    labels = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")

        if len(parts) == 1:
            case_id = parts[0].strip()
            label = ""
        else:
            field_a, field_b = parts[0].strip(), parts[1].strip()

            if UUID_PATTERN.match(field_a):
                case_id, label = field_a, field_b
            else:
                label, case_id = field_a, field_b

        case_ids.append(case_id)
        labels[case_id] = label

    return case_ids, labels


def load_metadata(path: Path):
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str).fillna("")

    if "case_id" not in df.columns:
        raise ValueError(f"{path} must contain a case_id column")

    return {
        str(row["case_id"]).strip(): row.to_dict()
        for _, row in df.iterrows()
    }


def map_primary_site_to_origin(primary_site):
    text = str(primary_site or "").strip().lower()

    if "bronchus" in text or "lung" in text:
        return "Lung"
    if "breast" in text:
        return "Breast"
    if "kidney" in text or "renal" in text:
        return "Kidney"
    if "colon" in text or "large intestine" in text:
        return "Colon"
    if "brain" in text:
        return "Brain"
    if "pancreas" in text:
        return "Pancreas"
    if "ovary" in text or "fallopian" in text:
        return "Ovary"
    if "uterus" in text or "endometrium" in text:
        return "Uterus"
    if any(x in text for x in [
        "larynx", "tongue", "oral cavity", "pharynx",
        "tonsil", "mouth", "lip",
    ]):
        return "Head and Neck"

    return "Unknown"


def load_fused_cptac(fused_dir: Path):
    matrix_path = fused_dir / "cptac_matrix.npz"
    dense_path = fused_dir / "cptac_matrix_dense.csv"
    rows_path = fused_dir / "cptac_matrix_rows.txt"
    columns_path = fused_dir / "cptac_matrix_columns.txt"
    metadata_path = fused_dir / "cptac_patient_metadata.csv"

    for path in [matrix_path, dense_path, rows_path, columns_path]:
        if not path.exists():
            raise FileNotFoundError(f"Expected fused CPTAC file not found: {path}")

    matrix = load_npz(matrix_path).tocsr()
    case_ids, row_labels = read_rows(rows_path)
    columns = [
        line.strip()
        for line in columns_path.read_text().splitlines()
        if line.strip()
    ]

    if matrix.shape != (len(case_ids), len(columns)):
        raise ValueError(
            f"Fused CPTAC matrix shape {matrix.shape} does not match "
            f"{len(case_ids)} rows and {len(columns)} columns"
        )

    dense = pd.read_csv(dense_path, index_col=0)
    dense.index = dense.index.astype(str)

    if dense.shape != matrix.shape:
        raise ValueError(
            f"Dense CPTAC matrix shape {dense.shape} does not match "
            f"sparse matrix shape {matrix.shape}"
        )

    if dense.index.tolist() != case_ids:
        raise ValueError(
            "Case-ID order in cptac_matrix_dense.csv does not match "
            "cptac_matrix_rows.txt"
        )

    if list(dense.columns) != columns:
        raise ValueError(
            "Column order in cptac_matrix_dense.csv does not match "
            "cptac_matrix_columns.txt"
        )

    metadata = load_metadata(metadata_path)

    true_labels = {}
    for case_id in case_ids:
        label = str(row_labels.get(case_id, "")).strip()

        if not label or label == "Unknown":
            primary_site = metadata.get(case_id, {}).get("primary_site", "")
            mapped = map_primary_site_to_origin(primary_site)
            if mapped != "Unknown":
                label = mapped

        true_labels[case_id] = label or "Unknown"

    return case_ids, true_labels, metadata, dense_path


def match_embeddings(case_ids, metadata, embeddings_dir, manual_map_csv=None):
    embeddings_dir = Path(embeddings_dir)
    h5_files = sorted(embeddings_dir.rglob("*.h5"))

    manual = defaultdict(list)

    if manual_map_csv:
        df = pd.read_csv(manual_map_csv, dtype=str).fillna("")
        required = {"case_id", "embedding_path"}

        if not required.issubset(df.columns):
            raise ValueError("embedding map CSV requires case_id,embedding_path")

        for _, row in df.iterrows():
            path = Path(row["embedding_path"])

            if not path.is_absolute():
                path = embeddings_dir / path

            manual[str(row["case_id"]).strip()].append(path)

    matches = {}
    report = []

    for case_id in case_ids:
        submitter = str(
            metadata.get(case_id, {}).get("case_submitter_id", "")
        ).strip()

        identifiers = [value for value in [case_id, submitter] if value]

        if case_id in manual:
            paths = sorted({path for path in manual[case_id] if path.exists()})
            status = "manual" if paths else "manual_missing"
        else:
            paths = []

            for path in h5_files:
                stem = path.stem

                if any(
                    stem == identifier
                    or stem.startswith(identifier + "-")
                    or stem.startswith(identifier + "_")
                    for identifier in identifiers
                ):
                    paths.append(path)

            paths = sorted(set(paths))
            status = "matched" if paths else "unmatched"

        if paths:
            matches[case_id] = paths

        report.append({
            "case_id": case_id,
            "case_submitter_id": submitter,
            "embedding_count": len(paths),
            "embedding_paths": "|".join(str(path) for path in paths),
            "status": status,
        })

    return matches, pd.DataFrame(report), len(h5_files)


def load_mil_model(model_dir, model_name, n_classes, device):
    model_dir = Path(model_dir)

    params = json.loads(
        (model_dir / f"{model_name}_final_params.json").read_text()
    )

    model = build_model(
        model_name,
        n_classes,
        params["hidden_dim"],
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_dir / f"{model_name}_final_best.pt",
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model


def score_mil(model, model_name, case_paths, device):
    out = {}

    with torch.no_grad():
        for case_id, h5_paths in case_paths.items():
            slide_probs = []

            for h5_path in h5_paths:
                try:
                    bag, _ = BagDataset([Path(h5_path)], [0])[0]
                    bag = bag.to(device)

                    if model_name == "clam":
                        logits, _, _ = model(bag)
                    else:
                        logits, _ = model(bag)

                    slide_probs.append(
                        F.softmax(logits, dim=1)
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )

                except Exception as error:
                    print(
                        f"WARNING: MIL failed for {case_id}, "
                        f"slide {h5_path}: {error}"
                    )

            if slide_probs:
                out[case_id] = np.mean(
                    np.stack(slide_probs, axis=0),
                    axis=0,
                )

    return out


def score_genomics(model_dir, dense_csv, model_name):
    dense = pd.read_csv(dense_csv, index_col=0)
    dense.index = dense.index.astype(str)

    X = csr_matrix(dense.astype(np.float32).values)

    artifact = joblib.load(
        Path(model_dir) / f"{model_name}_final_model.joblib"
    )

    model = artifact["model"]
    scaler = artifact.get("scaler")

    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu", eval_metric="mlogloss")
        except Exception:
            pass

    try:
        model.get_booster().set_param({
            "device": "cpu",
            "eval_metric": "mlogloss",
        })
    except Exception:
        pass

    X_scaled = scaler.transform(X) if scaler is not None else X

    encoder = joblib.load(
        Path(model_dir) / "label_encoder.joblib"
    )

    classes = list(encoder.classes_)

    scores = get_full_width_scores(
        model,
        X_scaled,
        len(classes),
    )

    return {
        case_id: scores[i]
        for i, case_id in enumerate(dense.index.tolist())
    }, classes


def train_stacking_meta_classifier(args, common_classes):
    from sklearn.linear_model import LogisticRegression
    from types import SimpleNamespace

    tcga_args = SimpleNamespace(
        patient_subset_file=args.tcga_patient_subset_file,
        patient_cohort_csv=args.tcga_patient_cohort_csv,
        case_id_barcode_map=args.tcga_case_id_barcode_map,
        site_fold_csv=args.tcga_site_fold_csv,
        embeddings_dir=args.tcga_embeddings_dir,
        min_class_size=args.min_class_size,
        mil_model_dir=args.mil_model_dir,
        mil_model_name=args.mil_model_name,
        mutation_matrix_prefix=args.tcga_mutation_matrix_prefix,
        mutation_model_dir=args.genomic_model_dir,
        mutation_model_name=args.genomic_model_name,
    )

    train_mil = []
    train_genomics = []
    train_y = []

    class_to_idx = {
        class_name: i
        for i, class_name in enumerate(common_classes)
    }

    patient_cohort = pd.read_csv(
        args.tcga_patient_cohort_csv
    )

    case_id_to_true_label = dict(
        zip(
            patient_cohort["case_id"],
            patient_cohort["tumor_origin"],
        )
    )

    for fold_idx in range(args.n_tcga_folds):
        print(
            f"Loading TCGA fold {fold_idx} "
            f"held-out predictions for stacking training..."
        )

        mil_probs, mil_classes = get_mil_fold_probabilities(
            tcga_args,
            fold_idx,
        )

        genomic_probs, genomic_classes = get_mutation_fold_probabilities(
            tcga_args,
            fold_idx,
        )

        mil_aligned = align_to_common_classes(
            mil_probs,
            mil_classes,
            common_classes,
        )

        genomic_aligned = align_to_common_classes(
            genomic_probs,
            genomic_classes,
            common_classes,
        )

        common_case_ids = sorted(
            set(mil_probs) & set(genomic_probs)
        )

        for case_id in common_case_ids:
            true_label = case_id_to_true_label.get(case_id)

            if true_label is None or true_label not in class_to_idx:
                continue

            train_mil.append(mil_aligned[case_id])
            train_genomics.append(genomic_aligned[case_id])
            train_y.append(class_to_idx[true_label])

    train_mil = np.array(train_mil)
    train_genomics = np.array(train_genomics)
    train_y = np.array(train_y)

    train_X = np.concatenate(
        [train_mil, train_genomics],
        axis=1,
    )

    print(
        f"Stacking meta-classifier: {train_X.shape[0]} training rows "
        f"(from all {args.n_tcga_folds} TCGA folds), "
        f"{train_X.shape[1]} features"
    )

    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    meta_model.fit(train_X, train_y)
    return meta_model


def apply_stacking_fusion(
    meta_model,
    genomic_probs,
    mil_probs,
    classes,
    both,
):
    fused = {}

    for case_id in both:
        x = np.concatenate([
            mil_probs[case_id],
            genomic_probs[case_id],
        ]).reshape(1, -1)

        proba = meta_model.predict_proba(x)[0]

        full = np.zeros(len(classes))

        for i, class_idx in enumerate(meta_model.classes_):
            full[int(class_idx)] = proba[i]

        fused[case_id] = full

    return fused


def score_and_report(
    args,
    output_dir,
    included,
    true_labels,
    embedding_matches,
    dense_path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    genomic_probs, classes = score_genomics(
        args.genomic_model_dir,
        dense_path,
        args.genomic_model_name,
    )

    mil_model = load_mil_model(
        args.mil_model_dir,
        args.mil_model_name,
        len(classes),
        device,
    )

    mil_probs = score_mil(
        mil_model,
        args.mil_model_name,
        {
            case_id: embedding_matches[case_id]
            for case_id in included
            if case_id in embedding_matches
        },
        device,
    )

    both = sorted(
        set(included)
        & set(genomic_probs)
        & set(mil_probs)
    )

    if not both:
        raise RuntimeError(
            "No CPTAC patients have both fused genomic "
            "features and usable histology embeddings."
        )

    if args.fusion_method == "stack":
        print(
            "Training stacking meta-classifier "
            "on TCGA held-out predictions..."
        )

        meta_model = train_stacking_meta_classifier(
            args,
            classes,
        )

        fused_probs = apply_stacking_fusion(
            meta_model,
            genomic_probs,
            mil_probs,
            classes,
            both,
        )

    results = []

    for case_id in both:
        true = true_labels.get(case_id, "")
        genomic_vec = genomic_probs[case_id]
        mil_vec = mil_probs[case_id]

        if args.fusion_method == "stack":
            fused_vec = fused_probs[case_id]
        else:
            fused_vec = genomic_vec * mil_vec

            if fused_vec.sum() > 0:
                fused_vec = fused_vec / fused_vec.sum()

        row = {
            "case_id": case_id,
            "true_tumor_origin": true,
            "embedding_count": len(embedding_matches[case_id]),
            "embedding_paths": "|".join(
                str(path)
                for path in embedding_matches[case_id]
            ),
        }

        for name, vec in [
            ("genomic", genomic_vec),
            ("mil", mil_vec),
            ("fused", fused_vec),
        ]:
            top3 = np.argsort(-vec)[:3]

            row[f"{name}_prediction"] = classes[top3[0]]
            row[f"{name}_correct"] = classes[top3[0]] == true
            row[f"{name}_top3_correct"] = (
                true in [classes[i] for i in top3]
            )
            row[f"{name}_top3"] = ", ".join(
                f"{classes[i]} ({vec[i]:.4f})"
                for i in top3
            )

        results.append(row)

    df = pd.DataFrame(results)

    df.to_csv(
        output_dir / "cptac_predictions.csv",
        index=False,
    )

    valid = df[
        df["true_tumor_origin"].notna()
        & df["true_tumor_origin"].ne("")
        & df["true_tumor_origin"].ne("Unknown")
    ]

    def metric(column):
        return float(valid[column].mean()) if len(valid) else None

    lines = [
        "CPTAC external validation",
        "=" * 70,
        (
            "Patients scored with fused genomic features + "
            f"histology embeddings: {len(df)}"
        ),
        f"Patients with known labels used for accuracy: {len(valid)}",
        "",
    ]

    for name in ["genomic", "mil", "fused"]:
        top1 = metric(f"{name}_correct")
        top3 = metric(f"{name}_top3_correct")

        if top1 is None:
            lines.append(f"{name}: n/a")
        else:
            weighted_f1 = f1_score(
                valid["true_tumor_origin"],
                valid[f"{name}_prediction"],
                average="weighted",
                zero_division=0,
            )

            lines.append(
                f"{name:<10} "
                f"top1={top1:.2%}  "
                f"top3={top3:.2%}  "
                f"weighted_f1={weighted_f1:.2%}"
            )

    lines.append("")

    if args.fusion_method == "stack":
        lines.append(
            "Fusion: logistic-regression stacking trained on "
            "TCGA held-out fold predictions."
        )
    else:
        lines.append(
            "Fusion: element-wise probability multiplication "
            "followed by normalization."
        )

    (output_dir / "cptac_scoring_report.txt").write_text(
        "\n".join(lines)
    )

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cptac-fused-dir",
        required=True,
        help=(
            "Output directory produced by "
            "data/cptac/fuse_cptac_tables.py"
        ),
    )

    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--embedding-map-csv", default=None)
    parser.add_argument("--mil-model-dir", required=True)
    parser.add_argument(
        "--mil-model-name",
        default="abmil",
        choices=["abmil", "clam", "transmil"],
    )
    parser.add_argument("--genomic-model-dir", required=True)
    parser.add_argument("--genomic-model-name", default="xgboost")
    parser.add_argument(
        "--fusion-method",
        default="multiply",
        choices=["multiply", "stack"],
    )

    parser.add_argument("--tcga-embeddings-dir", default=None)
    parser.add_argument("--tcga-patient-cohort-csv", default=None)
    parser.add_argument("--tcga-case-id-barcode-map", default=None)
    parser.add_argument("--tcga-site-fold-csv", default=None)
    parser.add_argument("--tcga-mutation-matrix-prefix", default=None)
    parser.add_argument("--tcga-patient-subset-file", default=None)
    parser.add_argument("--n-tcga-folds", type=int, default=5)
    parser.add_argument("--min-class-size", type=int, default=3)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    if args.fusion_method == "stack":
        required_tcga_args = [
            "tcga_embeddings_dir",
            "tcga_patient_cohort_csv",
            "tcga_case_id_barcode_map",
            "tcga_site_fold_csv",
            "tcga_mutation_matrix_prefix",
        ]

        missing = [
            arg_name
            for arg_name in required_tcga_args
            if getattr(args, arg_name) is None
        ]

        if missing:
            parser.error(
                "--fusion-method stack requires: "
                + ", ".join(
                    "--" + arg_name.replace("_", "-")
                    for arg_name in missing
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fused_dir = Path(args.cptac_fused_dir)

    included, labels, metadata, dense_path = load_fused_cptac(
        fused_dir
    )

    embedding_matches, match_report, total_h5 = match_embeddings(
        included,
        metadata,
        Path(args.embeddings_dir),
        Path(args.embedding_map_csv)
        if args.embedding_map_csv
        else None,
    )

    match_report.to_csv(
        output_dir / "cptac_embedding_match_report.csv",
        index=False,
    )

    print(
        f"Loaded {len(included)} patients from {fused_dir}"
    )

    print(
        f"Found {total_h5} embedding files; "
        f"{len(embedding_matches)} fused CPTAC patients "
        f"matched to at least one embedding."
    )

    score_and_report(
        args,
        output_dir,
        included,
        labels,
        embedding_matches,
        dense_path,
    )


if __name__ == "__main__":
    main()
