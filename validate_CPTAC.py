#!/usr/bin/env python3
"""
validate_CPTAC.py

Builds and scores a CPTAC external-validation cohort using patients with:
  1) at least one tabular genomic modality: mutation OR SBS96, and
  2) at least one matched UNI2-h .h5 embedding.

A patient does NOT need both mutation and SBS96. Missing mutation or SBS96
channels are filled with 0. CNA channels remain NaN in the saved dense matrix
and are filled with 0 only immediately before mutation-model scoring.

The final tabular vector is reindexed to the exact TCGA training-column order:
  som_mut_*  -> CPTAC mutation values, absent features = 0
  sig_*      -> CPTAC SBS96 values, absent channels = 0
  cna_*      -> NaN for every CPTAC patient

COLUMN NAMING (IMPORTANT -- updated from the original version of this script)
    The mutation and CNA extraction scripts (final_extract_muts.py,
    extract_cna_gistic.py) now bake their prefix directly into each
    column name AT EXTRACTION TIME -- e.g. "som_mut_TP53_any",
    "cna_ERBB2_amp" -- rather than producing bare gene names that this
    script would need to prefix itself. The original version of this
    script called add_prefix(cols, "mut__") / add_prefix(cols, "cna__"),
    which assumed a double-underscore prefix that was NEVER actually
    present on the real columns -- since those real columns already
    started with "som_mut_"/"cna_" (single underscore, different tag
    entirely), that add_prefix() call silently DOUBLE-PREFIXED every
    column (e.g. "mut__som_mut_TP53_any"), which would never match the
    training column list. Every mutation and CNA feature would have
    been silently treated as missing and zero-filled, with no error or
    warning. Confirmed directly by testing add_prefix() against the
    real current column names before this fix.

    This version does NOT call add_prefix() on mutation or CNA columns
    at all, since they already carry the correct prefix from extraction.
    The signature prefix ("sig_", single underscore) is assumed to match
    the same convention established by the other two extraction
    scripts, but has NOT been directly confirmed against your actual
    current signature extraction script -- verify this before trusting
    signature-feature alignment (see the printed column-prefix sample at
    startup, which shows the first few real signature column names).

    The training_columns categorization logic (deciding how to fill
    missing values per modality) was ALSO updated to check for the real
    "som_mut_"/"cna_" prefixes instead of the old "mut__"/"cna__".

The dense CSV preserves CNA NaN values. Immediately before mutation-model
scoring, NaN is replaced by 0 and the matrix is converted to sparse CSR,
matching the behavior of the existing external-validation script.

Patients are excluded only when they lack an embedding or lack both mutation and SBS96.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics import f1_score

from train_mil_models import BagDataset, build_model
from mutClassifier import get_full_width_scores
from fusion import (
    align_to_common_classes, get_mil_fold_probabilities, get_mutation_fold_probabilities,
)

# real prefixes as baked in by the current extraction scripts (verified
# directly against final_extract_muts.py and extract_cna_gistic.py).
# SIGNATURE is different from the other two: extract_mutational_
# signatures.py (the real TCGA-side script) produces BARE, unprefixed
# 96-channel names (e.g. "A[C>A]A", confirmed directly against its
# actual source) -- no prefix is baked in at extraction time the way
# mutation/CNA now are. The "sig__" prefix (double underscore) is
# instead added later, at FUSION time, by fuse_tables.py's own
# generic "<tag>__<column>" tagging mechanism. This script therefore
# needs to add that same "sig__" prefix itself before alignment,
# unlike mutation/CNA columns which already arrive correctly prefixed.
MUTATION_PREFIX = "som_mut_"
CNA_PREFIX = "cna_"
SIGNATURE_PREFIX = "sig__"


def read_rows(path: Path):
    """Read tumor_origin<TAB>case_id rows files."""
    case_ids, labels = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 1:
            origin, case_id = "", parts[0]
        else:
            origin, case_id = parts[0], parts[1]
        case_id = str(case_id).strip()
        case_ids.append(case_id)
        labels[case_id] = str(origin).strip()
    return case_ids, labels


def load_table(directory: Path, stem: str):
    matrix = load_npz(directory / f"{stem}.npz").tocsr()
    rows, labels = read_rows(directory / f"{stem}_rows.txt")
    columns = [
        x.strip()
        for x in (directory / f"{stem}_columns.txt").read_text().splitlines()
        if x.strip()
    ]
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError(
            f"{stem}: matrix shape {matrix.shape} does not match "
            f"{len(rows)} rows and {len(columns)} columns"
        )
    return matrix, rows, columns, labels


def load_metadata(path: Path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {str(r["case_id"]): r.to_dict() for _, r in df.iterrows()}


def load_manual_map(path, embeddings_dir):
    if not path:
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"case_id", "embedding_path"}
    if not required.issubset(df.columns):
        raise ValueError("embedding map CSV requires case_id,embedding_path")
    out = {}
    for _, row in df.iterrows():
        p = Path(row["embedding_path"])
        if not p.is_absolute():
            p = embeddings_dir / p
        out[str(row["case_id"])] = p
    return out


def match_embeddings(case_ids, metadata, embeddings_dir, manual_map_csv=None):
    """
    Return {case_id: [h5_path, ...]}.

    All matching slides are retained. Automatic matching uses case_id and
    case_submitter_id as exact stems or filename prefixes. This is important
    because CPTAC commonly has several WSI embeddings per patient.
    """
    embeddings_dir = Path(embeddings_dir)
    h5_files = sorted(embeddings_dir.rglob("*.h5"))

    manual = defaultdict(list)
    if manual_map_csv:
        df = pd.read_csv(manual_map_csv, dtype=str).fillna("")
        required = {"case_id", "embedding_path"}
        if not required.issubset(df.columns):
            raise ValueError("embedding map CSV requires case_id,embedding_path")
        for _, row in df.iterrows():
            p = Path(row["embedding_path"])
            if not p.is_absolute():
                p = embeddings_dir / p
            manual[str(row["case_id"]).strip()].append(p)

    matches, report = {}, []

    for case_id in case_ids:
        submitter = str(metadata.get(case_id, {}).get("case_submitter_id", "")).strip()
        identifiers = [x for x in [case_id, submitter] if x]

        if case_id in manual:
            paths = sorted({p for p in manual[case_id] if p.exists()})
            status = "manual" if paths else "manual_missing"
        else:
            paths = []
            for p in h5_files:
                stem = p.stem
                if any(
                    stem == identifier
                    or stem.startswith(identifier + "-")
                    or stem.startswith(identifier + "_")
                    for identifier in identifiers
                ):
                    paths.append(p)
            paths = sorted(set(paths))
            status = "matched" if paths else "unmatched"

        if paths:
            matches[case_id] = paths

        report.append({
            "case_id": case_id,
            "case_submitter_id": submitter,
            "embedding_count": len(paths),
            "embedding_paths": "|".join(str(p) for p in paths),
            "status": status,
        })

    return matches, pd.DataFrame(report), len(h5_files)


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
        "tonsil", "mouth", "lip"
    ]):
        return "Head and Neck"

    return "Unknown"


def build_validation_matrix(args, output_dir):
    mutation_dir = Path(args.mutation_dir)
    signature_dir = Path(args.signature_dir)
    cna_dir = Path(args.cna_dir)

    mut, mut_cases, mut_cols, mut_labels = load_table(
        mutation_dir, "patient_mutation_matrix"
    )
    sig, sig_cases, sig_cols, sig_labels = load_table(
        signature_dir, "patient_signature_matrix"
    )
    cna, cna_cases, cna_cols, cna_labels = load_table(
        cna_dir, "patient_cna_matrix"
    )

    # mutation and CNA columns already carry their correct prefix from
    # extraction (som_mut_/cna_) -- no change needed there. SIGNATURE
    # columns arrive BARE from mutSigs-CPTAC.py (matching the real
    # TCGA-side extract_mutational_signatures.py's own bare output),
    # since the "sig__" prefix only gets added later, at fusion time,
    # by fuse_tables.py. Add it here to match.
    print(f"Sample signature columns BEFORE prefixing: {sig_cols[:3]}")
    sig_cols = [c if c.startswith(SIGNATURE_PREFIX) else f"{SIGNATURE_PREFIX}{c}" for c in sig_cols]
    print(f"Sample signature columns AFTER prefixing: {sig_cols[:3]}")

    print(f"Sample mutation columns (first 3): {mut_cols[:3]}")
    print(f"Sample CNA columns (first 3): {cna_cols[:3]}")
    if not any(c.startswith(MUTATION_PREFIX) for c in mut_cols[:20]):
        print(f"WARNING: none of the first 20 mutation columns start with "
              f"'{MUTATION_PREFIX}' -- the naming convention may not match "
              f"what this script expects. Check the sample above.")
    if not any(c.startswith(CNA_PREFIX) for c in cna_cols[:20]):
        print(f"WARNING: none of the first 20 CNA columns start with "
              f"'{CNA_PREFIX}' -- the naming convention may not match "
              f"what this script expects. Check the sample above.")
    if not any(c.startswith(SIGNATURE_PREFIX) for c in sig_cols[:20]):
        print(f"WARNING: none of the first 20 signature columns start with "
              f"'{SIGNATURE_PREFIX}' -- SIGNATURE_PREFIX at the top of this "
              f"script may need updating. Check the sample above.")

    metadata = load_metadata(mutation_dir / "patient_metadata.csv")
    cna_metadata = load_metadata(cna_dir / "patient_metadata.csv")
    for case_id, record in cna_metadata.items():
        metadata.setdefault(case_id, {}).update(
            {k: v for k, v in record.items() if v and not metadata.get(case_id, {}).get(k)}
        )

    mut_index = {case_id: i for i, case_id in enumerate(mut_cases)}
    sig_index = {case_id: i for i, case_id in enumerate(sig_cases)}
    cna_index = {case_id: i for i, case_id in enumerate(cna_cases)}

    genomic_cases = sorted(set(mut_cases) | set(sig_cases) | set(cna_cases))

    embedding_matches, match_report, total_h5 = match_embeddings(
        genomic_cases,
        metadata,
        Path(args.embeddings_dir),
        Path(args.embedding_map_csv) if args.embedding_map_csv else None,
    )
    match_report.to_csv(output_dir / "cptac_embedding_match_report.csv", index=False)

    included = sorted(
        set(mut_cases)
        & set(sig_cases)
        & set(cna_cases)
        & set(embedding_matches)
    )
    if not included:
        raise RuntimeError(
            "No patients have embedding + mutation + SBS96 + CNA data."
        )

    mut_block = np.zeros((len(included), len(mut_cols)), dtype=np.float32)
    sig_block = np.zeros((len(included), len(sig_cols)), dtype=np.float32)
    cna_block = np.full((len(included), len(cna_cols)), np.nan, dtype=np.float32)

    has_mutation, has_signature, has_cna = {}, {}, {}

    for row_idx, case_id in enumerate(included):
        has_mutation[case_id] = case_id in mut_index
        has_signature[case_id] = case_id in sig_index
        has_cna[case_id] = case_id in cna_index

        if has_mutation[case_id]:
            mut_block[row_idx] = mut[mut_index[case_id]].toarray().reshape(-1)
        if has_signature[case_id]:
            sig_block[row_idx] = sig[sig_index[case_id]].toarray().reshape(-1)
        if has_cna[case_id]:
            cna_block[row_idx] = cna[cna_index[case_id]].toarray().reshape(-1)

    source = pd.DataFrame(
        np.concatenate([mut_block, sig_block, cna_block], axis=1),
        index=included,
        columns=mut_cols + sig_cols + cna_cols,
        dtype=np.float32,
    )

    training_columns = [
        x.strip()
        for x in Path(args.align_to_columns_file).read_text().splitlines()
        if x.strip()
    ]

    aligned = pd.DataFrame(index=included, columns=training_columns, dtype=np.float32)
    counts = Counter()

    for col in training_columns:
        if col in source.columns:
            aligned[col] = source[col]
            counts["matched"] += 1
        elif col.startswith(MUTATION_PREFIX):
            aligned[col] = 0.0
            counts["missing_mut_zero"] += 1
        elif col.startswith(SIGNATURE_PREFIX):
            aligned[col] = 0.0
            counts["missing_sig_zero"] += 1
        elif col.startswith(CNA_PREFIX):
            aligned[col] = [
                0.0 if has_cna[case_id] else np.nan
                for case_id in included
            ]
            counts["missing_cna_zero_or_nan"] += 1
        else:
            aligned[col] = 0.0
            counts["other_zero"] += 1

    true_labels = {}
    for case_id in included:
        primary_site = metadata.get(case_id, {}).get("primary_site", "")
        label = map_primary_site_to_origin(primary_site)
        if label == "Unknown":
            label = (
                mut_labels.get(case_id, "")
                or sig_labels.get(case_id, "")
                or cna_labels.get(case_id, "")
                or "Unknown"
            )
        true_labels[case_id] = label

    save_npz(
        output_dir / "cptac_matrix.npz",
        csr_matrix(aligned.fillna(0).astype(np.float32).values),
    )
    aligned.to_csv(output_dir / "cptac_matrix_dense_with_nan.csv")
    (output_dir / "cptac_matrix_columns.txt").write_text("\n".join(training_columns))

    with (output_dir / "cptac_matrix_rows.txt").open("w") as f:
        for case_id in included:
            f.write(f"{true_labels[case_id]}\t{case_id}\n")

    bridge_rows = []
    for case_id in included:
        paths = embedding_matches[case_id]
        bridge_rows.append({
            "case_id": case_id,
            "case_submitter_id": metadata.get(case_id, {}).get("case_submitter_id", ""),
            "tumor_origin": true_labels[case_id],
            "has_mutation": has_mutation[case_id],
            "has_signature": has_signature[case_id],
            "has_cna": has_cna[case_id],
            "embedding_count": len(paths),
            "embedding_paths": "|".join(str(p) for p in paths),
        })
    pd.DataFrame(bridge_rows).to_csv(
        output_dir / "cptac_case_id_embedding_map.csv", index=False
    )

    modality_counts = Counter(
        "+".join(
            name for name, present in [
                ("mutation", has_mutation[c]),
                ("sbs96", has_signature[c]),
                ("cna", has_cna[c]),
            ] if present
        )
        for c in included
    )
    origin_counts = Counter(true_labels.values())

    report = [
        "CPTAC validation matrix build",
        "=" * 70,
        f"Mutation patients: {len(mut_cases)}",
        f"SBS96 patients: {len(sig_cases)}",
        f"CNA patients: {len(cna_cases)}",
        f"Patients with mutation OR SBS96 OR CNA: {len(genomic_cases)}",
        f"Embedding files found: {total_h5}",
        f"Included embedding + mutation + SBS96 + CNA patients: {len(included)}",
        "",
        "Included modality combinations:",
    ]
    report.extend(f"  {k}: {v}" for k, v in sorted(modality_counts.items()))
    report.extend([
        "",
        f"Training columns: {len(training_columns)}",
        f"Directly matched columns: {counts['matched']}",
        f"Absent mutation columns set to 0: {counts['missing_mut_zero']}",
        f"Absent signature columns set to 0: {counts['missing_sig_zero']}",
        f"Absent CNA columns set to 0 for CNA patients / NaN for no-CNA patients: "
        f"{counts['missing_cna_zero_or_nan']}",
        f"Other absent columns set to 0: {counts['other_zero']}",
        "",
        "Included tumor origins:",
    ])
    report.extend(f"  {k}: {v}" for k, v in sorted(origin_counts.items()))

    match_fraction = counts["matched"] / max(len(training_columns), 1)
    if match_fraction < 0.01:
        report.append("")
        report.append(
            f"WARNING: only {counts['matched']}/{len(training_columns)} "
            f"({match_fraction:.2%}) training columns were directly matched. "
            f"This is suspiciously low and likely indicates a column-naming "
            f"mismatch between the CPTAC tables and the training column list "
            f"-- check the sample columns printed at the start of this run "
            f"before trusting any downstream scoring."
        )

    (output_dir / "cptac_qc_report.txt").write_text("\n".join(report))
    print("\n".join(report))

    return included, true_labels, embedding_matches


def load_mil_model(model_dir, model_name, n_classes, device):
    model_dir = Path(model_dir)
    params = json.loads(
        (model_dir / f"{model_name}_final_params.json").read_text()
    )
    model = build_model(
        model_name, n_classes, params["hidden_dim"]
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
    """
    Score every slide independently and average slide probabilities to obtain
    one patient-level MIL probability vector.
    """
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
                        f"WARNING: MIL failed for {case_id}, slide {h5_path}: {error}"
                    )

            if slide_probs:
                out[case_id] = np.mean(
                    np.stack(slide_probs, axis=0),
                    axis=0,
                )

    return out


def score_mutation(model_dir, dense_csv, model_name):
    dense = pd.read_csv(dense_csv, index_col=0)
    dense.index = dense.index.astype(str)

    n_nan = int(dense.isna().sum().sum())
    print(f"Mutation scoring: filling {n_nan} NaN cells with 0 at scoring time.")

    X = csr_matrix(dense.fillna(0).astype(np.float32).values)

    artifact = joblib.load(
        Path(model_dir) / f"{model_name}_final_model.joblib"
    )
    model = artifact["model"]
    scaler = artifact.get("scaler")

    # eval_metric=["merror", "mlogloss"] can break on some XGBoost/joblib
    # version combinations at load/score time (same bug class found and
    # fixed earlier in plot_shap.py's SHAP computation) -- fixed at both
    # the sklearn wrapper and the underlying Booster, since set_params()
    # on the wrapper isn't guaranteed to propagate to an already-fitted
    # Booster. eval_metric has no effect on predictions.
    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu", eval_metric="mlogloss")
        except Exception:
            pass
    try:
        model.get_booster().set_param({"device": "cpu", "eval_metric": "mlogloss"})
    except Exception:
        pass

    X_scaled = scaler.transform(X) if scaler is not None else X

    encoder = joblib.load(Path(model_dir) / "label_encoder.joblib")
    classes = list(encoder.classes_)
    scores = get_full_width_scores(model, X_scaled, len(classes))

    return {
        case_id: scores[i]
        for i, case_id in enumerate(dense.index.tolist())
    }, classes


def train_stacking_meta_classifier(args, common_classes):
    """
    Trains a LogisticRegression meta-classifier on [mil_probs, mut_probs]
    from ALL 5 TCGA cross-validation folds' held-out test predictions --
    unlike fuse_mil_mutation.py's run_stacking_fusion() (which excludes
    ONE fold at a time, since that fold IS the thing being evaluated),
    this uses every fold's data, since CPTAC -- not any TCGA fold -- is
    the held-out set being evaluated here. Reuses
    get_mil_fold_probabilities()/get_mutation_fold_probabilities() from
    fuse_mil_mutation.py directly (each fold's own held-out predictions,
    from that fold's own trained checkpoint, not the final model) -- the
    same leakage-free construction the meta-classifier was validated
    with during cross-validation.

    Requires a small args-like namespace with the TCGA-side paths
    (embeddings, splits, matrix) -- see build_tcga_fold_args() below,
    which maps validate_CPTAC.py's own --tcga-* arguments onto the
    attribute names those imported functions expect.
    """
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
        mutation_model_dir=args.mutation_model_dir,
        mutation_model_name=args.mutation_model_name,
    )

    train_mil, train_mut, train_y = [], [], []
    class_to_idx = {c: i for i, c in enumerate(common_classes)}

    for fold_idx in range(args.n_tcga_folds):
        print(f"Loading TCGA fold {fold_idx} held-out predictions for stacking training...")
        mil_probs, mil_classes = get_mil_fold_probabilities(tcga_args, fold_idx)
        mut_probs, mut_classes = get_mutation_fold_probabilities(tcga_args, fold_idx)

        mil_aligned = align_to_common_classes(mil_probs, mil_classes, common_classes)
        mut_aligned = align_to_common_classes(mut_probs, mut_classes, common_classes)

        patient_cohort = pd.read_csv(args.tcga_patient_cohort_csv)
        case_id_to_true_label = dict(zip(patient_cohort["case_id"], patient_cohort["tumor_origin"]))

        common_case_ids = sorted(set(mil_probs) & set(mut_probs))
        for cid in common_case_ids:
            true_label = case_id_to_true_label.get(cid)
            if true_label is None or true_label not in class_to_idx:
                continue
            train_mil.append(mil_aligned[cid])
            train_mut.append(mut_aligned[cid])
            train_y.append(class_to_idx[true_label])

    train_mil = np.array(train_mil)
    train_mut = np.array(train_mut)
    train_y = np.array(train_y)
    train_X = np.concatenate([train_mil, train_mut], axis=1)

    print(f"Stacking meta-classifier: {train_X.shape[0]} training rows "
          f"(from all {args.n_tcga_folds} TCGA folds), {train_X.shape[1]} features")

    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta_model.fit(train_X, train_y)
    return meta_model


def apply_stacking_fusion(meta_model, mut_probs, mil_probs, classes, both):
    """
    Applies an already-trained stacking meta-classifier to CPTAC's
    mutation+MIL probability vectors (from the FINAL models, since
    that's what's actually being validated -- the meta-classifier
    itself was trained on TCGA fold-level held-out predictions, but is
    applied here to new, unseen CPTAC patients). Returns
    {case_id: fused_probability_vector}, remapped back to the full
    `classes` width the same way run_stacking_fusion() does, since
    meta_model.classes_ may be a subset if some class had zero training rows.
    """
    fused = {}
    for case_id in both:
        x = np.concatenate([mil_probs[case_id], mut_probs[case_id]]).reshape(1, -1)
        proba = meta_model.predict_proba(x)[0]
        full = np.zeros(len(classes))
        for i, cls_idx in enumerate(meta_model.classes_):
            full[int(cls_idx)] = proba[i]
        fused[case_id] = full
    return fused


def score_and_report(args, output_dir, included, true_labels, embedding_matches):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mut_probs, classes = score_mutation(
        args.mutation_model_dir,
        output_dir / "cptac_matrix_dense_with_nan.csv",
        args.mutation_model_name,
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
        {c: embedding_matches[c] for c in included},
        device,
    )

    both = sorted(set(included) & set(mut_probs) & set(mil_probs))

    if args.fusion_method == "stack":
        print("Training stacking meta-classifier on all TCGA folds...")
        meta_model = train_stacking_meta_classifier(args, classes)
        fused_probs = apply_stacking_fusion(meta_model, mut_probs, mil_probs, classes, both)

    results = []

    for case_id in both:
        true = true_labels.get(case_id, "")
        mut_vec = mut_probs[case_id]
        mil_vec = mil_probs[case_id]

        if args.fusion_method == "stack":
            fused_vec = fused_probs[case_id]
        else:
            fused_vec = mut_vec * mil_vec
            if fused_vec.sum() > 0:
                fused_vec = fused_vec / fused_vec.sum()

        row = {
            "case_id": case_id,
            "true_tumor_origin": true,
            "embedding_count": len(embedding_matches[case_id]),
            "embedding_paths": "|".join(str(p) for p in embedding_matches[case_id]),
        }


        for name, vec in [
            ("mutation", mut_vec),
            ("mil", mil_vec),
            ("fused", fused_vec),
        ]:
            top3 = np.argsort(-vec)[:3]
            row[f"{name}_prediction"] = classes[top3[0]]
            row[f"{name}_correct"] = classes[top3[0]] == true
            row[f"{name}_top3_correct"] = true in [classes[i] for i in top3]
            row[f"{name}_top3"] = ", ".join(
                f"{classes[i]} ({vec[i]:.4f})" for i in top3
            )

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "cptac_predictions.csv", index=False)

    valid = df[
        df["true_tumor_origin"].notna()
        & df["true_tumor_origin"].ne("")
        & df["true_tumor_origin"].ne("Unknown")
    ]

    def metric(col):
        return float(valid[col].mean()) if len(valid) else None

    lines = [
        "CPTAC external validation",
        "=" * 70,
        f"Patients scored with embedding + mutation/SBS96 vector: {len(df)}",
        f"Patients with known labels used for accuracy: {len(valid)}",
        "",
    ]

    for name in ["mutation", "mil", "fused"]:
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

    lines.extend([
        "",
        "Fusion: element-wise probability multiplication followed by normalization.",
        "CNA columns are NaN in the saved dense matrix and filled with 0 only for scoring.",
    ])

    (output_dir / "cptac_scoring_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mutation-dir", required=True)
    parser.add_argument("--signature-dir", required=True)
    parser.add_argument("--cna-dir", required=True)
    parser.add_argument("--align-to-columns-file", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--embedding-map-csv", default=None)
    parser.add_argument("--mil-model-dir", required=True)
    parser.add_argument(
        "--mil-model-name",
        default="abmil",
        choices=["abmil", "clam", "transmil"],
    )
    parser.add_argument("--mutation-model-dir", required=True)
    parser.add_argument("--mutation-model-name", default="xgboost")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--fusion-method", default="multiply", choices=["multiply", "stack"],
                         help="multiply (default) needs no extra data -- element-wise probability "
                              "product, renormalized. stack trains a logistic regression "
                              "meta-classifier on ALL 5 TCGA folds' held-out predictions (requires "
                              "the --tcga-* arguments below) and applies it to CPTAC.")
    parser.add_argument("--tcga-embeddings-dir", default=None,
                         help="Required for --fusion-method stack. Same embeddings-dir originally "
                              "passed to train_mil_models.py (TCGA .h5 files, NOT CPTAC's).")
    parser.add_argument("--tcga-patient-cohort-csv", default=None,
                         help="Required for --fusion-method stack. patient_cohort.csv from "
                              "make_splits.py (TCGA case_id/tumor_origin/site).")
    parser.add_argument("--tcga-case-id-barcode-map", default=None,
                         help="Required for --fusion-method stack. case_id_barcode_map.csv from "
                              "make_splits.py.")
    parser.add_argument("--tcga-site-fold-csv", default=None,
                         help="Required for --fusion-method stack. site_stratified_5fold.csv -- the "
                              "same TCGA fold assignment used throughout the rest of the project.")
    parser.add_argument("--tcga-mutation-matrix-prefix", default=None,
                         help="Required for --fusion-method stack. Same --matrix-prefix originally "
                              "passed to mutClassifier.py (patient_fused_matrix, TCGA data).")
    parser.add_argument("--tcga-patient-subset-file", default=None,
                         help="Optional for --fusion-method stack. Same --patient-subset-file "
                              "originally passed to train_mil_models.py, if one was used.")
    parser.add_argument("--n-tcga-folds", type=int, default=5)
    parser.add_argument("--min-class-size", type=int, default=3,
                         help="Only used by --fusion-method stack, must match the value used when "
                              "training the TCGA MIL/mutation models.")
    args = parser.parse_args()

    if args.fusion_method == "stack":
        required_tcga_args = [
            "tcga_embeddings_dir", "tcga_patient_cohort_csv", "tcga_case_id_barcode_map",
            "tcga_site_fold_csv", "tcga_mutation_matrix_prefix",
        ]
        missing = [a for a in required_tcga_args if getattr(args, a) is None]
        if missing:
            parser.error(f"--fusion-method stack requires these arguments too: "
                         f"{', '.join('--' + a.replace('_', '-') for a in missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    included, labels, embeddings = build_validation_matrix(args, output_dir)

    if not args.build_only:
        score_and_report(
            args, output_dir, included, labels, embeddings
        )


if __name__ == "__main__":
    main()