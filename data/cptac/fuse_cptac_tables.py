import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

MUTATION_PREFIX = "som_mut_"
CNA_PREFIX = "cna_"
SIGNATURE_PREFIX = "sig__"


def read_rows(path):
    case_ids = []
    labels = {}

    for line in Path(path).read_text().splitlines():
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


def load_table(directory, stem):
    directory = Path(directory)
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


def load_metadata(path):
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str).fillna("")
    return {str(row["case_id"]): row.to_dict() for _, row in df.iterrows()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation-dir", required=True)
    parser.add_argument("--signature-dir", required=True)
    parser.add_argument("--cna-dir", required=True)
    parser.add_argument("--align-to-columns-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    mutation_dir = Path(args.mutation_dir)
    signature_dir = Path(args.signature_dir)
    cna_dir = Path(args.cna_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mut, mut_cases, mut_cols, mut_labels = load_table(
        mutation_dir, "patient_mutation_matrix"
    )
    sig, sig_cases, sig_cols, sig_labels = load_table(
        signature_dir, "patient_signature_matrix"
    )
    cna, cna_cases, cna_cols, cna_labels = load_table(
        cna_dir, "patient_cna_matrix"
    )

    sig_cols = [
        col if col.startswith(SIGNATURE_PREFIX) else f"{SIGNATURE_PREFIX}{col}"
        for col in sig_cols
    ]

    metadata = load_metadata(mutation_dir / "patient_metadata.csv")
    cna_metadata = load_metadata(cna_dir / "patient_metadata.csv")

    for case_id, record in cna_metadata.items():
        metadata.setdefault(case_id, {}).update(
            {
                key: value
                for key, value in record.items()
                if value and not metadata.get(case_id, {}).get(key)
            }
        )

    mut_index = {case_id: i for i, case_id in enumerate(mut_cases)}
    sig_index = {case_id: i for i, case_id in enumerate(sig_cases)}
    cna_index = {case_id: i for i, case_id in enumerate(cna_cases)}

    included = sorted(
        set(mut_cases)
        & set(sig_cases)
        & set(cna_cases)
    )

    if not included:
        raise RuntimeError("No patients have mutation + SBS96 + CNA data.")

    mut_block = np.zeros((len(included), len(mut_cols)), dtype=np.float32)
    sig_block = np.zeros((len(included), len(sig_cols)), dtype=np.float32)
    cna_block = np.full((len(included), len(cna_cols)), np.nan, dtype=np.float32)

    for row_idx, case_id in enumerate(included):
        mut_block[row_idx] = mut[mut_index[case_id]].toarray().reshape(-1)
        sig_block[row_idx] = sig[sig_index[case_id]].toarray().reshape(-1)
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

    aligned = pd.DataFrame(
        index=included,
        columns=training_columns,
        dtype=np.float32,
    )
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
            aligned[col] = 0.0
            counts["missing_cna_zero"] += 1
        else:
            aligned[col] = 0.0
            counts["other_zero"] += 1

    save_npz(
        output_dir / "cptac_matrix.npz",
        csr_matrix(aligned.astype(np.float32).values),
    )
    aligned.to_csv(output_dir / "cptac_matrix_dense.csv")
    (output_dir / "cptac_matrix_columns.txt").write_text(
        "\n".join(training_columns)
    )

    with (output_dir / "cptac_matrix_rows.txt").open("w") as handle:
        for case_id in included:
            source_label = (
                mut_labels.get(case_id, "")
                or sig_labels.get(case_id, "")
                or cna_labels.get(case_id, "")
            )
            handle.write(f"{source_label}\t{case_id}\n")

    metadata_rows = []
    for case_id in included:
        row = {"case_id": case_id}
        row.update(metadata.get(case_id, {}))
        row["case_id"] = case_id
        metadata_rows.append(row)

    pd.DataFrame(metadata_rows).to_csv(
        output_dir / "cptac_patient_metadata.csv",
        index=False,
    )

    report = [
        "CPTAC fused genomic matrix",
        f"Patients: {len(included)}",
        f"Training columns: {len(training_columns)}",
        f"Matched columns: {counts['matched']}",
        f"Absent mutation columns set to 0: {counts['missing_mut_zero']}",
        f"Absent SBS96 columns set to 0: {counts['missing_sig_zero']}",
        f"Absent CNA columns set to 0: {counts['missing_cna_zero']}",
        f"Other absent columns set to 0: {counts['other_zero']}",
    ]

    (output_dir / "cptac_fusion_report.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
