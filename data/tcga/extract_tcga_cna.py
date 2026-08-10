import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm

from data.tcga.tumor_origin_mapping import TCGA_COHORT_TO_TUMOR_ORIGIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("extract_cna_gistic")


def load_barcode_to_case_id(case_id_barcode_map_path: Path) -> dict:
    df = pd.read_csv(case_id_barcode_map_path)
    return dict(zip(df["barcode"].astype(str), df["case_id"].astype(str)))


def load_barcode_to_tumor_origin(matched_cohort_csv: Path) -> dict:
    matched_cohort = pd.read_csv(matched_cohort_csv)
    return {
        str(row["barcode"]): TCGA_COHORT_TO_TUMOR_ORIGIN.get(
            row["cohort"], row["cohort"]
        )
        for _, row in matched_cohort.iterrows()
    }


def get_alteration_flags(gene_name: str, value: float) -> list[str]:


    if pd.isna(value):
        return []

    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return []

    state_suffix = {
        -2: "del",
        -1: "loss",
         1: "gain",
         2: "amp",
    }.get(v)

    if state_suffix is None:
        return []

    return [
        f"cna_{gene_name}_any",
        f"cna_{gene_name}_{state_suffix}",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Extract binary CNA features from a GISTIC2 thresholded-by-genes file"
    )
    parser.add_argument(
        "--gistic-file", required=True,
        help="Path to the GISTIC2 all_thresholded.by_genes file (.gz OK)"
    )
    parser.add_argument(
        "--case-id-barcode-map", required=True,
        help="case_id_barcode_map.csv from make_splits.py"
    )
    parser.add_argument(
        "--matched-cohort-csv", required=True,
        help="matched_cohort.csv for barcode -> tumor_origin"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading identity bridge files...")
    barcode_to_case_id = load_barcode_to_case_id(Path(args.case_id_barcode_map))
    barcode_to_tumor_origin = load_barcode_to_tumor_origin(Path(args.matched_cohort_csv))
    log.info(
        f"case_id bridge: {len(barcode_to_case_id)} patients, "
        f"tumor_origin lookup: {len(barcode_to_tumor_origin)} patients"
    )

    log.info(
        f"Loading GISTIC2 file from {args.gistic_file} "
        "(this can take a minute -- it's a large wide matrix)..."
    )
    gistic_df = pd.read_csv(args.gistic_file, sep="\t", index_col=0)
    log.info(f"Loaded {gistic_df.shape[0]} genes x {gistic_df.shape[1]} samples")

    numeric_records = []
    case_id_to_tumor_origin = {}
    n_no_identity = 0
    n_matched_samples = 0

    for sample_col in tqdm(gistic_df.columns, desc="Processing samples", unit="sample"):
        short_barcode = str(sample_col)[:12]
        case_id = barcode_to_case_id.get(short_barcode)
        tumor_origin = barcode_to_tumor_origin.get(short_barcode)

        if case_id is None or tumor_origin is None:
            n_no_identity += 1
            continue

        n_matched_samples += 1
        column_values = gistic_df[sample_col]
        nonzero = column_values[column_values != 0]

        for gene_name, value in nonzero.items():
            for flag in get_alteration_flags(str(gene_name), value):
                numeric_records.append((case_id, flag))

        case_id_to_tumor_origin[case_id] = tumor_origin

    log.info(
        f"Matched {n_matched_samples} samples to a case_id/tumor_origin; "
        f"{n_no_identity} samples in the GISTIC2 file had no match in this project's cohort"
    )

    if not numeric_records:
        log.error("No CNA records produced -- nothing to save.")
        return

    log.info(f"Building sparse matrix from {len(numeric_records)} numeric records...")
    records_df = pd.DataFrame(numeric_records, columns=["case_id", "col"]).drop_duplicates()

    case_ids, row_idx = np.unique(records_df["case_id"].to_numpy(), return_inverse=True)
    col_names, col_idx = np.unique(records_df["col"].to_numpy(), return_inverse=True)
    values = np.ones(len(records_df), dtype=np.int64)

    sparse_matrix = coo_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(case_ids), len(col_names)),
    ).tocsr()

    npz_path = output_dir / "patient_cna_matrix.npz"
    save_npz(npz_path, sparse_matrix)

    rows_path = output_dir / "patient_cna_matrix_rows.txt"
    with open(rows_path, "w") as f:
        f.write(
            "\n".join(
                f"{case_id_to_tumor_origin.get(cid, '')}\t{cid}"
                for cid in case_ids
            )
        )

    cols_path = output_dir / "patient_cna_matrix_columns.txt"
    with open(cols_path, "w") as f:
        f.write("\n".join(col_names))

    n_any = sum(name.endswith("_any") for name in col_names)
    n_del = sum(name.endswith("_del") for name in col_names)
    n_loss = sum(name.endswith("_loss") for name in col_names)
    n_gain = sum(name.endswith("_gain") for name in col_names)
    n_amp = sum(name.endswith("_amp") for name in col_names)

    log.info(
        f"Saved binary CNA matrix ({sparse_matrix.shape[0]} patients x "
        f"{sparse_matrix.shape[1]} columns, {sparse_matrix.nnz} non-zero entries) to:\n"
        f"  {npz_path}\n"
        f"  {rows_path}\n"
        f"  {cols_path}"
    )

    summary_path = output_dir / "cna_extraction_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"GISTIC2 file samples: {gistic_df.shape[1]}\n")
        f.write(f"GISTIC2 file genes: {gistic_df.shape[0]}\n")
        f.write(f"Samples matched to this project's cohort: {n_matched_samples}\n")
        f.write(f"Samples with no match: {n_no_identity}\n")
        f.write(f"Final patient count: {len(case_ids)}\n")
        f.write(f"Final column count: {len(col_names)}\n")
        f.write("\nCNA feature columns by type:\n")
        f.write(f"  any: {n_any}\n")
        f.write(f"  del: {n_del}\n")
        f.write(f"  loss: {n_loss}\n")
        f.write(f"  gain: {n_gain}\n")
        f.write(f"  amp: {n_amp}\n")

    log.info(f"Saved summary to {summary_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()