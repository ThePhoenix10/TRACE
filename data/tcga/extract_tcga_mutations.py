import argparse
import gzip
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.tcga.tumor_origin_mapping import TCGA_COHORT_TO_TUMOR_ORIGIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("extract_muts")

WANTED_COLUMNS = [
    "Hugo_Symbol",
    "Variant_Classification",
    "Variant_Type",
    "Tumor_Sample_Barcode",
    "t_depth",
    "t_alt_count",
    "t_ref_count",
    "case_id",
]

NON_SILENT_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
}

def read_maf(maf_path: Path) -> pd.DataFrame:

    with gzip.open(maf_path, "rt") as f:
        header_line_idx = 0
        for i, line in enumerate(f):
            if not line.startswith("#"):
                header_line_idx = i
                break

    df = pd.read_csv(maf_path, sep="\t", skiprows=header_line_idx, compression="gzip", low_memory=False)
    available_cols = [c for c in WANTED_COLUMNS if c in df.columns]
    return df[available_cols]


def find_maf_files(maf_dir: Path, file_ids: set) -> list:

    return [p for p in maf_dir.rglob("*.maf.gz") if p.parent.name in file_ids]


def build_patient_records(df: pd.DataFrame, non_silent_only: bool):


    if df.empty or "case_id" not in df.columns:
        return []

    case_id = df["case_id"].iloc[0]
    source = df.dropna(subset=["Hugo_Symbol"]).copy()

    if non_silent_only:
        source = source[source["Variant_Classification"].isin(NON_SILENT_CLASSES)]

    if source.empty:
        return []

    numeric_records = []

    for gene, gene_group in source.groupby("Hugo_Symbol"):
        numeric_records.append((case_id, f"som_mut_{gene}_any", 1))

        for cls in gene_group["Variant_Classification"].dropna().unique():
            numeric_records.append((case_id, f"som_mut_{gene}_{cls}", 1))


    return numeric_records


def build_and_save_matrix(numeric_records: list, output_dir: Path,
                           case_id_to_tumor_origin: dict = None) -> tuple:


    import numpy as np
    from scipy.sparse import coo_matrix, save_npz

    if not numeric_records:
        return 0, 0

    case_id_to_tumor_origin = case_id_to_tumor_origin or {}

    numeric_df = pd.DataFrame(numeric_records, columns=["case_id", "col", "value"])
    numeric_df = numeric_df.groupby(["case_id", "col"], as_index=False)["value"].max()

    case_ids, row_idx = np.unique(numeric_df["case_id"].to_numpy(), return_inverse=True)
    col_names, col_idx = np.unique(numeric_df["col"].to_numpy(), return_inverse=True)
    values = numeric_df["value"].to_numpy(dtype=float)

    sparse_matrix = coo_matrix((values, (row_idx, col_idx)),
                                shape=(len(case_ids), len(col_names))).tocsr()

    npz_path = output_dir / "patient_mutation_matrix.npz"
    save_npz(npz_path, sparse_matrix)

    rows_path = output_dir / "patient_mutation_matrix_rows.txt"
    with open(rows_path, "w") as f:
        f.write("\n".join(
            f"{case_id_to_tumor_origin.get(cid, '')}\t{cid}" for cid in case_ids
        ))

    cols_path = output_dir / "patient_mutation_matrix_columns.txt"
    with open(cols_path, "w") as f:
        f.write("\n".join(col_names))

    log.info(f"Saved sparse patient mutation matrix ({sparse_matrix.shape[0]} patients x "
             f"{sparse_matrix.shape[1]} columns, {sparse_matrix.nnz} non-zero entries) to:\n"
             f"  {npz_path} (load with scipy.sparse.load_npz)\n"
             f"  {rows_path} (tumor_origin<TAB>case_id per row)\n"
             f"  {cols_path} (column name per column)")

    return sparse_matrix.shape


def cleanup_empty_maf_patients(mutations: pd.DataFrame, matched_manifest_path: Path,
                                matched_cohort_csv_path: Path, embeddings_dir: Path,
                                non_silent_only: bool):


    matched_cohort = pd.read_csv(matched_cohort_csv_path)
    mut_manifest = pd.read_csv(matched_manifest_path, sep="\t")

    source = mutations
    if non_silent_only:
        source = mutations[mutations["Variant_Classification"].isin(NON_SILENT_CLASSES)]

    barcodes_with_mutations = set(source["patient_barcode"].dropna())
    all_barcodes = set(matched_cohort["barcode"])
    empty_barcodes = all_barcodes - barcodes_with_mutations

    log.info(f"Empty-MAF cleanup: {len(all_barcodes)} total matched patients, "
             f"{len(all_barcodes) - len(empty_barcodes)} have qualifying mutation data, "
             f"{len(empty_barcodes)} have none (empty MAF or silent-only; will be removed)")

    if not empty_barcodes:
        log.info("No empty-MAF patients to remove.")
        return 0

    matched_cohort_filtered = matched_cohort[~matched_cohort["barcode"].isin(empty_barcodes)]
    matched_cohort_filtered.to_csv(matched_cohort_csv_path, index=False)

    empty_file_ids = set(matched_cohort[matched_cohort["barcode"].isin(empty_barcodes)]["file_id"])
    mut_manifest_filtered = mut_manifest[~mut_manifest["id"].isin(empty_file_ids)]
    mut_manifest_filtered.to_csv(matched_manifest_path, sep="\t", index=False)

    excluded_dir = embeddings_dir.parent / "embeddings_excluded"
    excluded_dir.mkdir(exist_ok=True)
    moved = 0
    for barcode in empty_barcodes:
        src = embeddings_dir / f"{barcode}.h5"
        if src.exists():
            shutil.move(str(src), str(excluded_dir / f"{barcode}.h5"))
            moved += 1

    log.info(f"matched_cohort.csv: {len(matched_cohort)} -> {len(matched_cohort_filtered)} rows")
    log.info(f"{matched_manifest_path.name}: {len(mut_manifest)} -> {len(mut_manifest_filtered)} rows")
    log.info(f"Moved {moved} .h5 files to {excluded_dir} (not deleted)")

    return len(empty_barcodes)


def main():
    parser = argparse.ArgumentParser(description="Extract mutation data and build the per-patient mutation matrix")
    parser.add_argument("--maf-dir", required=True, help="Directory containing downloaded MAF files")
    parser.add_argument("--matched-manifest", required=True,
                         help="Manifest listing the matched-cohort file_ids to process")
    parser.add_argument("--output-dir", required=True, help="Where to write output tables")
    parser.add_argument("--non-silent-only-in-matrix", action="store_true", default=True,
                         help="Restrict the mutation matrix to non-silent mutations (default: on)")
    parser.add_argument("--matched-cohort-csv", default=None,
                         help="Path to matched_cohort.csv from the matching step. If provided (along "
                              "with --embeddings-dir), empty-MAF patients are removed from it too.")
    parser.add_argument("--embeddings-dir", default=None,
                         help="Path to the extracted .h5 embeddings folder. If provided (along with "
                              "--matched-cohort-csv), empty-MAF patients' .h5 files are moved to "
                              "embeddings_excluded/ next to this folder.")
    args = parser.parse_args()


    maf_dir = Path(args.maf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.matched_manifest, sep="\t")
    file_ids = set(manifest["id"].astype(str))
    log.info(f"Matched manifest has {len(file_ids)} file_ids to process")

    maf_files = find_maf_files(maf_dir, file_ids)
    log.info(f"Found {len(maf_files)} matching MAF files under {maf_dir}")

    all_rows = []
    all_numeric_records = []
    failed_files = []

    for i, maf_path in enumerate(tqdm(maf_files, desc="Reading MAFs", unit="file"), 1):
        file_id = maf_path.parent.name

        try:
            df = read_maf(maf_path)
            if df.empty:
                continue

            if {"t_alt_count", "t_depth"}.issubset(df.columns):
                df["VAF"] = df["t_alt_count"] / df["t_depth"].replace(0, pd.NA)
            else:
                df["VAF"] = pd.NA

            df["file_id"] = file_id

            if "Tumor_Sample_Barcode" in df.columns:
                df["patient_barcode"] = df["Tumor_Sample_Barcode"].str.slice(0, 12)
            else:
                df["patient_barcode"] = None

            all_rows.append(df)


            numeric_records = build_patient_records(
                df, args.non_silent_only_in_matrix
            )
            all_numeric_records.extend(numeric_records)

        except Exception as e:
            log.warning(f"Failed to read {maf_path.name}: {e}")
            failed_files.append((file_id, str(e)))

        if i % 1000 == 0:
            log.info(f"[{i}/{len(maf_files)}] MAF files processed, "
                     f"{len(all_numeric_records)} matrix records so far")

    log.info(f"Done reading MAFs: {len(all_rows)} files contributed rows, {len(failed_files)} failed")

    if not all_rows:
        log.error("No mutation rows extracted -- nothing to save. Check --maf-dir and --matched-manifest.")
        return

    mutations = pd.concat(all_rows, ignore_index=True)
    log.info(f"Combined mutation table: {len(mutations)} rows, {mutations['case_id'].nunique()} unique cases")

    long_parquet_path = output_dir / "mutations_long.parquet"
    long_csv_path = output_dir / "mutations_long.csv"
    mutations.to_parquet(long_parquet_path, index=False)
    mutations.to_csv(long_csv_path, index=False)
    log.info(f"Saved long-format mutation table to {long_parquet_path} and {long_csv_path}")


    n_empty_removed = 0
    if args.matched_cohort_csv and args.embeddings_dir:
        n_empty_removed = cleanup_empty_maf_patients(
            mutations, Path(args.matched_manifest),
            Path(args.matched_cohort_csv), Path(args.embeddings_dir),
            args.non_silent_only_in_matrix,
        )
    else:
        log.info("Skipping empty-MAF patient cleanup (--matched-cohort-csv / --embeddings-dir not both provided)")


    case_id_to_tumor_origin = {}
    if args.matched_cohort_csv:
        matched_cohort = pd.read_csv(args.matched_cohort_csv)
        barcode_to_cohort = matched_cohort.set_index("barcode")["cohort"]
        case_to_barcode = mutations.drop_duplicates("case_id").set_index("case_id")["patient_barcode"]

        for case_id, barcode in case_to_barcode.items():
            cohort = barcode_to_cohort.get(barcode)
            if cohort is not None:
                case_id_to_tumor_origin[case_id] = TCGA_COHORT_TO_TUMOR_ORIGIN.get(cohort, cohort)

        n_mapped = len(case_id_to_tumor_origin)
        log.info(f"Mapped tumor origin for {n_mapped} / {mutations['case_id'].nunique()} patients")
    else:
        log.info("Skipping tumor origin mapping (--matched-cohort-csv not provided)")


    log.info(f"Building sparse matrix from {len(all_numeric_records)} numeric records...")
    matrix_shape = build_and_save_matrix(all_numeric_records, output_dir,
                                          case_id_to_tumor_origin)


    summary_path = output_dir / "extraction_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"MAF files found: {len(maf_files)}\n")
        f.write(f"MAF files successfully read: {len(all_rows)}\n")
        f.write(f"MAF files failed: {len(failed_files)}\n")
        f.write(f"Total mutation rows: {len(mutations)}\n")
        f.write(f"Unique cases (case_id): {mutations['case_id'].nunique()}\n")
        if "patient_barcode" in mutations.columns:
            f.write(f"Unique patient barcodes: {mutations['patient_barcode'].nunique()}\n")
        if "VAF" in mutations.columns:
            f.write(f"VAF computed for {mutations['VAF'].notna().sum()} / {len(mutations)} rows\n")
        f.write(f"Empty-MAF patients removed from cohort files: {n_empty_removed}\n")
        if "Hugo_Symbol" in mutations.columns:
            f.write("\nTop 20 most frequently mutated genes (all variant classes):\n")
            top_genes = mutations["Hugo_Symbol"].value_counts().head(20)
            for gene, count in top_genes.items():
                f.write(f"  {gene}: {count}\n")
        if matrix_shape and matrix_shape[0]:
            f.write(f"\nPatient mutation matrix shape: {matrix_shape[0]} patients x "
                    f"{matrix_shape[1]} columns (saved as sparse .npz, see log for file paths)\n")
        if failed_files:
            f.write(f"\nFailed files ({len(failed_files)}):\n")
            for file_id, reason in failed_files[:50]:
                f.write(f"  {file_id}: {reason}\n")
            if len(failed_files) > 50:
                f.write(f"  ... and {len(failed_files) - 50} more\n")

    log.info(f"Wrote summary to {summary_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()