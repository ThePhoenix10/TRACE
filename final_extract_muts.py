#!/usr/bin/env python3
"""
final_extract_muts.py

ALL-GENES VARIANT -- no gene panel restriction. Every gene seen in the
matched-cohort MAF files gets its own matrix columns (subject only to the
non-silent-mutation filter). For the version restricted to the 299-gene
Bailey et al. 2018 driver panel instead, see final_extract_muts_bailey.py.

Extracts mutation records from your matched-cohort MAF files and builds a
per-patient mutation feature matrix ONE PATIENT AT A TIME as each MAF is
read -- rather than accumulating every mutation into one giant long table
and pivoting it at the end (which for a large cohort means multiple
multi-million-row pivots). Each MAF file already IS one patient's data, so
each file's contribution to the matrix is computed immediately after
reading it, producing exactly one row, which is appended to a list of
per-patient rows. The final matrix is just those rows stacked together.

MAF COLUMNS KEPT
    Hugo_Symbol, case_id, Tumor_Sample_Barcode, Variant_Classification,
    Variant_Type, t_depth, t_alt_count, t_ref_count

    Variant_Type and the VAF-related columns (t_depth/t_alt_count/
    t_ref_count) are still read and preserved in the raw audit-trail table
    (mutations_long.parquet/.csv) for reference, but are NOT turned into
    matrix feature columns -- see below.

PATIENT MUTATION MATRIX -- COLUMN DESIGN
    One row per patient (case_id). For every gene that appears in the
    (non-silent, by default) mutation set, TWO kinds of columns are
    produced, both binary:

      <Gene>                        binary -- 1 if the patient has ANY
                                     qualifying mutation in this gene
      <Gene>_<VariantClassification> binary -- 1 if the patient has that
                                     SPECIFIC classification in this gene

    (A third column type, som_mut_<Gene>_HGVSp -- presence of a recorded
    protein-level change -- was included in earlier versions of this
    matrix, but removed: direct analysis of this project's actual TCGA
    MAF data showed HGVSp is present in >99% of rows for every retained
    non-silent mutation class (missense/nonsense: 99.99%, frameshift:
    >99.6%, splice-site: >99.07%), making it almost always identical to
    the broad som_mut_<Gene>_any indicator -- redundant, not an independent signal.)

    VariantType (structural type, e.g. SNP/DEL/INS) and VAF (variant
    allele frequency) are intentionally NOT included as matrix features.

    HANDLING MULTIPLE MUTATIONS IN THE SAME GENE
    Every column here is a simple presence flag, so multiple mutations in
    the same gene never lose information the way a "pick one representative
    mutation" scheme would -- every classification the patient has in that
    gene gets its own column set to 1.

EMPTY/SILENT-ONLY-MAF PATIENT HANDLING
    Some matched patients' MAF files exist but contain zero mutation rows,
    or contain mutations that are ALL silent/non-coding (Silent,
    Splice_Region, RNA, intronic, etc. -- none of the classes counted as
    "non-silent"). Either way, that patient ends up with zero rows in the
    matrix (build_patient_records() returns nothing for them). If
    --matched-cohort-csv and --embeddings-dir are provided, this script
    also removes those patients' rows from matched_cohort.csv and from
    --matched-manifest itself, and MOVES (not deletes) their .h5
    embedding file to embeddings_excluded/ next to --embeddings-dir --
    keeping every file consistent with exactly who ends up in the matrix.

USAGE
    python final_extract_muts.py \
        --maf-dir /workspace/maf_downloads \
        --matched-manifest /workspace/matched_cohort_uni/mut_manifest_matched.txt \
        --output-dir /workspace/mutation_tables_all_genes \
        --matched-cohort-csv /workspace/matched_cohort_uni/matched_cohort.csv \
        --embeddings-dir /workspace/matched_cohort_uni/embeddings

OUTPUT (under --output-dir)
    mutations_long.parquet / .csv           every mutation call, one row each
                                             (kept for auditing/reference)
    patient_mutation_matrix.parquet / .csv  one row per patient, columns as
                                             described above
    extraction_summary.txt                  counts, top genes, empty-MAF
                                             patients removed, any failures
"""

import argparse
import gzip
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cancer_mapping import TCGA_COHORT_TO_TUMOR_ORIGIN

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
    """Reads a gzipped GDC MAF, skipping leading '#' comment lines."""
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
    """Finds MAF files under maf_dir whose UUID subfolder matches a wanted file_id."""
    return [p for p in maf_dir.rglob("*.maf.gz") if p.parent.name in file_ids]


def build_patient_records(df: pd.DataFrame, non_silent_only: bool):
    """
    Builds this ONE patient's contribution to the matrix as long-format
    records instead of a wide dict -- (case_id, column_name, value)
    triplets. Building long records (which get combined into a sparse
    matrix later) instead of a wide per-patient dict avoids ever
    materializing a dense tens-of-thousands-of-columns DataFrame, which is
    what made the previous approach hang -- pandas' fillna/astype on a
    huge, mostly-NaN wide frame is extremely slow due to block
    fragmentation.

    Per gene, produces:
      som_mut_<Gene>_any              binary -- any non-silent mutation in this gene
      som_mut_<Gene>_<class>          binary -- one column per Variant_Classification
                           actually observed for that gene in this patient
                           (VariantType and VAF are intentionally NOT
                           included as matrix features; a third column
                           type, som_mut_<Gene>_HGVSp, was removed -- see the
                           module docstring for why).

    Processes ALL genes seen in the data -- no gene panel restriction.
    """
    if df.empty or "case_id" not in df.columns:
        return []

    case_id = df["case_id"].iloc[0]
    source = df.dropna(subset=["Hugo_Symbol"]).copy()

    if non_silent_only:
        source = source[source["Variant_Classification"].isin(NON_SILENT_CLASSES)]

    if source.empty:
        return []

    numeric_records = []  # (case_id, column, value)

    for gene, gene_group in source.groupby("Hugo_Symbol"):
        numeric_records.append((case_id, f"som_mut_{gene}_any", 1))

        for cls in gene_group["Variant_Classification"].dropna().unique():
            numeric_records.append((case_id, f"som_mut_{gene}_{cls}", 1))

        # NOTE: som_mut_<Gene>_HGVSp columns were deliberately removed here.
        # Direct analysis of this project's actual TCGA MAF data showed
        # HGVSp is present in >99% of rows for every retained non-silent
        # mutation class (missense/nonsense: 99.99%, frameshift: >99.6%,
        # splice-site: >99.07%), making som_mut_<Gene>_HGVSp almost always
        # identical to the broad som_mut_<Gene>_any indicator -- redundant rather
        # than an independent biological signal, not worth the extra
        # feature-space cost or the risk of it being misread in SHAP
        # analysis as "protein-level consequence matters" when it's
        # really just restating gene-mutation status.

    return numeric_records


def build_and_save_matrix(numeric_records: list, output_dir: Path,
                           case_id_to_tumor_origin: dict = None) -> tuple:
    """
    Builds the mutation matrix as a genuine SPARSE matrix (scipy.sparse)
    and saves it directly in sparse form -- it is NEVER converted to a
    dense array at any point, including at save time. This matters because
    even after fixing construction to be sparse, converting to a dense
    DataFrame just to call .to_csv()/.to_parquet() would still allocate
    the full (n_patients x n_columns) dense array -- at real scale
    (tens of thousands of columns), that's tens of GB on its own, enough
    to crash again even with sparse construction upstream.

    Saved as:
      patient_mutation_matrix.npz          scipy sparse matrix (load with
                                            scipy.sparse.load_npz)
      patient_mutation_matrix_rows.txt     "tumor_origin\tcase_id" per line, in
                                            order (tumor_origin is blank if
                                            case_id_to_tumor_origin wasn't
                                            provided or had no entry for that
                                            patient) -- row N here matches
                                            row N of the .npz matrix
      patient_mutation_matrix_columns.txt  column name for each column, in
                                            order (matches the .npz shape)

    Returns (n_patients, n_columns) for logging/summary purposes.
    """
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
    """
    Removes patients who don't end up with a row in the matrix from
    matched_cohort.csv and the matched manifest, and MOVES (does not
    delete) their .h5 embedding file to embeddings_excluded/. This means
    both patients with a truly empty MAF (zero mutation rows at all) AND
    -- when non_silent_only is set, matching the matrix's own filter --
    patients whose MAF has mutations but NONE of them are non-silent
    (e.g. every mutation is Silent/Splice_Region/RNA/intronic). Both
    cases leave that patient with zero rows in build_patient_records(),
    so both need to be excluded here for matched_cohort.csv/embeddings/
    to stay consistent with what's actually in the matrix.
    """
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

    all_rows = []            # for mutations_long (raw record, unchanged)
    all_numeric_records = [] # for the matrix -- (case_id, col, value) triplets
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

            # build this ONE patient's contribution right now, as long-format
            # records, from just their own MAF -- no giant cross-patient
            # wide table needed
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

    # --- empty/silent-only-MAF patient cleanup ---
    n_empty_removed = 0
    if args.matched_cohort_csv and args.embeddings_dir:
        n_empty_removed = cleanup_empty_maf_patients(
            mutations, Path(args.matched_manifest),
            Path(args.matched_cohort_csv), Path(args.embeddings_dir),
            args.non_silent_only_in_matrix,
        )
    else:
        log.info("Skipping empty-MAF patient cleanup (--matched-cohort-csv / --embeddings-dir not both provided)")

    # --- build case_id -> tumor_origin mapping (TCGA project code translated
    # to its full tumor origin name), if --matched-cohort-csv was provided ---
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

    # --- build and save the matrix as a genuine sparse matrix, saved
    # directly in sparse form (never converted to dense, at build OR save time) ---
    log.info(f"Building sparse matrix from {len(all_numeric_records)} numeric records...")
    matrix_shape = build_and_save_matrix(all_numeric_records, output_dir,
                                          case_id_to_tumor_origin)

    # --- summary ---
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