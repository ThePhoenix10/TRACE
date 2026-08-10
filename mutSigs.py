#!/usr/bin/env python3
"""
extract_mutational_signatures.py

Builds the standard 96-channel SBS (Single Base Substitution) trinucleotide
context feature matrix, one row per patient -- the same underlying
representation used by mutational signature tools (SigProfiler,
deconstructSigs, etc.), computed directly from your matched-cohort MAF
files. Does NOT require CNA/fusion data -- only SNV positions, which you
already have, plus a reference genome to look up the base immediately
before and after each mutation.

OUTPUT FORMAT MATCHES final_extract_muts.py's PATIENT MUTATION MATRIX
    Same three-file structure, same "tumor_origin<TAB>case_id per row"
    rows file, same binary values -- so this table and the mutation
    matrix are directly stackable/joinable by case_id for a later fusion
    step, without needing any format translation between them:
      patient_signature_matrix.npz          scipy sparse binary matrix
      patient_signature_matrix_rows.txt     "tumor_origin\tcase_id" per line
      patient_signature_matrix_columns.txt  the 96 channel names, in order

    Each of the 96 columns is BINARY: 1 if the patient has at least one
    SNV in that trinucleotide context class, 0 otherwise. This is a
    presence/absence encoding, matching the mutation matrix's binary gene
    columns -- not a continuous proportion. (An earlier version of this
    script used continuous proportions; this one is binary by design so
    every table in the fusion pipeline shares the same format.)

WHY THIS MATTERS
    A gene-level binary mutation matrix throws away WHERE in the genome
    and WHAT KIND of base change occurred. Two patients can both have
    "TP53 mutated = 1" while one got there via a C>T change at a UV-damage
    hotspot (melanoma signature) and the other via a C>A change typical of
    smoking damage (lung cancer signature) -- the trinucleotide context
    captures that difference, which is often highly tumor-origin-specific.

THE 96 CHANNELS
    Only single-base substitutions (Variant_Type == "SNP") are used --
    indels don't have a trinucleotide context in this scheme. Each SNV is
    described by:
      - the mutated base and what it changed to (6 possible substitution
        types, using the pyrimidine-normalized convention: C>A, C>G, C>T,
        T>A, T>C, T>G -- if the reference base is a purine (A or G), the
        whole trinucleotide + substitution is reverse-complemented first,
        since by convention signatures are always expressed relative to
        a pyrimidine (C or T) reference base)
      - the base immediately before and after it (4 x 4 = 16 possible
        flanking combinations)
      6 substitution types x 16 contexts = 96 channels total.

TUMOR ORIGIN
    Sourced the same way as final_extract_muts.py: --matched-cohort-csv
    (barcode -> TCGA cohort code) joined against each patient's own
    Tumor_Sample_Barcode (first 12 characters = patient barcode), then
    translated to the full tumor origin name via the same TCGA cohort
    mapping table.

USAGE
    python extract_mutational_signatures.py \
        --maf-dir /workspace/maf_downloads \
        --matched-manifest /workspace/matched_cohort_uni/mut_manifest_matched.txt \
        --matched-cohort-csv /workspace/matched_cohort_uni/matched_cohort.csv \
        --reference-fasta /workspace/hg38.fa \
        --output-dir /workspace/mutation_tables

REQUIREMENTS
    pip install --break-system-packages pyfaidx pandas scipy tqdm

OUTPUT (under --output-dir)
    patient_signature_matrix.npz / _rows.txt / _columns.txt   (see above)
    mutational_signatures_summary.txt
        counts, any patients skipped (e.g. zero SNVs, or a chromosome/
        position the reference FASTA didn't have)
"""

import argparse
import gzip
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm

from cancer_mapping import TCGA_COHORT_TO_TUMOR_ORIGIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("extract_mutational_signatures")

WANTED_COLUMNS = [
    "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2",
    "Variant_Type", "Tumor_Sample_Barcode", "case_id",
]

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
PYRIMIDINES = {"C", "T"}
SUBSTITUTION_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]

# build the fixed, ordered list of all 96 channel names once, e.g. "A[C>A]A"
CHANNEL_NAMES = [
    f"{five}[{sub}]{three}"
    for sub in SUBSTITUTION_TYPES
    for five in BASES
    for three in BASES
]

def get_channel(ref: str, alt: str, trinucleotide: str) -> str:
    """
    Returns the 96-channel name for one SNV, e.g. "A[C>T]G". Normalizes to
    the pyrimidine (C/T) reference-base convention: if ref is a purine
    (A/G), reverse-complement the whole trinucleotide and the ref/alt
    bases before building the channel name.
    """
    five, center, three = trinucleotide[0], trinucleotide[1], trinucleotide[2]

    if ref not in PYRIMIDINES:
        ref = COMPLEMENT[ref]
        alt = COMPLEMENT[alt]
        five, three = COMPLEMENT[three], COMPLEMENT[five]

    return f"{five}[{ref}>{alt}]{three}"


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


def build_patient_channels(df: pd.DataFrame, genome) -> set:
    """
    Returns the SET of 96-channel names this patient has AT LEAST ONE
    SNV in (binary presence, not counts). Indels (Variant_Type != "SNP")
    are skipped -- they don't have a trinucleotide context in this scheme.
    """
    channels_present = set()

    snvs = df[df["Variant_Type"] == "SNP"].dropna(
        subset=["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]
    )

    for _, row in snvs.iterrows():
        chrom = str(row["Chromosome"])
        pos = int(row["Start_Position"])
        ref = str(row["Reference_Allele"]).upper()
        alt = str(row["Tumor_Seq_Allele2"]).upper()

        if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
            continue  # skip anything not a clean single-base substitution

        try:
            # pyfaidx is 0-based; MAF Start_Position is 1-based, and we
            # want 1 base before through 1 base after the mutated base
            trinucleotide = str(genome[chrom][pos - 2:pos + 1]).upper()
        except (KeyError, ValueError):
            continue  # chromosome not in reference, or position out of range

        if len(trinucleotide) != 3 or trinucleotide[1] != ref:
            continue  # reference mismatch -- skip rather than silently miscount

        channels_present.add(get_channel(ref, alt, trinucleotide))

    return channels_present


def build_barcode_to_tumor_origin(matched_cohort_csv: Path) -> dict:
    matched_cohort = pd.read_csv(matched_cohort_csv)
    return {
        row["barcode"]: TCGA_COHORT_TO_TUMOR_ORIGIN.get(row["cohort"], row["cohort"])
        for _, row in matched_cohort.iterrows()
    }


def main():
    parser = argparse.ArgumentParser(description="Extract binary 96-channel SBS mutational signature features")
    parser.add_argument("--maf-dir", required=True)
    parser.add_argument("--matched-manifest", required=True)
    parser.add_argument("--matched-cohort-csv", default=None,
                         help="Path to matched_cohort.csv, used to attach tumor_origin to each patient "
                              "(same source as final_extract_muts.py). If omitted, tumor_origin is left "
                              "blank in the rows file.")
    parser.add_argument("--reference-fasta", required=True,
                         help="Path to a GRCh38 reference FASTA (e.g. hg38.fa from UCSC)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from pyfaidx import Fasta

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading reference genome from {args.reference_fasta}...")
    genome = Fasta(args.reference_fasta)

    barcode_to_tumor_origin = {}
    if args.matched_cohort_csv:
        barcode_to_tumor_origin = build_barcode_to_tumor_origin(Path(args.matched_cohort_csv))
        log.info(f"Loaded tumor_origin mapping for {len(barcode_to_tumor_origin)} patients")

    manifest = pd.read_csv(args.matched_manifest, sep="\t")
    file_ids = set(manifest["id"].astype(str))
    log.info(f"Matched manifest has {len(file_ids)} file_ids to process")

    maf_files = find_maf_files(Path(args.maf_dir), file_ids)
    log.info(f"Found {len(maf_files)} matching MAF files")

    numeric_records = []  # (case_id, channel) pairs -- presence only, value always 1
    case_id_to_tumor_origin = {}
    failed_files = []
    zero_snv_patients = 0

    for maf_path in tqdm(maf_files, desc="Computing trinucleotide contexts", unit="file"):
        file_id = maf_path.parent.name
        try:
            df = read_maf(maf_path)
            if df.empty or "case_id" not in df.columns:
                continue

            case_id = df["case_id"].iloc[0]
            channels_present = build_patient_channels(df, genome)

            if not channels_present:
                zero_snv_patients += 1
                continue

            for channel in channels_present:
                numeric_records.append((case_id, channel))

            if "Tumor_Sample_Barcode" in df.columns and barcode_to_tumor_origin:
                patient_barcode = str(df["Tumor_Sample_Barcode"].iloc[0])[:12]
                tumor_origin = barcode_to_tumor_origin.get(patient_barcode, "")
                case_id_to_tumor_origin[case_id] = tumor_origin

        except Exception as e:
            log.warning(f"Failed on {maf_path.name}: {e}")
            failed_files.append((file_id, str(e)))

    if not numeric_records:
        log.error("No patients produced signature features -- nothing to save.")
        return

    # --- build the sparse binary matrix, same pattern as final_extract_muts.py ---
    records_df = pd.DataFrame(numeric_records, columns=["case_id", "channel"]).drop_duplicates()

    case_ids, row_idx = np.unique(records_df["case_id"].to_numpy(), return_inverse=True)
    # fix column order to the canonical 96-channel ordering, not whatever
    # order they happened to appear in the data
    col_to_idx = {name: i for i, name in enumerate(CHANNEL_NAMES)}
    col_idx = records_df["channel"].map(col_to_idx).to_numpy()

    values = np.ones(len(records_df), dtype=np.int64)
    sparse_matrix = coo_matrix(
        (values, (row_idx, col_idx)), shape=(len(case_ids), len(CHANNEL_NAMES))
    ).tocsr()

    npz_path = output_dir / "patient_signature_matrix.npz"
    save_npz(npz_path, sparse_matrix)

    rows_path = output_dir / "patient_signature_matrix_rows.txt"
    with open(rows_path, "w") as f:
        f.write("\n".join(f"{case_id_to_tumor_origin.get(cid, '')}\t{cid}" for cid in case_ids))

    cols_path = output_dir / "patient_signature_matrix_columns.txt"
    with open(cols_path, "w") as f:
        f.write("\n".join(CHANNEL_NAMES))

    log.info(f"Saved binary signature matrix ({sparse_matrix.shape[0]} patients x "
             f"{sparse_matrix.shape[1]} channels, {sparse_matrix.nnz} non-zero entries) to:\n"
             f"  {npz_path} (load with scipy.sparse.load_npz)\n"
             f"  {rows_path} (tumor_origin<TAB>case_id per row)\n"
             f"  {cols_path} (96 channel names, in order)")

    summary_path = output_dir / "mutational_signatures_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"MAF files found: {len(maf_files)}\n")
        f.write(f"Patients with signature features: {len(case_ids)}\n")
        f.write(f"Patients skipped (zero usable SNVs): {zero_snv_patients}\n")
        f.write(f"Files failed: {len(failed_files)}\n")
        if failed_files:
            f.write("\nFailed files:\n")
            for file_id, reason in failed_files[:50]:
                f.write(f"  {file_id}: {reason}\n")

    log.info(f"Saved summary to {summary_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()