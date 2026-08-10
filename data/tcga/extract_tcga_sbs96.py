import argparse
import gzip
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
log = logging.getLogger("extract_mutational_signatures")

WANTED_COLUMNS = [
    "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2",
    "Variant_Type", "Tumor_Sample_Barcode", "case_id",
]

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
PYRIMIDINES = {"C", "T"}
SUBSTITUTION_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]


CHANNEL_NAMES = [
    f"{five}[{sub}]{three}"
    for sub in SUBSTITUTION_TYPES
    for five in BASES
    for three in BASES
]

def get_channel(ref: str, alt: str, trinucleotide: str) -> str:


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
            continue

        try:


            trinucleotide = str(genome[chrom][pos - 2:pos + 1]).upper()
        except (KeyError, ValueError):
            continue

        if len(trinucleotide) != 3 or trinucleotide[1] != ref:
            continue

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

    numeric_records = []
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


    records_df = pd.DataFrame(numeric_records, columns=["case_id", "channel"]).drop_duplicates()

    case_ids, row_idx = np.unique(records_df["case_id"].to_numpy(), return_inverse=True)


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