import argparse
import csv
import gzip
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("mutSigs-CPTAC")


REQUIRED_MUTATION_COLUMNS = [
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Variant_Type",
]

OPTIONAL_COLUMNS = [
    "case_id",
    "Tumor_Sample_Barcode",
    "Matched_Norm_Sample_Barcode",
]

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}

PYRIMIDINES = {"C", "T"}

SUBSTITUTION_TYPES = [
    "C>A",
    "C>G",
    "C>T",
    "T>A",
    "T>C",
    "T>G",
]

BASES = ["A", "C", "G", "T"]

CHANNEL_NAMES = [
    f"{five_prime}[{substitution}]{three_prime}"
    for substitution in SUBSTITUTION_TYPES
    for five_prime in BASES
    for three_prime in BASES
]

CHANNEL_TO_INDEX = {
    channel: index
    for index, channel in enumerate(CHANNEL_NAMES)
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract binary SBS96 trinucleotide-context features "
            "from CPTAC GDC MAF files."
        )
    )

    parser.add_argument(
        "--maf-dir",
        required=True,
        help=(
            "Root directory containing GDC-downloaded .maf.gz files. "
            "Each file is normally inside a directory named with its "
            "GDC file UUID."
        ),
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="GDC manifest used to download the CPTAC MAF files.",
    )

    parser.add_argument(
        "--reference-fasta",
        required=True,
        help="GRCh38 reference FASTA file, for example /workspace/hg38.fa.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which the SBS96 outputs will be saved.",
    )

    parser.add_argument(
        "--cohort-map-csv",
        default=None,
        help=(
            "Optional CSV mapping case_id to tumor_origin or cohort. "
            "Expected columns include case_id and either tumor_origin "
            "or cohort."
        ),
    )

    parser.add_argument(
        "--keep-duplicate-mafs",
        action="store_true",
        help=(
            "Process all MAF files even when multiple manifest records "
            "have the same filename. Patient-level channels are still "
            "deduplicated."
        ),
    )

    return parser.parse_args()


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_case_id(value) -> str:
    value = normalize_text(value)

    if value.lower() in {"", "nan", "none"}:
        return ""

    return value


def detect_maf_header_line(maf_path: Path) -> int:


    with gzip.open(maf_path, "rt", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if not line.startswith("#"):
                return line_number

    raise ValueError("No MAF header line was found")


def read_maf(maf_path: Path) -> pd.DataFrame:
    header_line = detect_maf_header_line(maf_path)

    header = pd.read_csv(
        maf_path,
        sep="\t",
        compression="gzip",
        skiprows=header_line,
        nrows=0,
    )

    available_columns = [
        column
        for column in REQUIRED_MUTATION_COLUMNS + OPTIONAL_COLUMNS
        if column in header.columns
    ]

    missing_required = [
        column
        for column in REQUIRED_MUTATION_COLUMNS
        if column not in header.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    return pd.read_csv(
        maf_path,
        sep="\t",
        compression="gzip",
        skiprows=header_line,
        usecols=available_columns,
        low_memory=False,
    )


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype=str,
    )

    required_columns = {"id", "filename"}

    missing = required_columns.difference(manifest.columns)

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )

    manifest["id"] = manifest["id"].astype(str).str.strip()
    manifest["filename"] = manifest["filename"].astype(str).str.strip()

    return manifest


def find_maf_files(
    maf_dir: Path,
    manifest: pd.DataFrame,
    keep_duplicate_mafs: bool,
) -> List[Path]:


    file_ids = set(manifest["id"])

    files = [
        path
        for path in maf_dir.rglob("*.maf.gz")
        if path.parent.name in file_ids
    ]

    files = sorted(files)

    if keep_duplicate_mafs:
        return files

    unique_files = []
    seen = set()

    for path in files:
        key = (
            path.parent.name,
            path.name,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_files.append(path)

    return unique_files


def load_cohort_mapping(
    mapping_path: Optional[Path],
) -> Dict[str, str]:
    if mapping_path is None:
        return {}

    mapping_df = pd.read_csv(
        mapping_path,
        dtype=str,
    )

    if "case_id" not in mapping_df.columns:
        raise ValueError(
            "Cohort mapping CSV must contain a case_id column"
        )

    if "tumor_origin" in mapping_df.columns:
        label_column = "tumor_origin"
    elif "cohort" in mapping_df.columns:
        label_column = "cohort"
    else:
        raise ValueError(
            "Cohort mapping CSV must contain tumor_origin or cohort"
        )

    mapping = {}

    for _, row in mapping_df.iterrows():
        case_id = normalize_case_id(row["case_id"])
        label = normalize_text(row[label_column])

        if case_id:
            mapping[case_id] = label

    return mapping


def get_case_id_from_group(
    group: pd.DataFrame,
    maf_path: Path,
) -> str:


    if "case_id" in group.columns:
        values = [
            normalize_case_id(value)
            for value in group["case_id"].dropna().unique()
        ]

        values = [
            value
            for value in values
            if value
        ]

        if values:
            return values[0]

    if "Tumor_Sample_Barcode" in group.columns:
        values = [
            normalize_case_id(value)
            for value in group[
                "Tumor_Sample_Barcode"
            ].dropna().unique()
        ]

        values = [
            value
            for value in values
            if value
        ]

        if values:
            return values[0]

    return maf_path.parent.name


def get_patient_groups(
    maf_df: pd.DataFrame,
    maf_path: Path,
) -> Iterable[Tuple[str, pd.DataFrame]]:


    if "case_id" in maf_df.columns:
        valid_case_ids = maf_df["case_id"].map(
            normalize_case_id
        )

        if valid_case_ids.ne("").any():
            working = maf_df.copy()
            working["_group_case_id"] = valid_case_ids

            for case_id, group in working.groupby(
                "_group_case_id",
                sort=False,
            ):
                if not case_id:
                    continue

                yield case_id, group.drop(
                    columns=["_group_case_id"]
                )

            return

    if "Tumor_Sample_Barcode" in maf_df.columns:
        valid_barcodes = maf_df[
            "Tumor_Sample_Barcode"
        ].map(normalize_case_id)

        if valid_barcodes.ne("").any():
            working = maf_df.copy()
            working["_group_case_id"] = valid_barcodes

            for case_id, group in working.groupby(
                "_group_case_id",
                sort=False,
            ):
                if not case_id:
                    continue

                yield case_id, group.drop(
                    columns=["_group_case_id"]
                )

            return

    fallback_case_id = get_case_id_from_group(
        maf_df,
        maf_path,
    )

    yield fallback_case_id, maf_df


def normalize_chromosome(value) -> str:
    chromosome = normalize_text(value)

    if chromosome.endswith(".0"):
        chromosome = chromosome[:-2]

    return chromosome


def chromosome_candidates(chromosome: str) -> List[str]:


    chromosome = normalize_chromosome(chromosome)

    candidates = [chromosome]

    if chromosome.startswith("chr"):
        candidates.append(chromosome[3:])
    else:
        candidates.append(f"chr{chromosome}")

    if chromosome in {"M", "MT", "chrM", "chrMT"}:
        candidates.extend(
            ["M", "MT", "chrM", "chrMT"]
        )

    unique_candidates = []

    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def fetch_trinucleotide(
    genome: Fasta,
    chromosome: str,
    position: int,
) -> Optional[str]:


    if position < 2:
        return None

    for candidate in chromosome_candidates(chromosome):
        try:
            trinucleotide = str(
                genome[candidate][position - 2:position + 1]
            ).upper()

            if len(trinucleotide) == 3:
                return trinucleotide

        except (KeyError, ValueError, IndexError):
            continue

    return None


def reverse_complement_base(base: str) -> str:
    return COMPLEMENT.get(base, "N")


def get_sbs96_channel(
    reference: str,
    alternate: str,
    trinucleotide: str,
) -> Optional[str]:


    reference = reference.upper()
    alternate = alternate.upper()
    trinucleotide = trinucleotide.upper()

    if len(trinucleotide) != 3:
        return None

    five_prime = trinucleotide[0]
    center = trinucleotide[1]
    three_prime = trinucleotide[2]

    if (
        reference not in BASES
        or alternate not in BASES
        or reference == alternate
    ):
        return None

    if center != reference:
        return None

    if reference not in PYRIMIDINES:
        reference = reverse_complement_base(reference)
        alternate = reverse_complement_base(alternate)

        five_prime, three_prime = (
            reverse_complement_base(three_prime),
            reverse_complement_base(five_prime),
        )

    substitution = f"{reference}>{alternate}"

    if substitution not in SUBSTITUTION_TYPES:
        return None

    channel = (
        f"{five_prime}[{substitution}]{three_prime}"
    )

    if channel not in CHANNEL_TO_INDEX:
        return None

    return channel


def is_snp_variant_type(value: str) -> bool:
    variant_type = normalize_text(value).upper()

    return variant_type in {
        "SNP",
        "SNV",
        "SINGLE_NUCLEOTIDE_VARIANT",
    }


def process_patient_mutations(
    patient_df: pd.DataFrame,
    genome: Fasta,
) -> Tuple[Set[str], Counter]:


    channels_present = set()

    qc = Counter(
        {
            "total_rows": 0,
            "non_snp_rows": 0,
            "missing_required_values": 0,
            "invalid_position": 0,
            "invalid_alleles": 0,
            "chromosome_not_found": 0,
            "reference_mismatch": 0,
            "usable_snvs": 0,
        }
    )

    for _, row in patient_df.iterrows():
        qc["total_rows"] += 1

        if not is_snp_variant_type(
            row.get("Variant_Type", "")
        ):
            qc["non_snp_rows"] += 1
            continue

        chromosome = normalize_chromosome(
            row.get("Chromosome", "")
        )

        reference = normalize_text(
            row.get("Reference_Allele", "")
        ).upper()

        alternate = normalize_text(
            row.get("Tumor_Seq_Allele2", "")
        ).upper()

        position_value = row.get("Start_Position")

        if (
            not chromosome
            or not reference
            or not alternate
            or pd.isna(position_value)
        ):
            qc["missing_required_values"] += 1
            continue

        try:
            position = int(float(position_value))
        except (TypeError, ValueError):
            qc["invalid_position"] += 1
            continue

        if (
            len(reference) != 1
            or len(alternate) != 1
            or reference not in BASES
            or alternate not in BASES
            or reference == alternate
        ):
            qc["invalid_alleles"] += 1
            continue

        trinucleotide = fetch_trinucleotide(
            genome,
            chromosome,
            position,
        )

        if trinucleotide is None:
            qc["chromosome_not_found"] += 1
            continue

        if trinucleotide[1] != reference:
            qc["reference_mismatch"] += 1
            continue

        channel = get_sbs96_channel(
            reference,
            alternate,
            trinucleotide,
        )

        if channel is None:
            qc["invalid_alleles"] += 1
            continue

        channels_present.add(channel)
        qc["usable_snvs"] += 1

    return channels_present, qc


def save_outputs(
    output_dir: Path,
    patient_channels: Dict[str, Set[str]],
    case_to_tumor_origin: Dict[str, str],
    case_qc: Dict[str, Counter],
    file_failures: List[Tuple[str, str]],
    manifest_count: int,
    maf_file_count: int,
) -> None:
    case_ids = sorted(patient_channels)

    row_indices = []
    column_indices = []
    values = []

    for row_index, case_id in enumerate(case_ids):
        for channel in sorted(patient_channels[case_id]):
            row_indices.append(row_index)
            column_indices.append(
                CHANNEL_TO_INDEX[channel]
            )
            values.append(1)

    matrix = coo_matrix(
        (
            np.asarray(values, dtype=np.int8),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(case_ids), len(CHANNEL_NAMES)),
        dtype=np.int8,
    ).tocsr()

    matrix_path = (
        output_dir / "patient_signature_matrix.npz"
    )

    rows_path = (
        output_dir
        / "patient_signature_matrix_rows.txt"
    )

    columns_path = (
        output_dir
        / "patient_signature_matrix_columns.txt"
    )

    qc_path = (
        output_dir / "patient_signature_qc.csv"
    )

    summary_path = (
        output_dir
        / "mutational_signatures_summary.txt"
    )

    save_npz(matrix_path, matrix)

    with rows_path.open("w") as handle:
        for case_id in case_ids:
            tumor_origin = case_to_tumor_origin.get(
                case_id,
                "",
            )

            handle.write(
                f"{tumor_origin}\t{case_id}\n"
            )

    with columns_path.open("w") as handle:
        for channel in CHANNEL_NAMES:
            handle.write(f"{channel}\n")

    qc_rows = []

    for case_id in case_ids:
        qc = case_qc.get(case_id, Counter())

        qc_rows.append(
            {
                "case_id": case_id,
                "tumor_origin": (
                    case_to_tumor_origin.get(
                        case_id,
                        "",
                    )
                ),
                "channels_present": len(
                    patient_channels[case_id]
                ),
                "total_rows": qc["total_rows"],
                "non_snp_rows": qc["non_snp_rows"],
                "missing_required_values": (
                    qc["missing_required_values"]
                ),
                "invalid_position": (
                    qc["invalid_position"]
                ),
                "invalid_alleles": (
                    qc["invalid_alleles"]
                ),
                "chromosome_not_found": (
                    qc["chromosome_not_found"]
                ),
                "reference_mismatch": (
                    qc["reference_mismatch"]
                ),
                "usable_snvs": qc["usable_snvs"],
            }
        )

    pd.DataFrame(qc_rows).to_csv(
        qc_path,
        index=False,
    )

    total_qc = Counter()

    for qc in case_qc.values():
        total_qc.update(qc)

    zero_channel_cases = sum(
        1
        for channels in patient_channels.values()
        if not channels
    )

    with summary_path.open("w") as handle:
        handle.write(
            "CPTAC SBS96 EXTRACTION SUMMARY\n"
        )
        handle.write("=" * 60 + "\n\n")

        handle.write(
            f"Manifest records: {manifest_count}\n"
        )
        handle.write(
            f"Matching MAF files found: {maf_file_count}\n"
        )
        handle.write(
            f"Unique patient identifiers encountered: "
            f"{len(patient_channels)}\n"
        )
        handle.write(
            f"Patients with at least one SBS96 channel: "
            f"{len(patient_channels) - zero_channel_cases}\n"
        )
        handle.write(
            f"Patients with zero usable SBS96 channels: "
            f"{zero_channel_cases}\n"
        )
        handle.write(
            f"Files that failed: {len(file_failures)}\n"
        )
        handle.write("\n")

        handle.write(
            f"Matrix shape: "
            f"{matrix.shape[0]} x {matrix.shape[1]}\n"
        )
        handle.write(
            f"Matrix nonzero entries: {matrix.nnz}\n"
        )

        if matrix.shape[0] > 0:
            density = (
                matrix.nnz
                / (
                    matrix.shape[0]
                    * matrix.shape[1]
                )
            )
        else:
            density = 0.0

        handle.write(
            f"Matrix density: {density:.8f}\n"
        )
        handle.write("\n")

        handle.write("Mutation-row QC totals\n")
        handle.write("-" * 60 + "\n")

        for key in [
            "total_rows",
            "non_snp_rows",
            "missing_required_values",
            "invalid_position",
            "invalid_alleles",
            "chromosome_not_found",
            "reference_mismatch",
            "usable_snvs",
        ]:
            handle.write(
                f"{key}: {total_qc[key]}\n"
            )

        if file_failures:
            handle.write("\nFailed files\n")
            handle.write("-" * 60 + "\n")

            for file_id, reason in file_failures:
                handle.write(
                    f"{file_id}\t{reason}\n"
                )

    log.info(
        "Saved sparse matrix: %s",
        matrix_path,
    )

    log.info(
        "Matrix shape: %s patients x %s channels",
        matrix.shape[0],
        matrix.shape[1],
    )

    log.info(
        "Matrix nonzero entries: %s",
        matrix.nnz,
    )

    log.info(
        "Saved row labels: %s",
        rows_path,
    )

    log.info(
        "Saved SBS96 columns: %s",
        columns_path,
    )

    log.info(
        "Saved patient QC: %s",
        qc_path,
    )

    log.info(
        "Saved summary: %s",
        summary_path,
    )


def main() -> None:
    args = parse_arguments()

    maf_dir = Path(args.maf_dir)
    manifest_path = Path(args.manifest)
    reference_fasta = Path(args.reference_fasta)
    output_dir = Path(args.output_dir)

    cohort_map_path = (
        Path(args.cohort_map_csv)
        if args.cohort_map_csv
        else None
    )

    for required_path, description in [
        (maf_dir, "MAF directory"),
        (manifest_path, "manifest"),
        (reference_fasta, "reference FASTA"),
    ]:
        if not required_path.exists():
            log.error(
                "%s does not exist: %s",
                description,
                required_path,
            )
            sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log.info(
        "Reading CPTAC manifest: %s",
        manifest_path,
    )

    manifest = read_manifest(manifest_path)

    log.info(
        "Manifest contains %s records",
        len(manifest),
    )

    maf_files = find_maf_files(
        maf_dir=maf_dir,
        manifest=manifest,
        keep_duplicate_mafs=args.keep_duplicate_mafs,
    )

    log.info(
        "Found %s matching MAF files under %s",
        len(maf_files),
        maf_dir,
    )

    if not maf_files:
        log.error(
            "No matching .maf.gz files were found. "
            "Check --maf-dir and --manifest."
        )
        sys.exit(1)

    case_to_tumor_origin = load_cohort_mapping(
        cohort_map_path
    )

    if case_to_tumor_origin:
        log.info(
            "Loaded cohort labels for %s cases",
            len(case_to_tumor_origin),
        )
    else:
        log.info(
            "No cohort mapping supplied; tumor_origin "
            "will be blank in the rows file."
        )

    log.info(
        "Loading reference genome: %s",
        reference_fasta,
    )

    genome = Fasta(
        str(reference_fasta),
        as_raw=False,
        sequence_always_upper=True,
    )

    log.info(
        "Reference contains %s sequences",
        len(genome.keys()),
    )

    patient_channels: Dict[str, Set[str]] = defaultdict(set)
    case_qc: Dict[str, Counter] = defaultdict(Counter)
    file_failures: List[Tuple[str, str]] = []

    for maf_path in tqdm(
        maf_files,
        desc="Computing CPTAC SBS96 contexts",
        unit="file",
    ):
        file_id = maf_path.parent.name

        try:
            maf_df = read_maf(maf_path)

            if maf_df.empty:
                log.warning(
                    "Empty MAF file: %s",
                    maf_path,
                )
                continue

            for detected_case_id, patient_df in get_patient_groups(
                maf_df,
                maf_path,
            ):
                case_id = normalize_case_id(
                    detected_case_id
                )

                if not case_id:
                    case_id = file_id

                channels, qc = process_patient_mutations(
                    patient_df,
                    genome,
                )

                patient_channels[case_id].update(
                    channels
                )

                case_qc[case_id].update(qc)

        except Exception as error:
            log.warning(
                "Failed on %s: %s",
                maf_path,
                error,
            )

            file_failures.append(
                (
                    file_id,
                    str(error),
                )
            )

    genome.close()

    patients_with_channels = {
        case_id: channels
        for case_id, channels in patient_channels.items()
        if channels
    }

    if not patients_with_channels:
        log.error(
            "No patient produced usable SBS96 features."
        )
        log.error(
            "The most likely causes are an incompatible "
            "reference FASTA build or chromosome naming mismatch."
        )
        sys.exit(1)

    save_outputs(
        output_dir=output_dir,
        patient_channels=patients_with_channels,
        case_to_tumor_origin=case_to_tumor_origin,
        case_qc=case_qc,
        file_failures=file_failures,
        manifest_count=len(manifest),
        maf_file_count=len(maf_files),
    )

    log.info("CPTAC SBS96 extraction finished.")


if __name__ == "__main__":
    main()