import argparse
import gzip
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("extract_muts-CPTAC")


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


MUTATION_PREFIX = "som_mut_"

MAF_COLUMNS = [
    "Hugo_Symbol",
    "Entrez_Gene_Id",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Variant_Classification",
    "Variant_Type",
    "Tumor_Sample_Barcode",
    "Matched_Norm_Sample_Barcode",
    "Tumor_Sample_UUID",
    "Matched_Norm_Sample_UUID",
    "HGVSc",
    "HGVSp",
    "HGVSp_Short",
    "Transcript_ID",
    "t_depth",
    "t_ref_count",
    "t_alt_count",
    "n_depth",
    "n_ref_count",
    "n_alt_count",
    "case_id",
    "GDC_FILTER",
    "hotspot",
    "callers",
]

REQUIRED_MAF_COLUMNS = [
    "Hugo_Symbol",
    "Variant_Classification",
]

NON_SILENT_CLASSES = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Nonstop_Mutation",
    "Translation_Start_Site",
    "Start_Codon_Del",
    "Start_Codon_Ins",
    "Start_Codon_SNP",
    "Stop_Codon_Del",
    "Stop_Codon_Ins",
    "De_novo_Start_InFrame",
}

SILENT_OR_NONCODING_CLASSES = {
    "Silent",
    "Intron",
    "IGR",
    "RNA",
    "3'UTR",
    "5'UTR",
    "3'Flank",
    "5'Flank",
    "Splice_Region",
    "Targeted_Region",
    "lincRNA",
}

UNKNOWN_VALUES = {
    "",
    "unknown",
    "not reported",
    "not available",
    "not applicable",
    "nan",
    "none",
}


PROJECT_TO_ORGAN = {
    "CPTAC-BRCA": "Breast",
    "CPTAC-CCRCC": "Kidney",
    "CPTAC-COAD": "Colon",
    "CPTAC-GBM": "Brain",
    "CPTAC-HNSC": "Head and Neck",
    "CPTAC-LSCC": "Lung",
    "CPTAC-LUAD": "Lung",
    "CPTAC-OV": "Ovary",
    "CPTAC-PDA": "Pancreas",
    "CPTAC-PDAC": "Pancreas",
    "CPTAC-UCEC": "Uterus",
}

ORGAN_KEYWORDS = [
    (
        "Lung",
        [
            "lung",
            "bronch",
            "pulmonary",
            "adenocarcinoma of lung",
            "lung adenocarcinoma",
            "lung squamous",
        ],
    ),
    (
        "Breast",
        [
            "breast",
            "mammary",
        ],
    ),
    (
        "Kidney",
        [
            "kidney",
            "renal",
            "clear cell renal",
            "ccrcc",
        ],
    ),
    (
        "Colon",
        [
            "colon",
            "colorectal",
            "large intestine",
            "cecum",
            "sigmoid",
            "ascending colon",
            "descending colon",
            "transverse colon",
        ],
    ),
    (
        "Brain",
        [
            "brain",
            "glioblastoma",
            "glioma",
            "central nervous system",
            "cerebr",
        ],
    ),
    (
        "Head and Neck",
        [
            "head and neck",
            "oral cavity",
            "oropharynx",
            "hypopharynx",
            "larynx",
            "laryngeal",
            "tongue",
            "tonsil",
            "pharynx",
            "mouth",
        ],
    ),
    (
        "Ovary",
        [
            "ovary",
            "ovarian",
            "fallopian tube",
            "fallopian",
        ],
    ),
    (
        "Pancreas",
        [
            "pancreas",
            "pancreatic",
            "pancreatic ductal",
        ],
    ),
    (
        "Uterus",
        [
            "uterus",
            "uterine",
            "endometrium",
            "endometrial",
            "corpus uteri",
        ],
    ),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CPTAC patient-level non-silent mutation features "
            "and organ-level tumor-origin labels."
        )
    )

    parser.add_argument(
        "--maf-dir",
        required=True,
        help=(
            "Root directory containing GDC-downloaded .maf.gz files. "
            "Files are normally under <file UUID>/<filename>.maf.gz."
        ),
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="GDC manifest used to download the CPTAC MAF files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which mutation outputs will be written.",
    )

    parser.add_argument(
        "--include-all-classes",
        action="store_true",
        help=(
            "Include all mutation classifications rather than only "
            "the predefined non-silent classes."
        ),
    )

    parser.add_argument(
        "--metadata-batch-size",
        type=int,
        default=300,
        help="Number of manifest file UUIDs queried per GDC API request.",
    )

    parser.add_argument(
        "--metadata-timeout",
        type=int,
        default=120,
        help="GDC API request timeout in seconds.",
    )

    parser.add_argument(
        "--metadata-retries",
        type=int,
        default=4,
        help="Number of retries for failed GDC metadata requests.",
    )

    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help=(
            "Ignore an existing gdc_metadata_cache.csv and query "
            "GDC metadata again."
        ),
    )

    return parser.parse_args()


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in UNKNOWN_VALUES:
        return ""

    return text


def sanitize_feature_component(value: str) -> str:
    text = clean_text(value)

    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def normalize_case_id(value) -> str:
    return clean_text(value)


def calculate_vaf(
    alt_count,
    depth,
) -> Optional[float]:
    try:
        alt = float(alt_count)
        total = float(depth)

        if total <= 0:
            return None

        return alt / total

    except (TypeError, ValueError):
        return None


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype=str,
    )

    required = {"id", "filename"}
    missing = required.difference(manifest.columns)

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )

    manifest["id"] = (
        manifest["id"]
        .astype(str)
        .str.strip()
    )

    manifest["filename"] = (
        manifest["filename"]
        .astype(str)
        .str.strip()
    )

    return manifest


def find_maf_files(
    maf_dir: Path,
    manifest: pd.DataFrame,
) -> List[Path]:
    wanted_file_ids = set(
        manifest["id"].astype(str)
    )

    maf_files = [
        path
        for path in maf_dir.rglob("*.maf.gz")
        if path.parent.name in wanted_file_ids
    ]

    return sorted(maf_files)


def detect_header_line(maf_path: Path) -> int:
    with gzip.open(
        maf_path,
        "rt",
        errors="replace",
    ) as handle:
        for line_number, line in enumerate(handle):
            if not line.startswith("#"):
                return line_number

    raise ValueError(
        "No non-comment MAF header line found"
    )


def read_maf(maf_path: Path) -> pd.DataFrame:
    header_line = detect_header_line(maf_path)

    header = pd.read_csv(
        maf_path,
        sep="\t",
        compression="gzip",
        skiprows=header_line,
        nrows=0,
    )

    missing_required = [
        column
        for column in REQUIRED_MAF_COLUMNS
        if column not in header.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required MAF columns: "
            + ", ".join(missing_required)
        )

    available_columns = [
        column
        for column in MAF_COLUMNS
        if column in header.columns
    ]

    return pd.read_csv(
        maf_path,
        sep="\t",
        compression="gzip",
        skiprows=header_line,
        usecols=available_columns,
        low_memory=False,
    )


def post_gdc_metadata_request(
    file_ids: List[str],
    timeout: int,
    retries: int,
) -> List[dict]:
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "files.file_id",
                "value": file_ids,
            },
        },
        "format": "JSON",
        "size": len(file_ids),
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "cases.case_id",
                "cases.submitter_id",
                "cases.project.project_id",
                "cases.project.name",
                "cases.primary_site",
                "cases.disease_type",
                "cases.samples.sample_id",
                "cases.samples.submitter_id",
                "cases.samples.sample_type",
                "cases.samples.tissue_type",
            ]
        ),
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                GDC_FILES_ENDPOINT,
                json=payload,
                timeout=timeout,
            )

            response.raise_for_status()

            body = response.json()

            return body.get(
                "data",
                {},
            ).get(
                "hits",
                [],
            )

        except (
            requests.RequestException,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            if attempt < retries:
                wait_seconds = min(
                    2 ** attempt,
                    30,
                )

                log.warning(
                    "GDC metadata request failed "
                    "(attempt %s/%s): %s. "
                    "Retrying in %s seconds.",
                    attempt,
                    retries,
                    error,
                    wait_seconds,
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"GDC metadata request failed: {last_error}"
    )


def query_gdc_metadata(
    manifest: pd.DataFrame,
    batch_size: int,
    timeout: int,
    retries: int,
) -> pd.DataFrame:
    file_ids = (
        manifest["id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    records = []

    for start in tqdm(
        range(0, len(file_ids), batch_size),
        desc="Querying GDC metadata",
        unit="batch",
    ):
        batch = file_ids[
            start:start + batch_size
        ]

        hits = post_gdc_metadata_request(
            file_ids=batch,
            timeout=timeout,
            retries=retries,
        )

        for hit in hits:
            file_id = clean_text(
                hit.get("file_id")
                or hit.get("id")
            )

            file_name = clean_text(
                hit.get("file_name")
            )

            cases = hit.get("cases", []) or []

            if not cases:
                records.append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "case_id": "",
                        "case_submitter_id": "",
                        "project_id": "",
                        "project_name": "",
                        "primary_site": "",
                        "disease_type": "",
                    }
                )

                continue

            for case in cases:
                project = (
                    case.get("project", {})
                    or {}
                )

                records.append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "case_id": clean_text(
                            case.get("case_id")
                        ),
                        "case_submitter_id": clean_text(
                            case.get("submitter_id")
                        ),
                        "project_id": clean_text(
                            project.get("project_id")
                        ),
                        "project_name": clean_text(
                            project.get("name")
                        ),
                        "primary_site": clean_text(
                            case.get("primary_site")
                        ),
                        "disease_type": clean_text(
                            case.get("disease_type")
                        ),
                    }
                )

    metadata = pd.DataFrame(records)

    if metadata.empty:
        raise RuntimeError(
            "The GDC API returned no metadata records."
        )

    metadata = metadata.drop_duplicates()

    return metadata


def map_to_organ(
    project_id: str,
    project_name: str,
    disease_type: str,
    primary_site: str,
) -> str:
    project_id = clean_text(project_id)

    if project_id in PROJECT_TO_ORGAN:
        return PROJECT_TO_ORGAN[project_id]

    combined = " | ".join(
        [
            clean_text(project_id),
            clean_text(project_name),
            clean_text(disease_type),
            clean_text(primary_site),
        ]
    ).lower()

    for organ, keywords in ORGAN_KEYWORDS:
        for keyword in keywords:
            if keyword.lower() in combined:
                return organ

    return "Unknown"


def build_metadata_maps(
    metadata: pd.DataFrame,
) -> Tuple[
    Dict[str, str],
    Dict[str, dict],
    Dict[str, dict],
]:
    file_id_to_case_id = {}
    case_metadata = {}
    submitter_to_case_metadata = {}

    for _, row in metadata.iterrows():
        file_id = clean_text(
            row.get("file_id")
        )

        case_id = clean_text(
            row.get("case_id")
        )

        case_submitter_id = clean_text(
            row.get("case_submitter_id")
        )

        project_id = clean_text(
            row.get("project_id")
        )

        project_name = clean_text(
            row.get("project_name")
        )

        primary_site = clean_text(
            row.get("primary_site")
        )

        disease_type = clean_text(
            row.get("disease_type")
        )

        tumor_origin = map_to_organ(
            project_id=project_id,
            project_name=project_name,
            disease_type=disease_type,
            primary_site=primary_site,
        )

        record = {
            "case_id": case_id,
            "case_submitter_id": case_submitter_id,
            "tumor_origin": tumor_origin,
            "project_id": project_id,
            "project_name": project_name,
            "primary_site": primary_site,
            "disease_type": disease_type,
        }

        if file_id and case_id:
            file_id_to_case_id[file_id] = case_id

        if case_id:
            case_metadata[case_id] = record

        if case_submitter_id:
            submitter_to_case_metadata[
                case_submitter_id
            ] = record

    return (
        file_id_to_case_id,
        case_metadata,
        submitter_to_case_metadata,
    )


def choose_case_id(
    maf_df: pd.DataFrame,
    file_id: str,
    file_id_to_case_id: Dict[str, str],
) -> str:
    if "case_id" in maf_df.columns:
        case_values = [
            normalize_case_id(value)
            for value in maf_df[
                "case_id"
            ].dropna().unique()
        ]

        case_values = [
            value
            for value in case_values
            if value
        ]

        if case_values:
            return case_values[0]

    mapped_case_id = file_id_to_case_id.get(
        file_id,
        "",
    )

    if mapped_case_id:
        return mapped_case_id

    if "Tumor_Sample_UUID" in maf_df.columns:
        values = [
            clean_text(value)
            for value in maf_df[
                "Tumor_Sample_UUID"
            ].dropna().unique()
        ]

        values = [
            value
            for value in values
            if value
        ]

        if values:
            return values[0]

    return file_id


def filter_mutations(
    maf_df: pd.DataFrame,
    include_all_classes: bool,
) -> pd.DataFrame:
    working = maf_df.copy()

    working["Hugo_Symbol"] = (
        working["Hugo_Symbol"]
        .map(clean_text)
    )

    working["Variant_Classification"] = (
        working["Variant_Classification"]
        .map(clean_text)
    )

    working = working[
        working["Hugo_Symbol"].ne("")
    ]

    working = working[
        ~working["Hugo_Symbol"].isin(
            {"Unknown", "UNKNOWN", "."}
        )
    ]

    if not include_all_classes:
        working = working[
            working[
                "Variant_Classification"
            ].isin(NON_SILENT_CLASSES)
        ]

    return working


def mutation_features_for_patient(
    patient_df: pd.DataFrame,
) -> Set[str]:


    features = set()

    for _, row in patient_df.iterrows():
        gene = sanitize_feature_component(
            row.get("Hugo_Symbol", "")
        )

        classification = (
            sanitize_feature_component(
                row.get(
                    "Variant_Classification",
                    "",
                )
            )
        )

        if not gene:
            continue


        features.add(f"{MUTATION_PREFIX}{gene}_any")


        if classification:
            features.add(
                f"{MUTATION_PREFIX}{gene}_{classification}"
            )

    return features


def prepare_long_mutation_table(
    filtered_df: pd.DataFrame,
    case_id: str,
    metadata_record: dict,
    file_id: str,
    maf_path: Path,
) -> pd.DataFrame:
    output = filtered_df.copy()

    output.insert(
        0,
        "source_file_id",
        file_id,
    )

    output.insert(
        1,
        "source_filename",
        maf_path.name,
    )

    if "case_id" in output.columns:
        output["case_id"] = case_id
    else:
        output.insert(
            2,
            "case_id",
            case_id,
        )

    output["tumor_origin"] = (
        metadata_record.get(
            "tumor_origin",
            "Unknown",
        )
    )

    output["project_id"] = (
        metadata_record.get(
            "project_id",
            "",
        )
    )

    output["primary_site"] = (
        metadata_record.get(
            "primary_site",
            "",
        )
    )

    output["disease_type"] = (
        metadata_record.get(
            "disease_type",
            "",
        )
    )

    if (
        "t_alt_count" in output.columns
        and "t_depth" in output.columns
    ):
        output["VAF"] = [
            calculate_vaf(alt, depth)
            for alt, depth in zip(
                output["t_alt_count"],
                output["t_depth"],
            )
        ]

    return output


def save_sparse_matrix(
    patient_features: Dict[str, Set[str]],
    case_metadata: Dict[str, dict],
    output_dir: Path,
) -> Tuple[int, int, int]:
    case_ids = sorted(patient_features)

    all_features = sorted(
        {
            feature
            for features in patient_features.values()
            for feature in features
        }
    )

    feature_to_index = {
        feature: index
        for index, feature in enumerate(
            all_features
        )
    }

    row_indices = []
    column_indices = []

    for row_index, case_id in enumerate(case_ids):
        for feature in patient_features[case_id]:
            row_indices.append(row_index)
            column_indices.append(
                feature_to_index[feature]
            )

    values = np.ones(
        len(row_indices),
        dtype=np.int8,
    )

    matrix = coo_matrix(
        (
            values,
            (
                np.asarray(
                    row_indices,
                    dtype=np.int64,
                ),
                np.asarray(
                    column_indices,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(case_ids),
            len(all_features),
        ),
        dtype=np.int8,
    ).tocsr()

    matrix_path = (
        output_dir
        / "patient_mutation_matrix.npz"
    )

    rows_path = (
        output_dir
        / "patient_mutation_matrix_rows.txt"
    )

    columns_path = (
        output_dir
        / "patient_mutation_matrix_columns.txt"
    )

    save_npz(
        matrix_path,
        matrix,
    )

    with rows_path.open("w") as handle:
        for case_id in case_ids:
            tumor_origin = (
                case_metadata
                .get(case_id, {})
                .get(
                    "tumor_origin",
                    "Unknown",
                )
            )

            handle.write(
                f"{tumor_origin}\t{case_id}\n"
            )

    with columns_path.open("w") as handle:
        for feature in all_features:
            handle.write(
                f"{feature}\n"
            )

    log.info(
        "Saved mutation matrix: %s",
        matrix_path,
    )

    log.info(
        "Mutation matrix shape: %s patients x %s features",
        matrix.shape[0],
        matrix.shape[1],
    )

    log.info(
        "Matrix nonzero entries: %s",
        matrix.nnz,
    )

    return (
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
    )


def save_long_mutations(
    long_tables: List[pd.DataFrame],
    output_dir: Path,
) -> Tuple[str, int]:
    if not long_tables:
        return "", 0

    mutations_long = pd.concat(
        long_tables,
        ignore_index=True,
        sort=False,
    )

    parquet_path = (
        output_dir
        / "mutations_long.parquet"
    )

    try:
        mutations_long.to_parquet(
            parquet_path,
            index=False,
        )

        log.info(
            "Saved mutation audit table: %s",
            parquet_path,
        )

        return str(parquet_path), len(
            mutations_long
        )

    except Exception as error:
        csv_path = (
            output_dir
            / "mutations_long.csv"
        )

        log.warning(
            "Could not write Parquet (%s). "
            "Saving CSV instead.",
            error,
        )

        mutations_long.to_csv(
            csv_path,
            index=False,
        )

        log.info(
            "Saved mutation audit table: %s",
            csv_path,
        )

        return str(csv_path), len(
            mutations_long
        )


def save_patient_metadata(
    patient_features: Dict[str, Set[str]],
    case_metadata: Dict[str, dict],
    file_counts: Counter,
    mutation_counts: Counter,
    output_dir: Path,
) -> pd.DataFrame:
    records = []

    for case_id in sorted(patient_features):
        metadata = case_metadata.get(
            case_id,
            {},
        )

        records.append(
            {
                "case_id": case_id,
                "case_submitter_id": metadata.get(
                    "case_submitter_id",
                    "",
                ),
                "tumor_origin": metadata.get(
                    "tumor_origin",
                    "Unknown",
                ),
                "project_id": metadata.get(
                    "project_id",
                    "",
                ),
                "project_name": metadata.get(
                    "project_name",
                    "",
                ),
                "primary_site": metadata.get(
                    "primary_site",
                    "",
                ),
                "disease_type": metadata.get(
                    "disease_type",
                    "",
                ),
                "maf_file_count": file_counts[
                    case_id
                ],
                "qualifying_mutation_count": (
                    mutation_counts[case_id]
                ),
                "mutation_feature_count": len(
                    patient_features[case_id]
                ),
            }
        )

    metadata_df = pd.DataFrame(records)

    metadata_path = (
        output_dir
        / "patient_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    log.info(
        "Saved patient metadata: %s",
        metadata_path,
    )

    return metadata_df


def save_qc(
    qc_records: List[dict],
    output_dir: Path,
) -> None:
    qc_path = (
        output_dir
        / "patient_mutation_qc.csv"
    )

    pd.DataFrame(qc_records).to_csv(
        qc_path,
        index=False,
    )

    log.info(
        "Saved mutation QC: %s",
        qc_path,
    )


def save_summary(
    output_dir: Path,
    manifest_count: int,
    maf_count: int,
    empty_mafs: int,
    silent_only_mafs: int,
    failed_files: List[Tuple[str, str]],
    patient_metadata: pd.DataFrame,
    matrix_rows: int,
    matrix_columns: int,
    matrix_nonzeros: int,
    long_table_path: str,
    long_mutation_count: int,
) -> None:
    summary_path = (
        output_dir
        / "extraction_summary.txt"
    )

    organ_counts = Counter(
        patient_metadata[
            "tumor_origin"
        ].fillna("Unknown")
    )

    project_counts = Counter(
        patient_metadata[
            "project_id"
        ].fillna("")
    )

    with summary_path.open("w") as handle:
        handle.write(
            "CPTAC MUTATION EXTRACTION SUMMARY\n"
        )
        handle.write("=" * 70 + "\n\n")

        handle.write(
            f"Manifest records: {manifest_count}\n"
        )
        handle.write(
            f"Matching MAF files found: {maf_count}\n"
        )
        handle.write(
            f"Header-only/empty MAF files: {empty_mafs}\n"
        )
        handle.write(
            "MAFs with mutation rows but no qualifying "
            f"non-silent mutations: {silent_only_mafs}\n"
        )
        handle.write(
            f"Failed MAF files: {len(failed_files)}\n"
        )
        handle.write("\n")

        handle.write(
            f"Patients in mutation matrix: {matrix_rows}\n"
        )
        handle.write(
            f"Mutation features: {matrix_columns}\n"
        )
        handle.write(
            f"Matrix nonzero entries: {matrix_nonzeros}\n"
        )

        if matrix_rows and matrix_columns:
            density = (
                matrix_nonzeros
                / (
                    matrix_rows
                    * matrix_columns
                )
            )
        else:
            density = 0.0

        handle.write(
            f"Matrix density: {density:.10f}\n"
        )
        handle.write(
            f"Qualifying mutation records: "
            f"{long_mutation_count}\n"
        )
        handle.write(
            f"Mutation audit table: {long_table_path}\n"
        )

        handle.write("\nORGAN-LEVEL TUMOR ORIGINS\n")
        handle.write("-" * 70 + "\n")

        for organ, count in sorted(
            organ_counts.items()
        ):
            handle.write(
                f"{organ}: {count}\n"
            )

        handle.write("\nGDC PROJECTS\n")
        handle.write("-" * 70 + "\n")

        for project, count in sorted(
            project_counts.items()
        ):
            label = project or "Unknown"
            handle.write(
                f"{label}: {count}\n"
            )

        if failed_files:
            handle.write("\nFAILED FILES\n")
            handle.write("-" * 70 + "\n")

            for file_id, reason in failed_files:
                handle.write(
                    f"{file_id}\t{reason}\n"
                )

    log.info(
        "Saved extraction summary: %s",
        summary_path,
    )


def main() -> None:
    args = parse_arguments()

    maf_dir = Path(args.maf_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if not maf_dir.exists():
        log.error(
            "MAF directory does not exist: %s",
            maf_dir,
        )
        sys.exit(1)

    if not manifest_path.exists():
        log.error(
            "Manifest does not exist: %s",
            manifest_path,
        )
        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log.info(
        "Reading manifest: %s",
        manifest_path,
    )

    manifest = read_manifest(
        manifest_path
    )

    log.info(
        "Manifest contains %s records",
        len(manifest),
    )

    maf_files = find_maf_files(
        maf_dir=maf_dir,
        manifest=manifest,
    )

    log.info(
        "Found %s matching MAF files",
        len(maf_files),
    )

    if not maf_files:
        log.error(
            "No matching MAF files were found."
        )
        sys.exit(1)

    metadata_cache_path = (
        output_dir
        / "gdc_metadata_cache.csv"
    )

    if (
        metadata_cache_path.exists()
        and not args.refresh_metadata
    ):
        log.info(
            "Loading cached GDC metadata: %s",
            metadata_cache_path,
        )

        gdc_metadata = pd.read_csv(
            metadata_cache_path,
            dtype=str,
        ).fillna("")

    else:
        log.info(
            "Querying GDC metadata for CPTAC files..."
        )

        gdc_metadata = query_gdc_metadata(
            manifest=manifest,
            batch_size=args.metadata_batch_size,
            timeout=args.metadata_timeout,
            retries=args.metadata_retries,
        )

        gdc_metadata.to_csv(
            metadata_cache_path,
            index=False,
        )

        log.info(
            "Saved GDC metadata cache: %s",
            metadata_cache_path,
        )

    (
        file_id_to_case_id,
        case_metadata,
        submitter_to_case_metadata,
    ) = build_metadata_maps(
        gdc_metadata
    )

    log.info(
        "Metadata contains %s unique case IDs",
        len(case_metadata),
    )

    patient_features: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    file_counts = Counter()
    mutation_counts = Counter()

    long_tables = []
    qc_records = []
    failed_files = []

    empty_mafs = 0
    silent_only_mafs = 0

    for maf_path in tqdm(
        maf_files,
        desc="Extracting CPTAC mutations",
        unit="file",
    ):
        file_id = maf_path.parent.name

        try:
            maf_df = read_maf(maf_path)

            if maf_df.empty:
                empty_mafs += 1

                qc_records.append(
                    {
                        "file_id": file_id,
                        "filename": maf_path.name,
                        "case_id": (
                            file_id_to_case_id.get(
                                file_id,
                                "",
                            )
                        ),
                        "status": "empty_maf",
                        "total_rows": 0,
                        "qualifying_rows": 0,
                    }
                )

                continue

            case_id = choose_case_id(
                maf_df=maf_df,
                file_id=file_id,
                file_id_to_case_id=(
                    file_id_to_case_id
                ),
            )

            metadata_record = case_metadata.get(
                case_id,
                {},
            )


            if (
                not metadata_record
                and "Tumor_Sample_Barcode"
                in maf_df.columns
            ):
                barcodes = [
                    clean_text(value)
                    for value in maf_df[
                        "Tumor_Sample_Barcode"
                    ].dropna().unique()
                ]

                for barcode in barcodes:
                    if barcode in submitter_to_case_metadata:
                        metadata_record = (
                            submitter_to_case_metadata[
                                barcode
                            ]
                        )
                        break

            if not metadata_record:
                metadata_record = {
                    "case_id": case_id,
                    "case_submitter_id": "",
                    "tumor_origin": "Unknown",
                    "project_id": "",
                    "project_name": "",
                    "primary_site": "",
                    "disease_type": "",
                }

                case_metadata[
                    case_id
                ] = metadata_record

            filtered_df = filter_mutations(
                maf_df=maf_df,
                include_all_classes=(
                    args.include_all_classes
                ),
            )

            file_counts[case_id] += 1

            if filtered_df.empty:
                silent_only_mafs += 1

                qc_records.append(
                    {
                        "file_id": file_id,
                        "filename": maf_path.name,
                        "case_id": case_id,
                        "tumor_origin": (
                            metadata_record.get(
                                "tumor_origin",
                                "Unknown",
                            )
                        ),
                        "status": (
                            "no_qualifying_mutations"
                        ),
                        "total_rows": len(maf_df),
                        "qualifying_rows": 0,
                    }
                )

                continue

            features = mutation_features_for_patient(
                filtered_df
            )

            patient_features[case_id].update(
                features
            )

            mutation_counts[case_id] += len(
                filtered_df
            )

            long_table = prepare_long_mutation_table(
                filtered_df=filtered_df,
                case_id=case_id,
                metadata_record=metadata_record,
                file_id=file_id,
                maf_path=maf_path,
            )

            long_tables.append(
                long_table
            )

            qc_records.append(
                {
                    "file_id": file_id,
                    "filename": maf_path.name,
                    "case_id": case_id,
                    "tumor_origin": (
                        metadata_record.get(
                            "tumor_origin",
                            "Unknown",
                        )
                    ),
                    "status": "processed",
                    "total_rows": len(maf_df),
                    "qualifying_rows": len(
                        filtered_df
                    ),
                    "features_from_file": len(
                        features
                    ),
                }
            )

        except Exception as error:
            log.warning(
                "Failed on %s: %s",
                maf_path,
                error,
            )

            failed_files.append(
                (
                    file_id,
                    str(error),
                )
            )

            qc_records.append(
                {
                    "file_id": file_id,
                    "filename": maf_path.name,
                    "case_id": (
                        file_id_to_case_id.get(
                            file_id,
                            "",
                        )
                    ),
                    "status": "failed",
                    "error": str(error),
                }
            )

    if not patient_features:
        log.error(
            "No patient produced qualifying mutation features."
        )
        sys.exit(1)

    matrix_rows, matrix_columns, matrix_nonzeros = (
        save_sparse_matrix(
            patient_features=patient_features,
            case_metadata=case_metadata,
            output_dir=output_dir,
        )
    )

    long_table_path, long_mutation_count = (
        save_long_mutations(
            long_tables=long_tables,
            output_dir=output_dir,
        )
    )

    patient_metadata = save_patient_metadata(
        patient_features=patient_features,
        case_metadata=case_metadata,
        file_counts=file_counts,
        mutation_counts=mutation_counts,
        output_dir=output_dir,
    )

    save_qc(
        qc_records=qc_records,
        output_dir=output_dir,
    )

    save_summary(
        output_dir=output_dir,
        manifest_count=len(manifest),
        maf_count=len(maf_files),
        empty_mafs=empty_mafs,
        silent_only_mafs=silent_only_mafs,
        failed_files=failed_files,
        patient_metadata=patient_metadata,
        matrix_rows=matrix_rows,
        matrix_columns=matrix_columns,
        matrix_nonzeros=matrix_nonzeros,
        long_table_path=long_table_path,
        long_mutation_count=long_mutation_count,
    )

    log.info(
        "CPTAC mutation extraction finished."
    )


if __name__ == "__main__":
    main()