#!/usr/bin/env python3
"""
extract_cna_CPTAC.py

Extracts CPTAC GDC ASCAT gene-level copy-number files into a binary
patient-level CNA matrix compatible with the TCGA fused feature pipeline.

TCGA-compatible feature rules:
    First estimate integer sample ploidy as the mode of autosomal gene-level
    copy-number values (ties resolved upward).

    copy_number == 0                  -> <GENE>_DEL = 1
    copy_number >= 2 * sample_ploidy -> <GENE>_AMP = 1
    either event                      -> <GENE>_ANY_CNA = 1

    Shallow loss (-1), neutral (0), and low-level gain (+1) are not emitted,
    because the TCGA training pipeline only retained GISTIC -2 and +2 events.

COLUMN NAMING (IMPORTANT -- fixed from the original version of this script)
    The real, current TCGA-side CNA extraction script
    (extract_cna_gistic.py) produces columns prefixed "cna_" (single
    underscore, e.g. "cna_ERBB2_amp"), each gene contributing BOTH a
    broad "cna_<GENE>_any" indicator AND a state-specific
    "cna_<GENE>_<del|loss|gain|amp>" indicator.

    The original version of this script's load_training_cna_columns()
    filtered for columns starting with "cna__" (DOUBLE underscore) --
    which never matches any real training column, since the real prefix
    is "cna_" (single underscore). This caused training_cna_full to come
    back completely empty, and the CNA matrix built by this script would
    have zero columns. Confirmed directly by testing the old filter
    against real training column names before this fix.

    Additionally, the original CNA_SUFFIXES tuple only handled
    ("amp", "del", "gain", "loss") -- it never built a mapping for the
    "_any" broad indicator that extract_cna_gistic.py also produces for
    every gene with a qualifying alteration, meaning even a
    correctly-prefixed version would have been unable to populate that
    feature. Both issues are fixed below.
"""

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

# real prefix as baked in by extract_cna_gistic.py -- confirmed directly
# against its actual source (single underscore, not "cna__")
CNA_PREFIX = "cna_"


def clean(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "unknown", "not reported", "not available"}:
        return ""
    return text


def map_primary_site(primary_site):
    text = clean(primary_site).lower()
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


def read_manifest(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "id" not in df.columns:
        raise ValueError("Manifest must contain an 'id' column")
    return df


def query_batch(file_ids, timeout=120, retries=5):
    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "files.file_id", "value": file_ids},
        },
        "format": "JSON",
        "size": len(file_ids),
        "fields": ",".join([
            "file_id",
            "file_name",
            "cases.case_id",
            "cases.submitter_id",
            "cases.project.project_id",
            "cases.primary_site",
            "cases.disease_type",
        ]),
    }

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(GDC_FILES_ENDPOINT, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("data", {}).get("hits", [])
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError(f"GDC metadata request failed: {last_error}")


def get_metadata(manifest, cache_path, refresh=False, batch_size=300):
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype=str).fillna("")

    ids = manifest["id"].astype(str).drop_duplicates().tolist()
    records = []

    for start in tqdm(range(0, len(ids), batch_size), desc="Querying GDC CNA metadata", unit="batch"):
        for hit in query_batch(ids[start:start + batch_size]):
            file_id = clean(hit.get("file_id") or hit.get("id"))
            file_name = clean(hit.get("file_name"))
            for case in hit.get("cases", []) or []:
                project = case.get("project", {}) or {}
                records.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "case_id": clean(case.get("case_id")),
                    "case_submitter_id": clean(case.get("submitter_id")),
                    "project_id": clean(project.get("project_id")),
                    "primary_site": clean(case.get("primary_site")),
                    "disease_type": clean(case.get("disease_type")),
                })

    result = pd.DataFrame(records).drop_duplicates()
    result.to_csv(cache_path, index=False)
    return result


# real suffixes as produced by extract_cna_gistic.py: broad "_any" plus
# the four state-specific suffixes -- the original version of this
# script was missing "any", meaning it could never populate that feature
# even once the prefix bug was fixed
CNA_SUFFIXES = ("any", "amp", "del", "gain", "loss")


def load_training_cna_columns(path):
    full_columns = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip().startswith(CNA_PREFIX)
    ]
    base_to_columns = defaultdict(dict)

    for full in full_columns:
        body = full[len(CNA_PREFIX):]
        for suffix in CNA_SUFFIXES:
            token = "_" + suffix
            if body.endswith(token):
                base = body[:-len(token)]
                base_to_columns[base][suffix] = body
                break

    return full_columns, base_to_columns


def choose_training_identifier(gene_name, gene_id, training_bases):
    gene_name = clean(gene_name)
    gene_id = clean(gene_id)

    candidates = []
    if gene_name and gene_id:
        candidates.append(f"{gene_name}|{gene_id}")
        if "." in gene_id:
            candidates.append(f"{gene_name}|{gene_id.split('.')[0]}")
    if gene_name:
        candidates.append(gene_name)

    for candidate in candidates:
        if candidate in training_bases:
            return candidate
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cna-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--align-to-columns-file", required=True)
    parser.add_argument("--refresh-metadata", action="store_true")
    args = parser.parse_args()

    cna_dir = Path(args.cna_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(Path(args.manifest))
    training_cna_full, training_base_map = load_training_cna_columns(
        args.align_to_columns_file
    )
    training_bases = set(training_base_map)

    print(f"Loaded {len(training_cna_full)} training CNA columns "
          f"({len(training_bases)} unique genes) using prefix '{CNA_PREFIX}'")
    if not training_cna_full:
        print(f"WARNING: zero training columns matched prefix '{CNA_PREFIX}' -- "
              f"check --align-to-columns-file actually contains CNA columns, "
              f"and that they use this prefix. Sample of first 5 lines from "
              f"that file for reference:")
        sample = Path(args.align_to_columns_file).read_text().splitlines()[:5]
        for line in sample:
            print(f"  {line}")

    wanted_ids = set(manifest["id"].astype(str))

    cna_files = sorted(
        p for p in cna_dir.rglob("*.tsv")
        if p.parent.name in wanted_ids
        and "gene_level.copy_number_variation" in p.name
        and "/logs/" not in str(p)
    )

    print(f"Manifest records: {len(manifest)}")
    print(f"Matching gene-level CNA TSV files: {len(cna_files)}")

    metadata = get_metadata(
        manifest,
        output_dir / "gdc_cna_metadata_cache.csv",
        refresh=args.refresh_metadata,
    )

    file_to_case = {}
    case_meta = {}
    for _, row in metadata.iterrows():
        fid = clean(row.get("file_id"))
        cid = clean(row.get("case_id"))
        if fid and cid:
            file_to_case[fid] = cid
        if cid:
            case_meta[cid] = {
                "case_id": cid,
                "case_submitter_id": clean(row.get("case_submitter_id")),
                "project_id": clean(row.get("project_id")),
                "primary_site": clean(row.get("primary_site")),
                "disease_type": clean(row.get("disease_type")),
                "tumor_origin": map_primary_site(row.get("primary_site")),
            }

    patient_features = defaultdict(set)
    file_counts = Counter()
    gain_counts = Counter()
    loss_counts = Counter()
    qc = []
    failures = []

    for path in tqdm(cna_files, desc="Extracting CPTAC CNA", unit="file"):
        file_id = path.parent.name
        case_id = file_to_case.get(file_id, "")
        if not case_id:
            failures.append((file_id, "No case_id metadata"))
            continue

        try:
            df = pd.read_csv(
                path,
                sep="\t",
                usecols=["gene_id", "gene_name", "chromosome", "copy_number"],
                dtype={"gene_id": str, "gene_name": str, "chromosome": str},
            )
            df["copy_number"] = pd.to_numeric(df["copy_number"], errors="coerce")
            df["gene_id"] = df["gene_id"].astype(str).str.strip()
            df["gene_name"] = df["gene_name"].astype(str).str.strip()
            df["chromosome"] = df["chromosome"].astype(str).str.strip()
            df = df[df["gene_name"].ne("") & df["copy_number"].notna()].copy()

            autosomal = df[
                df["chromosome"]
                .str.replace("chr", "", regex=False)
                .isin([str(i) for i in range(1, 23)])
            ]
            ploidy_source = autosomal if not autosomal.empty else df
            modes = ploidy_source["copy_number"].mode()
            if modes.empty:
                raise ValueError("Could not estimate sample ploidy")
            sample_ploidy = max(1, int(modes.max()))

            # Ensure patients with fully neutral CNA are still represented.
            patient_features[case_id]

            state_counts = Counter()
            unmatched_identifiers = 0
            file_gene_states = {}

            for row in df.itertuples(index=False):
                base = choose_training_identifier(
                    row.gene_name, row.gene_id, training_bases
                )
                if not base:
                    unmatched_identifiers += 1
                    continue

                cn = float(row.copy_number)

                if cn == 0:
                    state = "del"
                elif 0 < cn < sample_ploidy:
                    state = "loss"
                elif cn == sample_ploidy:
                    state = None
                elif sample_ploidy < cn < (2 * sample_ploidy):
                    state = "gain"
                elif cn >= (2 * sample_ploidy):
                    state = "amp"
                else:
                    state = None

                if state is not None:
                    file_gene_states[base] = state

            # Mutually exclusive specific state per gene, PLUS the broad
            # "any" indicator whenever a specific state was assigned --
            # matches extract_cna_gistic.py's real two-column-per-alteration
            # design (a broad flag plus a state-specific flag).
            for base, state in file_gene_states.items():
                specific_feature = training_base_map.get(base, {}).get(state)
                any_feature = training_base_map.get(base, {}).get("any")
                if specific_feature:
                    patient_features[case_id].add(specific_feature)
                    state_counts[state] += 1
                if any_feature:
                    patient_features[case_id].add(any_feature)

            file_counts[case_id] += 1
            gain_counts[case_id] += state_counts["amp"] + state_counts["gain"]
            loss_counts[case_id] += state_counts["del"] + state_counts["loss"]

            qc.append({
                "file_id": file_id,
                "filename": path.name,
                "case_id": case_id,
                "rows": len(df),
                "estimated_integer_ploidy": sample_ploidy,
                "amp_genes": state_counts["amp"],
                "gain_genes": state_counts["gain"],
                "del_genes": state_counts["del"],
                "loss_genes": state_counts["loss"],
                "unmatched_training_identifiers": unmatched_identifiers,
                "status": "processed",
            })
        except Exception as error:
            failures.append((file_id, str(error)))
            qc.append({
                "file_id": file_id,
                "filename": path.name,
                "case_id": case_id,
                "status": "failed",
                "error": str(error),
            })

    case_ids = sorted(patient_features)
    # internal indexing uses stripped (no-prefix) names throughout this
    # script -- that's fine and consistent. But the SAVED columns file
    # needs the prefix restored here, since validate_CPTAC.py expects
    # CNA columns to already carry "cna_" from extraction (matching
    # extract_cna_gistic.py's real TCGA-side behavior) -- confirmed
    # directly from a real run: without this, saved columns came out as
    # bare "7SK|ENSG00000232512.2_amp" instead of the correct
    # "cna_7SK|ENSG00000232512.2_amp", so they never matched anything
    # during validate_CPTAC.py's alignment step.
    stripped_columns = [full[len(CNA_PREFIX):] for full in training_cna_full]
    columns = [f"{CNA_PREFIX}{name}" for name in stripped_columns]
    col_index = {name: i for i, name in enumerate(stripped_columns)}

    rows, cols = [], []
    for row_idx, case_id in enumerate(case_ids):
        for feature in patient_features[case_id]:
            rows.append(row_idx)
            cols.append(col_index[feature])

    matrix = coo_matrix(
        (
            np.ones(len(rows), dtype=np.int8),
            (np.asarray(rows), np.asarray(cols)),
        ),
        shape=(len(case_ids), len(columns)),
        dtype=np.int8,
    ).tocsr()

    save_npz(output_dir / "patient_cna_matrix.npz", matrix)
    (output_dir / "patient_cna_matrix_columns.txt").write_text("\n".join(columns))

    with (output_dir / "patient_cna_matrix_rows.txt").open("w") as handle:
        for case_id in case_ids:
            origin = case_meta.get(case_id, {}).get("tumor_origin", "Unknown")
            handle.write(f"{origin}\t{case_id}\n")

    metadata_rows = []
    for case_id in case_ids:
        record = dict(case_meta.get(case_id, {}))
        record.update({
            "case_id": case_id,
            "cna_file_count": file_counts[case_id],
            "amplification_feature_count": gain_counts[case_id],
            "deep_deletion_feature_count": loss_counts[case_id],
            "total_cna_feature_count": len(patient_features[case_id]),
        })
        metadata_rows.append(record)

    pd.DataFrame(metadata_rows).to_csv(output_dir / "patient_metadata.csv", index=False)
    pd.DataFrame(qc).to_csv(output_dir / "patient_cna_qc.csv", index=False)

    origin_counts = Counter(
        case_meta.get(case_id, {}).get("tumor_origin", "Unknown")
        for case_id in case_ids
    )

    lines = [
        "CPTAC CNA EXTRACTION SUMMARY",
        "=" * 70,
        f"Manifest records: {len(manifest)}",
        f"Matching CNA TSV files: {len(cna_files)}",
        f"Unique patients: {len(case_ids)}",
        f"CNA features: {len(columns)}",
        f"Matrix nonzero entries: {matrix.nnz}",
        f"Failed files: {len(failures)}",
        "",
        "Tumor origins:",
    ]
    lines.extend(f"  {k}: {v}" for k, v in sorted(origin_counts.items()))
    if failures:
        lines.extend(["", "Failures:"])
        lines.extend(f"  {fid}: {reason}" for fid, reason in failures[:100])

    (output_dir / "extraction_summary.txt").write_text("\n".join(lines))

    print(f"Saved matrix: {matrix.shape[0]} patients x {matrix.shape[1]} CNA features")
    print(f"Nonzero entries: {matrix.nnz}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()