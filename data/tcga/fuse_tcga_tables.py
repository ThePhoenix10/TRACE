import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, load_npz, save_npz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fuse_tables")


UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def load_table(prefix: str, label: str = None):


    prefix_path = Path(prefix)
    npz_path = Path(f"{prefix}.npz")
    rows_path = Path(f"{prefix}_rows.txt")
    columns_path = Path(f"{prefix}_columns.txt")

    for p in (npz_path, rows_path, columns_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    matrix = load_npz(npz_path)

    rows_raw = rows_path.read_text().splitlines()
    case_ids, tumor_origins = [], []
    for line in rows_raw:
        field_a, field_b = line.split("\t")


        if UUID_PATTERN.match(field_a):
            case_id, tumor_origin = field_a, field_b
        else:
            tumor_origin, case_id = field_a, field_b
        case_ids.append(case_id)
        tumor_origins.append(tumor_origin)

    columns = columns_path.read_text().splitlines()

    if label is None:
        label = prefix_path.name

    case_id_to_tumor_origin = dict(zip(case_ids, tumor_origins))
    case_id_to_row_idx = {cid: i for i, cid in enumerate(case_ids)}

    return matrix, case_id_to_tumor_origin, case_id_to_row_idx, columns, label


def parse_table_arg(table_arg: str):

    if ":" in table_arg:
        prefix, label = table_arg.rsplit(":", 1)
        return prefix, label
    return table_arg, None


def main():
    parser = argparse.ArgumentParser(description="Fuse any number of per-patient sparse feature tables")
    parser.add_argument("--table", action="append", required=True, dest="tables",
                         help="path_prefix:label for one table. Repeat this flag for each table "
                              "you want to fuse. Label is optional (defaults to the prefix's own "
                              "filename if omitted).")
    parser.add_argument("--output-prefix", required=True,
                         help="Path prefix for the fused output (no extension -- .npz/_rows.txt/"
                              "_columns.txt are appended automatically)")
    args = parser.parse_args()

    log.info(f"Fusing {len(args.tables)} tables...")

    loaded = []
    for table_arg in args.tables:
        prefix, label = parse_table_arg(table_arg)
        matrix, case_to_type, case_to_idx, columns, resolved_label = load_table(prefix, label)
        log.info(f"  '{resolved_label}': {matrix.shape[0]} patients x {matrix.shape[1]} columns "
                 f"(from {prefix})")
        loaded.append({
            "label": resolved_label, "matrix": matrix, "case_to_type": case_to_type,
            "case_to_idx": case_to_idx, "columns": columns,
        })


    case_id_sets = [set(t["case_to_idx"].keys()) for t in loaded]
    shared_case_ids = sorted(set.intersection(*case_id_sets))
    log.info(f"Patients present in ALL {len(loaded)} tables: {len(shared_case_ids)}")

    for t in loaded:
        n_total = len(t["case_to_idx"])
        n_dropped = n_total - len(shared_case_ids)
        log.info(f"  '{t['label']}': {n_total} total patients, {n_dropped} dropped "
                 f"(not present in every table)")

    if not shared_case_ids:
        log.error("No patients are shared across all tables -- nothing to fuse. "
                    "Check that the tables were built from the same/overlapping patient cohort.")
        return


    mismatches = []
    reference_types = {}
    for cid in shared_case_ids:
        types_seen = {t["case_to_type"].get(cid, "") for t in loaded}
        types_seen.discard("")
        if len(types_seen) > 1:
            mismatches.append((cid, {t["label"]: t["case_to_type"].get(cid, "") for t in loaded}))
        reference_types[cid] = next(iter(types_seen), "")

    if mismatches:
        log.warning(f"{len(mismatches)} patients have DIFFERENT tumor_origin across tables -- "
                     f"using the first non-empty value found for each. See fusion summary for details.")


    sub_matrices = []
    fused_columns = []
    for t in loaded:
        row_order = [t["case_to_idx"][cid] for cid in shared_case_ids]
        sub_matrix = t["matrix"][row_order]
        sub_matrices.append(sub_matrix)


        for col in t["columns"]:
            if col.startswith("som_mut_") or col.startswith("cna_"):
                fused_columns.append(col)
            else:
                fused_columns.append(f"{t['label']}__{col}")

    fused_matrix = hstack(sub_matrices, format="csr")
    log.info(f"Fused matrix: {fused_matrix.shape[0]} patients x {fused_matrix.shape[1]} columns "
             f"({fused_matrix.nnz} non-zero entries)")


    output_prefix = args.output_prefix
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(f"{output_prefix}.npz")
    save_npz(npz_path, fused_matrix)

    rows_path = Path(f"{output_prefix}_rows.txt")
    with open(rows_path, "w") as f:
        f.write("\n".join(f"{cid}\t{reference_types[cid]}" for cid in shared_case_ids))

    columns_path = Path(f"{output_prefix}_columns.txt")
    with open(columns_path, "w") as f:
        f.write("\n".join(fused_columns))

    log.info(f"Saved fused table to:\n  {npz_path}\n  {rows_path}\n  {columns_path}")

    summary_path = Path(f"{output_prefix}_fusion_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Tables fused: {len(loaded)}\n")
        for t in loaded:
            n_total = len(t["case_to_idx"])
            f.write(f"  '{t['label']}': {n_total} patients, {t['matrix'].shape[1]} columns\n")
        f.write(f"\nShared patients (in ALL tables): {len(shared_case_ids)}\n")
        f.write(f"Fused matrix shape: {fused_matrix.shape[0]} patients x {fused_matrix.shape[1]} columns\n")
        f.write(f"\ntumor_origin mismatches across tables: {len(mismatches)}\n")
        if mismatches:
            f.write("First 20 mismatches:\n")
            for cid, types_by_table in mismatches[:20]:
                f.write(f"  {cid}: {types_by_table}\n")

    log.info(f"Saved fusion summary to {summary_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()