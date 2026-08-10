import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.tcga.tumor_origin_mapping import TCGA_COHORT_TO_TUMOR_ORIGIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("make_splits")


def build_patient_cohort(
    matched_cohort_csv: Path,
    mutations_long: Path,
    output_dir: Path,
) -> pd.DataFrame:
    matched_cohort = pd.read_csv(matched_cohort_csv)
    matched_cohort["tumor_origin"] = matched_cohort["cohort"].map(
        lambda c: TCGA_COHORT_TO_TUMOR_ORIGIN.get(c, c)
    )

    mutations = pd.read_parquet(
        mutations_long,
        columns=["case_id", "patient_barcode"],
    )

    case_to_barcode = mutations.drop_duplicates("case_id").rename(
        columns={"patient_barcode": "barcode"}
    )[["case_id", "barcode"]]

    cohort = matched_cohort.merge(
        case_to_barcode,
        on="barcode",
        how="inner",
    )

    cohort = cohort[
        ["case_id", "tumor_origin", "barcode", "cohort"]
    ].drop_duplicates("barcode")

    cohort["site"] = cohort["barcode"].str.split("-").str[1]

    n_unmatched = len(matched_cohort) - len(cohort)

    if n_unmatched > 0:
        log.warning(
            f"{n_unmatched} patients in matched_cohort.csv had no matching "
            f"case_id in mutations_long.parquet and were excluded"
        )

    bridge_path = output_dir / "case_id_barcode_map.csv"
    cohort[["case_id", "barcode"]].to_csv(
        bridge_path,
        index=False,
    )

    log.info(
        f"Saved case_id/barcode map to {bridge_path}"
    )

    cohort = cohort[
        ["case_id", "tumor_origin", "site"]
    ]

    return cohort.reset_index(drop=True)


def print_and_save_counts(
    title: str,
    df: pd.DataFrame,
    group_col: str,
    class_col: str,
    out_lines: list,
):
    header = f"\n=== {title} ==="
    log.info(header)
    out_lines.append(header)

    overall = df[group_col].value_counts().sort_index()

    line = (
        f"Overall counts per {group_col}:\n"
        f"{overall.to_string()}"
    )

    log.info(line)
    out_lines.append(line)

    crosstab = pd.crosstab(
        df[class_col],
        df[group_col],
    )

    line = (
        f"\nPer-class breakdown:\n"
        f"{crosstab.to_string()}"
    )

    log.info(line)
    out_lines.append(line)


def main():
    parser = argparse.ArgumentParser(
        description="Build site-stratified cross-validation splits"
    )

    parser.add_argument(
        "--matched-cohort-csv",
        required=True,
    )

    parser.add_argument(
        "--mutations-long",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-class-size",
        type=int,
        default=5,
        help="Drop tumor origins with fewer patients than this",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_lines = []

    log.info("Building canonical patient cohort")

    cohort = build_patient_cohort(
        Path(args.matched_cohort_csv),
        Path(args.mutations_long),
        output_dir,
    )

    log.info(
        f"Canonical cohort: {len(cohort)} patients, "
        f"{cohort['tumor_origin'].nunique()} tumor origins, "
        f"{cohort['site'].nunique()} sites"
    )

    class_counts = cohort[
        "tumor_origin"
    ].value_counts()

    small_classes = class_counts[
        class_counts < args.min_class_size
    ].index.tolist()

    if small_classes:
        log.warning(
            f"Dropping {len(small_classes)} tumor origin(s) "
            f"with fewer than {args.min_class_size} patients: "
            f"{small_classes}"
        )

        cohort = cohort[
            ~cohort["tumor_origin"].isin(
                small_classes
            )
        ].reset_index(drop=True)

    cohort_path = (
        output_dir / "patient_cohort.csv"
    )

    cohort.to_csv(
        cohort_path,
        index=False,
    )

    log.info(
        f"Saved patient cohort to {cohort_path}"
    )

    idx_all = np.arange(len(cohort))
    y = cohort["tumor_origin"].to_numpy()
    groups = cohort["site"].to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    site_fold_col = np.empty(
        len(cohort),
        dtype=int,
    )

    for fold_idx, (_, test_idx) in enumerate(
        sgkf.split(
            idx_all,
            y,
            groups,
        )
    ):
        site_fold_col[test_idx] = fold_idx

    site_cv = cohort[
        ["case_id", "tumor_origin", "site"]
    ].copy()

    site_cv["fold"] = site_fold_col

    site_cv_path = (
        output_dir
        / "site_stratified_5fold.csv"
    )

    site_cv.to_csv(
        site_cv_path,
        index=False,
    )

    log.info(
        f"Saved site-stratified "
        f"{args.n_folds}-fold split to "
        f"{site_cv_path}"
    )

    print_and_save_counts(
        f"Site-stratified {args.n_folds}-fold CV",
        site_cv,
        "fold",
        "tumor_origin",
        out_lines,
    )

    site_fold_counts = (
        site_cv.groupby("site")["fold"]
        .nunique()
    )

    leaking_sites = site_fold_counts[
        site_fold_counts > 1
    ]

    check_line = (
        f"\nSite-leakage check: "
        f"{len(leaking_sites)} site(s) "
        f"appear in more than one fold "
        f"(should be 0)"
    )

    log.info(check_line)
    out_lines.append(check_line)

    if len(leaking_sites) > 0:
        log.warning(
            f"Sites appearing in multiple folds: "
            f"{leaking_sites.to_dict()}"
        )

        out_lines.append(
            f"Leaking sites: "
            f"{leaking_sites.to_dict()}"
        )

    summary_path = (
        output_dir
        / "split_summary.txt"
    )

    with open(summary_path, "w") as f:
        f.write(
            "\n".join(out_lines)
        )

    log.info(
        f"Saved summary to {summary_path}"
    )


if __name__ == "__main__":
    main()