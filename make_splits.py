#!/usr/bin/env python3
"""
make_splits.py

Builds ONE canonical patient cohort table (joining the mutation side,
case_id-keyed, with the WSI side, barcode-keyed) and generates three
split schemes from it, all saved to --output-dir so the mutation
classifier and the WSI/MIL pipeline read the exact same splits -- no
risk of the two modalities silently training/testing on different
patient sets.

SPLITS PRODUCED
    1. train_val_test_split.csv  a single stratified train/val/test split
                                  (stratified by tumor_origin)
    2. stratified_5fold.csv      5-fold CV, stratified by tumor_origin
                                  (StratifiedKFold) -- standard CV, no
                                  control for which institution/site a
                                  patient came from
    3. site_stratified_5fold.csv 5-fold CV, stratified by tumor_origin AND
                                  grouped by TSS (Tissue Source Site --
                                  the institution that submitted the
                                  sample, parsed from the TCGA barcode,
                                  e.g. "AA" in TCGA-AA-0001) so that all
                                  of one site's patients land in the same
                                  fold. This matters because site is a
                                  known confounder in computational
                                  pathology (scanner/staining batch
                                  effects) -- without this, a model can
                                  look like it's learning tumor biology
                                  when it's partly learning to recognize
                                  which institution scanned the slide.

Both fold files record each patient's fold index (0-4); downstream code
just filters by fold to get that fold's train (fold != k) and test
(fold == k) sets, rather than needing five separate files.

USAGE
    python make_splits.py \
        --matched-cohort-csv /workspace/matched_cohort_uni/matched_cohort.csv \
        --mutations-long /workspace/mutation_tables/mutations_long.parquet \
        --output-dir /workspace/splits \
        --test-size 0.2 --val-size 0.1 --n-folds 5

OUTPUT (under --output-dir)
    patient_cohort.csv           canonical reference table: case_id, barcode,
                                  cohort, tumor_origin, site -- one row per
                                  matched patient
    train_val_test_split.csv     case_id, barcode, tumor_origin, split
    stratified_5fold.csv         case_id, barcode, tumor_origin, fold
    site_stratified_5fold.csv    case_id, barcode, tumor_origin, site, fold
    split_summary.txt            every printed table below, saved to a file
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split

from cancer_mapping import TCGA_COHORT_TO_TUMOR_ORIGIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("make_splits")

def build_patient_cohort(matched_cohort_csv: Path, mutations_long: Path, output_dir: Path) -> pd.DataFrame:
    """
    Joins the WSI side (matched_cohort.csv: barcode -> cohort) with the
    mutation side (mutations_long.parquet: case_id -> patient_barcode) into
    one canonical table, so downstream splits can be used by either
    pipeline regardless of which ID they key on.

    barcode is only needed INTERNALLY here, as the join key linking the
    two sides (there's no direct case_id<->cohort link without it) --
    it's not carried into the main cohort/split files, which are kept to
    case_id + tumor_origin (+ site, needed for site-stratified grouping).
    Scripts that still need to resolve case_id -> barcode (e.g. MIL
    training, which finds its .h5 files by barcode filename) should read
    the small separate case_id_barcode_map.csv this function also saves,
    rather than the main files.
    """
    matched_cohort = pd.read_csv(matched_cohort_csv)
    matched_cohort["tumor_origin"] = matched_cohort["cohort"].map(
        lambda c: TCGA_COHORT_TO_TUMOR_ORIGIN.get(c, c)
    )

    mutations = pd.read_parquet(mutations_long, columns=["case_id", "patient_barcode"])
    case_to_barcode = mutations.drop_duplicates("case_id").rename(
        columns={"patient_barcode": "barcode"}
    )[["case_id", "barcode"]]

    cohort = matched_cohort.merge(case_to_barcode, on="barcode", how="inner")
    cohort = cohort[["case_id", "tumor_origin", "barcode", "cohort"]].drop_duplicates("barcode")

    # TSS (Tissue Source Site) -- second segment of a TCGA barcode,
    # e.g. "AA" in TCGA-AA-0001. This is the submitting institution, used
    # as the group for site-stratified CV. Derived from barcode BEFORE
    # barcode gets dropped from the main table below.
    cohort["site"] = cohort["barcode"].str.split("-").str[1]

    n_unmatched = len(matched_cohort) - len(cohort)
    if n_unmatched > 0:
        log.warning(f"{n_unmatched} patients in matched_cohort.csv had no matching case_id "
                     f"in mutations_long.parquet -- excluded from the split (likely already "
                     f"removed by extract_muts.py's empty-MAF cleanup)")

    # separate bridge file -- ONLY place barcode lives going forward
    bridge_path = output_dir / "case_id_barcode_map.csv"
    cohort[["case_id", "barcode"]].to_csv(bridge_path, index=False)
    log.info(f"Saved case_id<->barcode bridge file to {bridge_path} "
             f"(needed by scripts that locate files by barcode, e.g. MIL training)")

    cohort = cohort[["case_id", "tumor_origin", "site"]]

    return cohort.reset_index(drop=True)


def print_and_save_counts(title: str, df: pd.DataFrame, group_col: str, class_col: str, out_lines: list):
    header = f"\n=== {title} ==="
    log.info(header)
    out_lines.append(header)

    overall = df[group_col].value_counts().sort_index()
    line = f"Overall counts per {group_col}:\n{overall.to_string()}"
    log.info(line)
    out_lines.append(line)

    crosstab = pd.crosstab(df[class_col], df[group_col])
    line = f"\nPer-class breakdown:\n{crosstab.to_string()}"
    log.info(line)
    out_lines.append(line)


def main():
    parser = argparse.ArgumentParser(description="Build shared train/val/test and CV splits for mutation + WSI pipelines")
    parser.add_argument("--matched-cohort-csv", required=True)
    parser.add_argument("--mutations-long", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-class-size", type=int, default=5,
                         help="Drop tumor origins with fewer patients than this "
                              "(too few to stratify across folds reliably)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_lines = []

    log.info("Building canonical patient cohort...")
    cohort = build_patient_cohort(Path(args.matched_cohort_csv), Path(args.mutations_long), output_dir)
    log.info(f"Canonical cohort: {len(cohort)} patients, {cohort['tumor_origin'].nunique()} tumor origins, "
             f"{cohort['site'].nunique()} unique sites")

    class_counts = cohort["tumor_origin"].value_counts()
    small_classes = class_counts[class_counts < args.min_class_size].index.tolist()
    if small_classes:
        log.warning(f"Dropping {len(small_classes)} tumor origin(s) with fewer than "
                     f"{args.min_class_size} patients: {small_classes}")
        cohort = cohort[~cohort["tumor_origin"].isin(small_classes)].reset_index(drop=True)

    cohort_path = output_dir / "patient_cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    log.info(f"Saved canonical cohort table to {cohort_path}")

    y = cohort["tumor_origin"].to_numpy()
    groups = cohort["site"].to_numpy()

    # ------------------------------------------------------------------
    # 1. single stratified train/val/test split
    # ------------------------------------------------------------------
    idx_all = np.arange(len(cohort))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=args.val_size / (1 - args.test_size),
        stratify=y[idx_train], random_state=args.random_state,
    )

    split_col = np.empty(len(cohort), dtype=object)
    split_col[idx_train] = "train"
    split_col[idx_val] = "val"
    split_col[idx_test] = "test"

    tvt = cohort[["case_id", "tumor_origin"]].copy()
    tvt["split"] = split_col
    tvt_path = output_dir / "train_val_test_split.csv"
    tvt.to_csv(tvt_path, index=False)
    log.info(f"Saved train/val/test split to {tvt_path}")

    print_and_save_counts("Train / Val / Test split", tvt, "split", "tumor_origin", out_lines)

    # ------------------------------------------------------------------
    # 2. stratified 5-fold CV (no site control)
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    fold_col = np.empty(len(cohort), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(idx_all, y)):
        fold_col[test_idx] = fold_idx

    strat_cv = cohort[["case_id", "tumor_origin"]].copy()
    strat_cv["fold"] = fold_col
    strat_cv_path = output_dir / "stratified_5fold.csv"
    strat_cv.to_csv(strat_cv_path, index=False)
    log.info(f"Saved stratified {args.n_folds}-fold CV assignment to {strat_cv_path}")

    print_and_save_counts(f"Stratified {args.n_folds}-fold CV", strat_cv, "fold", "tumor_origin", out_lines)

    # ------------------------------------------------------------------
    # 3. site-stratified 5-fold CV (stratified by class, grouped by site)
    # ------------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    site_fold_col = np.empty(len(cohort), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(idx_all, y, groups)):
        site_fold_col[test_idx] = fold_idx

    site_cv = cohort[["case_id", "tumor_origin", "site"]].copy()
    site_cv["fold"] = site_fold_col
    site_cv_path = output_dir / "site_stratified_5fold.csv"
    site_cv.to_csv(site_cv_path, index=False)
    log.info(f"Saved site-stratified {args.n_folds}-fold CV assignment to {site_cv_path}")

    print_and_save_counts(f"Site-stratified {args.n_folds}-fold CV", site_cv, "fold", "tumor_origin", out_lines)

    # sanity check: confirm no site appears in more than one fold
    site_fold_counts = site_cv.groupby("site")["fold"].nunique()
    leaking_sites = site_fold_counts[site_fold_counts > 1]
    check_line = (f"\nSite-leakage check: {len(leaking_sites)} site(s) appear in more than one fold "
                   f"(should be 0)")
    log.info(check_line)
    out_lines.append(check_line)
    if len(leaking_sites) > 0:
        log.warning(f"Sites appearing in multiple folds: {leaking_sites.to_dict()}")
        out_lines.append(f"Leaking sites: {leaking_sites.to_dict()}")

    summary_path = output_dir / "split_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(out_lines))
    log.info(f"\nSaved full summary to {summary_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()