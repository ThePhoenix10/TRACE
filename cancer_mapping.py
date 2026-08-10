"""
cancer_mapping.py

SINGLE SOURCE OF TRUTH for tumor origin labels across the whole project.
Every script that needs to turn a raw TCGA cohort code into a tumor
origin (organ/anatomical site) imports from HERE -- not from its own
local copy.

Maps DIRECTLY from TCGA cohort code to organ -- there is no intermediate
"full disease name" layer (e.g. no "Bladder Urothelial Carcinoma"). Some
organs are shared by multiple cohort codes, which is expected and
intentional (e.g. TCGA-KICH, TCGA-KIRC, and TCGA-KIRP all map to
"Kidney" -- they're the same organ, different histological subtypes).

If you ever want to change how tumor origins are labeled, this is the
ONLY file that needs editing -- every script that imports from here
picks up the change automatically the next time it runs.
"""

# TCGA project code -> organ / tumor origin. BRCA_IDC / BRCA_OTHERS are
# UNI2-h's own split of TCGA-BRCA into IDC vs non-IDC subtypes, not
# official TCGA project codes -- both map to the same organ. Mesothelioma
# is grouped with Lung here (pleura); change this if you'd rather treat
# it as its own site.
TCGA_COHORT_TO_TUMOR_ORIGIN = {
    "TCGA-ACC": "Adrenal Gland/Paraganglia",
    "TCGA-BLCA": "Bladder",
    "TCGA-BRCA": "Breast",
    "TCGA-BRCA_IDC": "Breast",
    "TCGA-BRCA_OTHERS": "Breast",
    "TCGA-CESC": "Cervix",
    "TCGA-CHOL": "Bile Duct",
    "TCGA-COAD": "Colon",
    "TCGA-DLBC": "Lymphatic System",
    "TCGA-ESCA": "Esophagus",
    "TCGA-GBM": "Brain",
    "TCGA-HNSC": "Head and Neck",
    "TCGA-KICH": "Kidney",
    "TCGA-KIRC": "Kidney",
    "TCGA-KIRP": "Kidney",
    "TCGA-LGG": "Brain",
    "TCGA-LIHC": "Liver",
    "TCGA-LUAD": "Lung",
    "TCGA-LUSC": "Lung",
    "TCGA-MESO": "Pleura/Mesothelium",
    "TCGA-OV": "Ovary",
    "TCGA-PAAD": "Pancreas",
    "TCGA-PCPG": "Adrenal Gland/Paraganglia",
    "TCGA-PRAD": "Prostate",
    "TCGA-READ": "Rectum",
    "TCGA-SARC": "Soft Tissue",
    "TCGA-SKCM": "Skin",
    "TCGA-STAD": "Stomach",
    "TCGA-TGCT": "Testis",
    "TCGA-THCA": "Thyroid",
    "TCGA-THYM": "Thymus",
    "TCGA-UCEC": "Uterus",
    "TCGA-UCS": "Uterus",
    "TCGA-UVM": "Eye",
}


def cohort_to_tumor_origin(cohort_code: str) -> str:
    """TCGA cohort code -> tumor origin (organ). Falls back to the raw code if unmapped."""
    return TCGA_COHORT_TO_TUMOR_ORIGIN.get(cohort_code, cohort_code)