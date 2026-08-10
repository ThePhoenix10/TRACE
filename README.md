# TRACE

### Multimodal Fusion of Pathology Foundation-Model Embeddings and Genomics for Tumor Origin Prediction

TRACE is a multimodal machine learning framework for tumor tissue-of-origin prediction that integrates histopathology foundation-model representations with somatic genomic features.

Using The Cancer Genome Atlas (TCGA) for model development, the histopathology branch uses precomputed UNI2-h tile embeddings modeled with ABMIL, CLAM, and TransMIL. The genomic branch uses XGBoost trained on somatic mutation, copy-number alteration (CNA), and SBS96 trinucleotide-context features. Patient-level predictions from both modalities are integrated using logistic-regression stacking and externally validated using the Clinical Proteomic Tumor Analysis Consortium (CPTAC).

## Paper

**TRACE: Multimodal Fusion of Pathology Foundation-Model Embeddings and Genomics for Tumor Origin Prediction**

Currently under review.

## Repository Structure

```text
TRACE/
├── data/
│   ├── tcga/
│   │   ├── extract_tcga_mutations.py
│   │   ├── extract_tcga_cna.py
│   │   ├── extract_tcga_sbs96.py
│   │   ├── fuse_tcga_tables.py
│   │   ├── make_tcga_splits.py
│   │   └── tumor_origin_mapping.py
│   │
│   └── cptac/
│       ├── extract_cptac_mutations.py
│       ├── extract_cptac_cna.py
│       ├── extract_cptac_sbs96.py
│       └── fuse_cptac_tables.py
│
├── interpretability/
│   ├── SHAP/
│   │   ├── figures/
│   │   └── shap.py
│   │
│   └── ablation/
│       ├── figures/
│       └── ablation.py
│
├── models/
│   ├── genomics/
│   │   ├── figures/
│   │   └── tcga_genomic_classifier.py
│   │
│   ├── histology/
│   │   ├── figures/
│   │   └── tcga_histology_classifier.py
│   │
│   ├── fusion/
│   │   ├── confusion_matrix.png
│   │   └── tcga_fusion_classifier.py
│   │
│   └── validation/
│       └── cptac_external_validation.py
│
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/ThePhoenix10/TRACE.git
cd TRACE
```

## Authors

- Saicharan Vellanki — Issaquah High School
- Paraic Kenny, PhD — Gundersen Medical Foundation

## Contact

- [saivellanki10@gmail.com](mailto:saivellanki10@gmail.com)
- [pakenny@emplifyhealth.org](mailto:pakenny@emplifyhealth.org)

## License

MIT License

Copyright (c) 2026 Saicharan Vellanki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
