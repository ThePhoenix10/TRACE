#!/usr/bin/env python3
"""
train_mil_models.py

Trains ABMIL, CLAM-SB, and TransMIL on your UNI2-h tile embeddings
(embeddings/<barcode>.h5, each containing a variable-length bag of
1536-dim tile features + coords) to predict tumor origin -- a slide-level
Multiple Instance Learning (MIL) problem, no mutation data involved.

MODELS
    abmil     Attention-based Deep MIL (Ilse et al. 2018). Gated attention
              pooling over all tiles in a bag -> weighted slide embedding
              -> classifier head. The simplest, fastest, and a very strong
              MIL baseline in practice.

    clam      CLAM-SB (Lu et al. 2021), single-attention-branch version.
              Same gated attention pooling as ABMIL for the bag-level
              prediction, PLUS an instance-level clustering loss: for each
              training bag, the top-K highest-attention tiles are treated
              as pseudo-positive instances and the bottom-K lowest-
              attention tiles as pseudo-negative instances for that bag's
              true class, and a small per-class instance classifier is
              trained to tell them apart. This auxiliary loss encourages
              the attention mechanism to actually separate discriminative
              tiles from background/irrelevant ones, not just optimize
              the bag-level loss. Instance classifiers are only used
              during training, not at inference.

    transmil  TransMIL (Shao et al. 2021)-style architecture: a learned
              class token + two transformer self-attention layers with a
              PPEG (Pyramid Position Encoding Generator) module in
              between them for positional context. Uses
              F.scaled_dot_product_attention directly (memory-efficient/
              flash-attention backend when available, mathematically
              exact -- not an approximation) rather than
              nn.MultiheadAttention. --max-tiles-transmil (default 4000)
              still randomly subsamples bags larger than that, for this
              model only, as a safety margin against very large bags;
              set to 0 to disable and use the full bag every time.

BAG-LEVEL TRAINING
    MIL bags (slides) have wildly different tile counts, so this trains
    with an effective batch size of 1 bag at a time (standard practice in
    MIL literature/code -- padding to the max bag length in a mini-batch
    would waste huge amounts of memory given how much tile counts vary
    here). --accum-steps lets you simulate a larger effective batch via
    gradient accumulation without needing to pad anything.

USAGE
    python train_mil_models.py \
        --embeddings-dir /workspace/matched_cohort_uni/embeddings \
        --patient-cohort-csv /workspace/splits/patient_cohort.csv \
        --case-id-barcode-map /workspace/splits/case_id_barcode_map.csv \
        --site-fold-csv /workspace/splits/site_stratified_5fold.csv \
        --output-dir /workspace/mil_models \
        --models abmil clam transmil \
        --epochs 20

REQUIREMENTS
    pip install --break-system-packages torch h5py scikit-learn pandas

OUTPUT (under --output-dir)
    <model>_best.pt                  best checkpoint (by val macro F1) per model
    <model>_classification_report.txt  per-class test metrics per model
    <model>_history.csv              per-epoch train/val loss + val macro F1
    model_comparison.csv             test accuracy/macro-F1 for every model, sorted
    label_encoder.joblib
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_mil_models")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_dataset(h5_file, possible_names):
    names = []
    h5_file.visititems(lambda n, obj: names.append(n) if isinstance(obj, h5py.Dataset) else None)
    for candidate in possible_names:
        for name in names:
            if name == candidate or name.split("/")[-1] == candidate:
                return h5_file[name][:]
    raise KeyError(f"Could not find any of {possible_names}. Available: {names}")


def load_bag(h5_path: Path) -> np.ndarray:
    """Loads a slide's tile embeddings, robust to varying UNI2-h key naming."""
    with h5py.File(h5_path, "r") as f:
        feats = find_dataset(f, ["features", "tile_embeds", "tile_embeddings", "embeddings", "feats", "x"])
    feats = np.asarray(feats)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    return feats.astype(np.float32)


class BagDataset(torch.utils.data.Dataset):
    """One item = one slide's full bag of tile embeddings + its label."""

    def __init__(self, h5_paths, labels):
        self.h5_paths = h5_paths
        self.labels = labels

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        bag = load_bag(self.h5_paths[idx])
        return torch.from_numpy(bag), self.labels[idx]


def load_barcode_to_fold(site_fold_csv: Path, case_id_barcode_map_csv: Path) -> dict:
    """
    Builds barcode -> fold (int) by joining site_stratified_5fold.csv
    (case_id, tumor_origin, site, fold) with case_id_barcode_map.csv
    (case_id, barcode) -- both from make_splits.py.

    This replaces the old single train_val_test_split.csv-based approach:
    mutClassifier.py's REAL, reported results come from nested 5-fold
    site-stratified CV, not a single fixed split -- so for MIL results to
    be genuinely comparable to the mutation classifier's, MIL needs to
    train/evaluate on the SAME 5 folds, not a different splitting scheme
    entirely. This costs ~5x the training time (5 separate trainings per
    architecture instead of 1), same tradeoff as mutClassifier.py's
    nested CV -- worth it for a real comparison.
    """
    fold_df = pd.read_csv(site_fold_csv)
    case_id_barcode_map = pd.read_csv(case_id_barcode_map_csv)
    merged = fold_df.merge(case_id_barcode_map, on="case_id", how="inner")
    return dict(zip(merged["barcode"], merged["fold"]))


def load_case_id_subset(subset_file_path: Path) -> set:
    """
    Loads a set of case_ids to restrict training to, from a file in the
    same "case_id<TAB>..." row format used throughout this project (e.g.
    fuse_tables.py's patient_fused_matrix_rows.txt, where case_id is the
    first field). Used for --patient-subset-file, so MIL can be trained
    on the EXACT SAME patient set as a given mutation-side table -- e.g.
    the fused mutation+signature+CNA table, whose patient count is
    usually smaller than the full cohort (CNA coverage isn't 100%
    identical to mutation coverage), so a direct comparison between MIL
    and mutation-classifier results is only apples-to-apples if both are
    evaluated on the identical patient set.
    """
    lines = Path(subset_file_path).read_text().splitlines()
    return {line.split("\t")[0] for line in lines if line.strip()}


def load_barcode_to_tumor_origin(patient_cohort_csv: Path, case_id_barcode_map_csv: Path,
                                  case_id_subset: set = None) -> dict:
    """
    Builds barcode -> tumor_origin by joining two files:
      - patient_cohort.csv (case_id, tumor_origin) -- the SAME label source
        every other script in the project reads from, so relabeling it
        (e.g. switching to organ-level labels) automatically applies here
        too, without any code change in this file.
      - case_id_barcode_map.csv (case_id, barcode) -- a small separate
        bridge file (also from make_splits.py) that's the ONLY place
        barcode is kept now. MIL training still needs barcode specifically
        because that's what the .h5 tile embedding files are named by --
        this join reconstructs barcode -> tumor_origin without patient_cohort.csv
        itself needing to carry barcode as a column.

    If case_id_subset is given, only patients in that set are kept --
    see load_case_id_subset() / --patient-subset-file.
    """
    patient_cohort = pd.read_csv(patient_cohort_csv)
    case_id_barcode_map = pd.read_csv(case_id_barcode_map_csv)
    merged = patient_cohort.merge(case_id_barcode_map, on="case_id", how="inner")

    if case_id_subset is not None:
        before = len(merged)
        merged = merged[merged["case_id"].isin(case_id_subset)]
        log.info(f"--patient-subset-file: restricted from {before} to {len(merged)} patients")

    return dict(zip(merged["barcode"], merged["tumor_origin"]))


# ---------------------------------------------------------------------------
# Shared gated attention module (used by both ABMIL and CLAM-SB)
# ---------------------------------------------------------------------------

class GatedAttention(nn.Module):
    """Attn_Net_Gated, as in Ilse et al. / CLAM: two parallel projections
    (tanh-gated by a sigmoid branch) combined into a single attention score
    per tile."""

    def __init__(self, in_dim, attn_dim=128, dropout=0.25):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_U = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_w = nn.Linear(attn_dim, 1)

    def forward(self, h):
        # h: (N, in_dim) -> A: (N, 1) raw attention scores (pre-softmax)
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        return A


# ---------------------------------------------------------------------------
# ABMIL
# ---------------------------------------------------------------------------

class ABMIL(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, attn_dim=128, n_classes=32, dropout=0.25):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention = GatedAttention(hidden_dim, attn_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (N, input_dim)
        h = self.feature_extractor(x)          # (N, hidden_dim)
        A = self.attention(h)                    # (N, 1)
        A = F.softmax(A, dim=0)                   # normalize over tiles
        M = (A * h).sum(dim=0, keepdim=True)      # (1, hidden_dim) slide embedding
        logits = self.classifier(M)               # (1, n_classes)
        return logits, A


# ---------------------------------------------------------------------------
# CLAM-SB (single attention branch + instance-level clustering loss)
# ---------------------------------------------------------------------------

class CLAM_SB(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, attn_dim=128, n_classes=32,
                 dropout=0.25, k_sample=8):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention = GatedAttention(hidden_dim, attn_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        # one small binary instance classifier per class, used only during
        # training for the instance-level clustering loss
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_classes)])
        self.k_sample = k_sample
        self.n_classes = n_classes

    def forward(self, x, label=None):
        h = self.feature_extractor(x)             # (N, hidden_dim)
        A = self.attention(h)                        # (N, 1)
        A_softmax = F.softmax(A, dim=0)
        M = (A_softmax * h).sum(dim=0, keepdim=True)  # (1, hidden_dim)
        logits = self.classifier(M)

        instance_loss = None
        if label is not None and self.training:
            instance_loss = self._instance_loss(h, A.squeeze(-1), int(label))

        return logits, A, instance_loss

    def _instance_loss(self, h, attn_scores, label):
        """Top-K / bottom-K attended tiles -> pseudo-positive/negative
        instances for the TRUE class's own instance classifier."""
        n = h.shape[0]
        k = min(self.k_sample, n // 2) if n >= 2 else 0
        if k == 0:
            return None

        top_idx = torch.topk(attn_scores, k).indices
        bottom_idx = torch.topk(-attn_scores, k).indices

        pos_instances = h[top_idx]
        neg_instances = h[bottom_idx]

        instance_clf = self.instance_classifiers[label]
        pos_logits = instance_clf(pos_instances)
        neg_logits = instance_clf(neg_instances)

        pos_targets = torch.ones(k, dtype=torch.long, device=h.device)
        neg_targets = torch.zeros(k, dtype=torch.long, device=h.device)

        loss = F.cross_entropy(pos_logits, pos_targets) + F.cross_entropy(neg_logits, neg_targets)
        return loss / 2


# ---------------------------------------------------------------------------
# TransMIL-style (class token + PPEG + 2 transformer layers, full attention)
# ---------------------------------------------------------------------------

class PPEG(nn.Module):
    """Pyramid Position Encoding Generator: injects positional info via a
    few depthwise convolutions over a roughly-square reshaping of the tile
    sequence, added back as a residual."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x):
        # x: (1, N+1, dim) with x[:, 0] the class token
        B, total_len, C = x.shape
        cls_token, tokens = x[:, :1], x[:, 1:]
        n = tokens.shape[1]
        side = int(np.ceil(np.sqrt(n)))
        padded = side * side - n
        if padded > 0:
            tokens = torch.cat([tokens, tokens[:, :padded]], dim=1)  # pad by repeating

        grid = tokens.transpose(1, 2).view(B, C, side, side)
        grid = grid + self.proj(grid) + self.proj1(grid) + self.proj2(grid)
        tokens = grid.flatten(2).transpose(1, 2)[:, :n]

        return torch.cat([cls_token, tokens], dim=1)


class TransLayer(nn.Module):
    """
    Functionally equivalent self-attention to nn.MultiheadAttention (same
    combined QKV projection -> scaled dot-product attention -> output
    projection pattern it uses internally), but calls
    F.scaled_dot_product_attention directly instead of the nn.Module.
    This matters specifically because F.scaled_dot_product_attention
    (PyTorch 2.0+) automatically dispatches to a memory-efficient/
    flash-attention backend when available (e.g. on an H100) -- it
    computes the EXACT same result as standard attention (not an
    approximation, not a subsample), but never materializes the full
    N x N attention matrix in memory at once. This is what lets TransMIL
    process a full, untruncated bag (e.g. 100,000+ tiles) without the
    OOM a naive full-attention implementation would hit, while still
    using every single tile's information -- no information is discarded.

    NOTE: this only fixes MEMORY, not TIME -- the actual compute is still
    O(N^2), so a forward pass on a very large bag will still be slow,
    just no longer at serious risk of crashing with an out-of-memory error.
    """

    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.norm = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        # x: (1, N, dim)
        B, N, C = x.shape
        normed = self.norm(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, N, head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.out_proj(attn_out)
        return x + attn_out


class TransMIL(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, n_classes=32, n_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layer1 = TransLayer(hidden_dim, n_heads, dropout)
        self.ppeg = PPEG(hidden_dim)
        self.layer2 = TransLayer(hidden_dim, n_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (N, input_dim) -> add batch dim of 1
        h = self.proj(x).unsqueeze(0)              # (1, N, hidden_dim)
        h = torch.cat([self.cls_token, h], dim=1)     # (1, N+1, hidden_dim)
        h = self.layer1(h)
        h = self.ppeg(h)
        h = self.layer2(h)
        h = self.norm(h)
        cls_out = h[:, 0]                              # (1, hidden_dim)
        logits = self.classifier(cls_out)
        return logits, None


# ---------------------------------------------------------------------------
# Training / evaluation loop (shared across all three models)
# ---------------------------------------------------------------------------

def compute_ranking_metrics(y_true, logits, preds, n_classes, k_values=(1, 3, 5)):
    """
    Computes top-1/top-3/top-5 accuracy and weighted F1 from raw logits.
    Simpler than the equivalent in mutClassifier.py -- these MIL models'
    output layer is always n_classes wide regardless of which classes
    appeared during training, so there's no class-subset remapping needed
    (unlike sklearn models, which only learn classes present in y_train).
    """
    logits = np.asarray(logits)
    y_true = np.asarray(y_true)
    results = {}
    ranked = np.argsort(-logits, axis=1)
    for k in k_values:
        topk = ranked[:, :k]
        correct = (topk == y_true[:, None]).any(axis=1)
        results[f"top{k}_accuracy"] = correct.mean()

    results["weighted_f1"] = f1_score(y_true, preds, labels=np.arange(n_classes),
                                        average="weighted", zero_division=0)
    results["macro_f1"] = f1_score(y_true, preds, labels=np.arange(n_classes),
                                     average="macro", zero_division=0)
    return results


def run_epoch(model, model_name, dataset, indices, device, n_classes, optimizer=None,
              accum_steps=1, instance_loss_weight=0.3, max_tiles_transmil=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels, all_logits = 0.0, [], [], []

    if training:
        np.random.shuffle(indices)
        optimizer.zero_grad()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, idx in enumerate(indices):
            bag, label = dataset[idx]
            bag = bag.to(device)
            label_t = torch.tensor([label], device=device)

            if model_name == "transmil" and max_tiles_transmil and bag.shape[0] > max_tiles_transmil:
                sel = np.random.choice(bag.shape[0], max_tiles_transmil, replace=False)
                bag = bag[sel]

            if model_name == "clam":
                logits, _, instance_loss = model(bag, label=label if training else None)
                loss = F.cross_entropy(logits, label_t)
                if instance_loss is not None:
                    loss = loss + instance_loss_weight * instance_loss
            else:
                logits, _ = model(bag)
                loss = F.cross_entropy(logits, label_t)

            if training:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).item())
            all_labels.append(label)
            all_logits.append(logits.detach().cpu().numpy().reshape(-1))

        if training and len(indices) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(indices)
    metrics = compute_ranking_metrics(all_labels, np.stack(all_logits), all_preds, n_classes)
    return avg_loss, metrics, all_preds, all_labels


def build_model(model_name, n_classes, hidden_dim, input_dim=1536):
    if model_name == "abmil":
        return ABMIL(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    elif model_name == "clam":
        return CLAM_SB(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    elif model_name == "transmil":
        return TransMIL(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    raise ValueError(f"Unknown model: {model_name}")



# reduced to 8 configs (from 18) given real time constraints -- keeps the
# core comparisons: known-good hidden_dim (256) vs one step down (128),
# known-good lr (1e-4) vs one step down (5e-5), known-good weight_decay
# (1e-5) vs one step up (5e-5). Dropped hidden_dim=384 and lr=5e-4 to cut
# runtime roughly in half; the fixed baseline (256/1e-4/1e-5) is still
# directly present in this grid as one of the 8 combinations tested.
GRID_HIDDEN_DIMS = [128, 256]
GRID_LRS = [5e-5, 1e-4]
GRID_WEIGHT_DECAYS = [1e-5, 5e-5]


def grid_search_mil(model_name, dataset, idx_train_full, y, n_classes, device, args, output_dir, fold_idx):
    """
    Exhaustive grid search over hidden_dim x lr x weight_decay -- replaces
    the earlier Optuna-based search entirely. Each config is trained for
    --tuning-epochs epochs on an INNER train/val split carved from
    idx_train_full (this outer fold's training pool only -- the outer
    test fold is never touched, same nested-CV principle as before).
    Selection criterion is the best plain VALIDATION ACCURACY reached
    during those epochs -- same convention as mutClassifier.py.

    Unlike Optuna, EVERY config's result is known and saved (not just the
    winner) -- this is a real, disclosable "grid search" for a methods
    section, and the full results are directly reproducible from the
    saved grid file, not hidden inside an opaque study database.

    RESUMABLE AT THE PER-CONFIG LEVEL: each config's result is written to
    <model>_fold<N>_grid_search.json immediately after it finishes, not
    just once the whole grid is done. If the process gets interrupted
    mid-grid-search (e.g. a pod running out of funds and shutting down),
    rerunning the same command re-loads whatever configs already
    finished and only computes the remaining ones -- instead of
    restarting all of them from config 1. This does NOT change behavior
    for already-COMPLETED folds: those are skipped entirely before this
    function is ever called (see the fold-level check in main()) -- this
    only helps a fold that was mid-grid-search when interrupted.

    Returns (best_params, best_val_acc, all_grid_results, total_tuning_time).
    total_tuning_time sums every config's own recorded time (including
    ones resumed from a previous, now-dead process), since a simple
    time.time()-based timer in THIS call would miss time spent before
    the interruption.
    """
    import itertools

    try:
        idx_tr, idx_val = train_test_split(
            idx_train_full, test_size=0.15, stratify=y[idx_train_full], random_state=0
        )
    except ValueError:
        idx_tr, idx_val = train_test_split(idx_train_full, test_size=0.15, random_state=0)

    grid = list(itertools.product(GRID_HIDDEN_DIMS, GRID_LRS, GRID_WEIGHT_DECAYS))
    log.info(f"  [{model_name}] grid search: {len(grid)} configs x {args.tuning_epochs} epochs "
             f"(train={len(idx_tr)}, val={len(idx_val)})")

    grid_path = output_dir / f"{model_name}_fold{fold_idx}_grid_search.json"
    all_results = []
    completed_keys = set()
    if grid_path.exists():
        all_results = json.loads(grid_path.read_text())
        completed_keys = {(r["hidden_dim"], r["lr"], r["weight_decay"]) for r in all_results}
        if completed_keys:
            log.info(f"  [{model_name}] {fold_idx}: resuming grid search -- "
                     f"{len(completed_keys)}/{len(grid)} configs already done from a previous run")

    for i, (hidden_dim, lr, weight_decay) in enumerate(grid):
        if (hidden_dim, lr, weight_decay) in completed_keys:
            log.info(f"  [{model_name}] grid {i + 1}/{len(grid)}: already done -- skipping "
                     f"(hidden_dim={hidden_dim}, lr={lr:.2e}, weight_decay={weight_decay:.2e})")
            continue

        config_start = time.time()
        log.info(f"  [{model_name}] grid {i + 1}/{len(grid)}: hidden_dim={hidden_dim}, "
                 f"lr={lr:.2e}, weight_decay={weight_decay:.2e}")
        model = build_model(model_name, n_classes, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        config_best_val_acc = 0.0
        for epoch in range(args.tuning_epochs):
            epoch_start = time.time()
            run_epoch(
                model, model_name, dataset, idx_tr.copy(), device, n_classes, optimizer,
                accum_steps=args.accum_steps, instance_loss_weight=args.instance_loss_weight,
                max_tiles_transmil=args.max_tiles_transmil,
            )
            _, val_metrics, _, _ = run_epoch(
                model, model_name, dataset, idx_val.copy(), device, n_classes, optimizer=None,
                max_tiles_transmil=args.max_tiles_transmil,
            )
            config_best_val_acc = max(config_best_val_acc, val_metrics["top1_accuracy"])
            log.info(f"    grid {i + 1} epoch {epoch + 1}/{args.tuning_epochs}: "
                     f"val_top1={val_metrics['top1_accuracy']:.4f} ({time.time() - epoch_start:.0f}s)")

        config_elapsed = time.time() - config_start
        log.info(f"  [{model_name}] grid {i + 1} done: best_val_acc={config_best_val_acc:.4f} "
                 f"({config_elapsed:.0f}s total)")

        all_results.append({
            "hidden_dim": hidden_dim, "lr": lr, "weight_decay": weight_decay,
            "val_top1_accuracy": config_best_val_acc, "time_seconds": config_elapsed,
        })

        # save immediately after EACH config, not just once the whole grid
        # finishes -- this is what makes a mid-grid-search interruption
        # (e.g. a pod shutdown) resumable at the config level
        with open(grid_path, "w") as f:
            json.dump(all_results, f, indent=2)

    best_result = max(all_results, key=lambda r: r["val_top1_accuracy"])
    best_params = {
        "hidden_dim": best_result["hidden_dim"], "lr": best_result["lr"],
        "weight_decay": best_result["weight_decay"],
    }
    best_val_acc = best_result["val_top1_accuracy"]
    total_tuning_time = sum(r["time_seconds"] for r in all_results)

    return best_params, best_val_acc, all_results, total_tuning_time


def select_final_hyperparameters(output_dir, model_name, n_folds):
    """
    Chooses hyperparameters for the final, full-dataset model by
    averaging each grid config's val_top1_accuracy ACROSS EVERY
    already-completed fold's saved grid_search.json for this model,
    then picking whichever config has the highest average. This uses
    all n_folds x n_configs data points (e.g. 5 x 8 = 40), not just
    "whichever config won most often" -- a config that won one fold by
    luck but did poorly elsewhere won't get picked over one that was
    consistently strong. Avoids the cost of re-running a whole fresh
    grid search on the full dataset when this same grid was already
    evaluated 5 times over via the CV folds.

    Returns (best_params, full_ranking) where full_ranking is every
    config's average, saved alongside the final model for transparency.
    """
    from collections import defaultdict

    config_scores = defaultdict(list)
    for fold_idx in range(n_folds):
        grid_path = output_dir / f"{model_name}_fold{fold_idx}_grid_search.json"
        if not grid_path.exists():
            raise FileNotFoundError(
                f"{grid_path} not found -- --final-model needs a completed 5-fold grid-search run "
                f"for {model_name} in this --output-dir first, so hyperparameters can be chosen "
                f"from real cross-validated evidence rather than guessed. Alternatively, pass "
                f"--final-hidden-dim/--final-lr/--final-weight-decay to skip this requirement."
            )
        grid_results = json.loads(grid_path.read_text())
        for r in grid_results:
            key = (r["hidden_dim"], r["lr"], r["weight_decay"])
            config_scores[key].append(r["val_top1_accuracy"])

    ranking = []
    for (hidden_dim, lr, weight_decay), scores in config_scores.items():
        ranking.append({
            "hidden_dim": hidden_dim, "lr": lr, "weight_decay": weight_decay,
            "mean_val_top1_accuracy": float(np.mean(scores)), "std_val_top1_accuracy": float(np.std(scores)),
            "n_folds_seen": len(scores),
        })
    ranking.sort(key=lambda r: r["mean_val_top1_accuracy"], reverse=True)

    best = ranking[0]
    best_params = {"hidden_dim": best["hidden_dim"], "lr": best["lr"], "weight_decay": best["weight_decay"]}
    log.info(f"[{model_name}] final hyperparameters chosen by aggregating {n_folds}-fold grid search "
             f"results: {best_params} (mean val_top1_accuracy={best['mean_val_top1_accuracy']:.4f} "
             f"+/-{best['std_val_top1_accuracy']:.4f} across all {n_folds} folds)")
    return best_params, ranking


def run_final_model_training(args, model_name, dataset, y, n_classes, device, output_dir):
    """
    Trains ONE model on the FULL dataset (every patient), with
    hyperparameters chosen by DEFAULT from the already-completed 5-fold
    grid search (see select_final_hyperparameters) -- aggregating each
    config's validation accuracy across all 5 folds, rather than
    re-running a whole fresh grid search on the full dataset (the same
    8 configs were already evaluated 5 times over via the CV folds, so
    that evidence is reused rather than duplicated).

    Pass --final-hidden-dim/--final-lr/--final-weight-decay to instead
    skip this entirely and use a manually-specified fixed config.

    After hyperparameters are set, the actual final model is trained on
    ALL patients, with a small internal split carved out purely for
    early stopping (not a test set, just enough held-back data to know
    when to stop training).

    Saves <model>_final_best.pt, <model>_final_params.json (chosen
    hyperparameters + how they were chosen), and <model>_final_history.csv
    (per-epoch train/val metrics).
    """
    final_model_path = output_dir / f"{model_name}_final_best.pt"
    final_params_path = output_dir / f"{model_name}_final_params.json"
    if final_model_path.exists() and final_params_path.exists():
        log.info(f"[{model_name}] final model already complete -- skipping "
                 f"(delete {final_model_path.name} to force a rerun)")
        return

    all_idx = np.arange(len(y))

    if args.final_hidden_dim is not None:
        if args.final_lr is None or args.final_weight_decay is None:
            raise ValueError("--final-hidden-dim requires --final-lr and --final-weight-decay too")
        best_params = {"hidden_dim": args.final_hidden_dim, "lr": args.final_lr,
                        "weight_decay": args.final_weight_decay}
        ranking = None
        log.info(f"[{model_name}] final model: using MANUALLY-SPECIFIED fixed hyperparameters "
                 f"{best_params} (--final-hidden-dim/--final-lr/--final-weight-decay)")
    else:
        best_params, ranking = select_final_hyperparameters(output_dir, model_name, args.n_folds)

    try:
        idx_train, idx_val = train_test_split(all_idx, test_size=0.1, stratify=y, random_state=0)
    except ValueError:
        idx_train, idx_val = train_test_split(all_idx, test_size=0.1, random_state=0)

    log.info(f"[{model_name}] final model: train={len(idx_train)}, val={len(idx_val)} "
             f"(full dataset, no held-out test fold)")

    model = build_model(model_name, n_classes, best_params["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

    best_val_metric, best_state, epochs_no_improve = -1, None, 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_metrics, _, _ = run_epoch(
            model, model_name, dataset, idx_train.copy(), device, n_classes, optimizer,
            accum_steps=args.accum_steps, instance_loss_weight=args.instance_loss_weight,
            max_tiles_transmil=args.max_tiles_transmil,
        )
        val_loss, val_metrics, _, _ = run_epoch(
            model, model_name, dataset, idx_val.copy(), device, n_classes, optimizer=None,
            max_tiles_transmil=args.max_tiles_transmil,
        )
        log.info(f"[{model_name}] final epoch {epoch + 1}/{args.epochs} "
                 f"train_loss={train_loss:.4f} train_top1={train_metrics['top1_accuracy']:.4f} "
                 f"val_loss={val_loss:.4f} val_top1={val_metrics['top1_accuracy']:.4f}")
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["top1_accuracy"] > best_val_metric:
            best_val_metric = val_metrics["top1_accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log.info(f"[{model_name}] final model: early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), final_model_path)

    pd.DataFrame(history).to_csv(output_dir / f"{model_name}_final_history.csv", index=False)
    with open(final_params_path, "w") as f:
        json.dump({**best_params, "best_val_top1_accuracy": best_val_metric,
                   "n_train": len(idx_train), "n_val": len(idx_val), "n_total": len(y),
                   "selected_from_cross_fold_ranking": ranking}, f, indent=2)

    log.info(f"[{model_name}] final model saved to {final_model_path} "
             f"(best val_top1_accuracy={best_val_metric:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train ABMIL / CLAM-SB / TransMIL on tile embeddings")
    parser.add_argument("--embeddings-dir", required=True, help="Folder of <barcode>.h5 tile embedding files")
    parser.add_argument("--patient-cohort-csv", required=True,
                         help="Path to patient_cohort.csv from make_splits.py (case_id, tumor_origin, "
                              "site). Same label source of truth every other script in the project "
                              "reads from.")
    parser.add_argument("--case-id-barcode-map", required=True,
                         help="Path to case_id_barcode_map.csv from make_splits.py -- the small "
                              "bridge file that's the only place barcode is kept now. Needed because "
                              "MIL locates its .h5 files by barcode, but patient_cohort.csv itself no "
                              "longer carries that column.")
    parser.add_argument("--patient-subset-file", default=None,
                         help="Optional: restrict training to only the patients in this file (e.g. "
                              "/workspace/mutation_tables/fused/patient_fused_matrix_rows.txt). Use "
                              "this to make MIL's patient set match a mutation-side table exactly, "
                              "for a clean apples-to-apples comparison -- e.g. the fused table's "
                              "patient count is usually smaller than the full cohort (CNA coverage "
                              "isn't 100% identical to mutation coverage), so without this MIL trains "
                              "on a larger, different patient set than mutClassifier.py does.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["abmil", "clam", "transmil"],
                         choices=["abmil", "clam", "transmil"])
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs for FINAL per-fold training")
    parser.add_argument("--tuning-epochs", type=int, default=3,
                         help="Epochs per grid-search config during tuning -- deliberately SHORTER than "
                              "--epochs (the final training length), since a full-length run for all "
                              "18 grid configs would be prohibitively expensive. The final model, once "
                              "hyperparameters are locked in, still trains for the full --epochs.")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps (bags per update)")
    parser.add_argument("--instance-loss-weight", type=float, default=0.3, help="CLAM instance loss weight")
    parser.add_argument("--max-tiles-transmil", type=int, default=4000,
                         help="Randomly subsample bags larger than this, for TransMIL only "
                              "(full attention is O(N^2)); set 0 to disable subsampling entirely. "
                              "Default 4000 keeps memory usage bounded on very large bags -- "
                              "TransLayer now uses F.scaled_dot_product_attention (memory-efficient, "
                              "mathematically exact) rather than nn.MultiheadAttention, but this cap "
                              "is still the safer default given how large some slides' bags can get.")
    parser.add_argument("--site-fold-csv", required=True,
                         help="Path to site_stratified_5fold.csv from make_splits.py -- the SAME "
                              "5-fold site-stratified CV assignment mutClassifier.py is evaluated "
                              "against. Runs full 5-fold CV here too (not a single train/val/test "
                              "split), so MIL and the mutation classifier are trained/evaluated on "
                              "IDENTICAL folds -- a genuine apples-to-apples comparison, at the cost "
                              "of ~5x the training time (5 separate trainings per architecture).")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (must match the fold file)")
    parser.add_argument("--min-class-size", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--final-model", action="store_true",
                         help="Train ONE final model per --models entry on the FULL dataset (all "
                              "patients, no held-out test fold), instead of the 5-fold nested-CV "
                              "loop. This is the model you'd actually deploy -- the 5-fold CV's job "
                              "was to evaluate the approach, not to produce 5 separate deployable "
                              "models (each fold model only ever saw ~80% of the data). By default, "
                              "hyperparameters are chosen by aggregating each grid config's "
                              "val_top1_accuracy across the 5 already-completed CV folds' "
                              "grid_search.json files (--output-dir must already have those) and "
                              "picking the config with the best mean -- reusing that evidence rather "
                              "than re-running the same 8-config search on the full dataset. Pass "
                              "--final-hidden-dim/--final-lr/--final-weight-decay instead to skip "
                              "this and use a manually-specified fixed config.")
    parser.add_argument("--final-hidden-dim", type=int, default=None,
                         help="Skip tuning entirely for --final-model and use this fixed hidden_dim "
                              "(must also pass --final-lr and --final-weight-decay). Use this when "
                              "you already have strong evidence for which config wins -- e.g. "
                              "aggregated across the 5 CV folds' own grid-search results -- rather "
                              "than running a whole new fresh grid search on the full dataset.")
    parser.add_argument("--final-lr", type=float, default=None)
    parser.add_argument("--final-weight-decay", type=float, default=None)
    args = parser.parse_args()

    if args.max_tiles_transmil == 0:
        args.max_tiles_transmil = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # full run manifest -- every CLI arg, device, and timestamp, saved
    # once per invocation, for exact reproducibility later (e.g. writing
    # up methods for a paper): this is the complete record of exactly
    # how this run was configured, independent of any per-fold detail
    import datetime
    run_manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": device,
        "args": vars(args),
    }
    manifest_path = output_dir / f"run_manifest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2, default=str)
    log.info(f"Saved run manifest to {manifest_path}")

    case_id_subset = None
    if args.patient_subset_file:
        case_id_subset = load_case_id_subset(Path(args.patient_subset_file))
        log.info(f"Loaded patient subset: {len(case_id_subset)} case_ids from {args.patient_subset_file}")

    log.info("Loading barcode -> tumor_origin mapping from patient_cohort.csv...")
    barcode_to_tumor_origin = load_barcode_to_tumor_origin(
        Path(args.patient_cohort_csv), Path(args.case_id_barcode_map), case_id_subset
    )

    log.info("Loading barcode -> fold assignment from site_stratified_5fold.csv...")
    barcode_to_fold = load_barcode_to_fold(
        Path(args.site_fold_csv), Path(args.case_id_barcode_map)
    )

    embeddings_dir = Path(args.embeddings_dir)
    h5_paths, tumor_origins, barcodes = [], [], []
    for h5_path in sorted(embeddings_dir.glob("*.h5")):
        barcode = h5_path.stem
        tumor_origin = barcode_to_tumor_origin.get(barcode)
        if tumor_origin is not None and barcode in barcode_to_fold:
            h5_paths.append(h5_path)
            tumor_origins.append(tumor_origin)
            barcodes.append(barcode)

    n_no_fold = sum(
        1 for h5_path in sorted(embeddings_dir.glob("*.h5"))
        if barcode_to_tumor_origin.get(h5_path.stem) is not None
        and h5_path.stem not in barcode_to_fold
    )
    if n_no_fold > 0:
        log.warning(f"{n_no_fold} slides have a tumor_origin but no fold assignment in "
                     f"site_stratified_5fold.csv -- excluded (likely a patient dropped by "
                     f"make_splits.py's --min-class-size filter)")

    log.info(f"Found {len(h5_paths)} slides with a known tumor origin and fold assignment")

    class_counts = pd.Series(tumor_origins).value_counts()
    small_classes = class_counts[class_counts < args.min_class_size].index.tolist()
    if small_classes:
        log.warning(f"Dropping {len(small_classes)} tumor origin(s) with fewer than "
                     f"{args.min_class_size} slides: {small_classes}")
        keep_mask = ~pd.Series(tumor_origins).isin(small_classes).to_numpy()
        h5_paths = [p for p, keep in zip(h5_paths, keep_mask) if keep]
        tumor_origins = [c for c, keep in zip(tumor_origins, keep_mask) if keep]
        barcodes = [b for b, keep in zip(barcodes, keep_mask) if keep]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tumor_origins)
    n_classes = len(label_encoder.classes_)
    log.info(f"Final dataset: {len(h5_paths)} slides, {n_classes} tumor origins, {args.n_folds} CV folds")

    folds = np.array([barcode_to_fold[b] for b in barcodes])
    dataset = BagDataset(h5_paths, y)

    if args.final_model:
        for model_name in args.models:
            run_final_model_training(args, model_name, dataset, y, n_classes, device, output_dir)
        log.info("Done (final-model mode).")
        return

    results = []
    for model_name in args.models:
        fold_results = []
        best_overall_metric = -1
        best_overall_state = None
        best_overall_fold = None

        for fold_idx in range(args.n_folds):
            fold_model_path = output_dir / f"{model_name}_fold{fold_idx}_best.pt"
            fold_report_path = output_dir / f"{model_name}_fold{fold_idx}_classification_report.txt"
            fold_params_path = output_dir / f"{model_name}_fold{fold_idx}_best_params.json"

            if fold_model_path.exists() and fold_report_path.exists() and fold_params_path.exists():
                log.info(f"[{model_name}] fold {fold_idx + 1}/{args.n_folds}: already complete -- "
                         f"skipping (delete this fold's files to force a rerun)")
                report_text = fold_report_path.read_text()
                metrics = {}
                for line in report_text.splitlines():
                    for key, label in [("top1_accuracy", "Test top-1 accuracy"),
                                        ("top3_accuracy", "Test top-3 accuracy"),
                                        ("top5_accuracy", "Test top-5 accuracy"),
                                        ("weighted_f1", "Test weighted F1"),
                                        ("macro_f1", "Test macro F1")]:
                        if line.startswith(f"{label}:"):
                            metrics[key] = float(line.split(":")[1].strip())
                saved_params = json.loads(fold_params_path.read_text())
                extra = {
                    "tuning_val_accuracy": saved_params.get("tuning_val_accuracy"),
                    "tuning_time_seconds": saved_params.get("tuning_time_seconds"),
                    **{f"param_{k}": v for k, v in saved_params.items()
                       if k not in ("tuning_val_accuracy", "n_trials", "tuning_epochs", "tuning_time_seconds")},
                }
                fold_results.append({"fold": fold_idx, **metrics, **extra})
                if metrics.get("weighted_f1", -1) > best_overall_metric:
                    best_overall_metric = metrics["weighted_f1"]
                    best_overall_fold = fold_idx
                continue

            log.info(f"=== [{model_name}] fold {fold_idx + 1}/{args.n_folds} ===")

            idx_test = np.where(folds == fold_idx)[0]
            idx_train_full = np.where(folds != fold_idx)[0]

            # --- INNER grid search: pool is ONLY this fold's training data --
            # the outer test fold above is never passed in here, so it
            # can't influence hyperparameter selection for this fold
            # (same nested-CV principle as mutClassifier.py). Resumable
            # at the per-config level -- see grid_search_mil's docstring. ---
            best_params, tuning_val_acc, grid_results, tuning_elapsed = grid_search_mil(
                model_name, dataset, idx_train_full, y, n_classes, device, args, output_dir, fold_idx
            )
            log.info(f"[{model_name}] fold {fold_idx + 1}: grid search done ({tuning_elapsed:.0f}s total "
                     f"across all configs) -- best validation accuracy {tuning_val_acc:.4f}, "
                     f"params: {best_params}")

            with open(fold_params_path, "w") as f:
                json.dump({**best_params, "tuning_val_accuracy": tuning_val_acc,
                           "n_grid_configs": len(grid_results), "tuning_epochs": args.tuning_epochs,
                           "tuning_time_seconds": tuning_elapsed}, f, indent=2)

            # small internal validation slice carved out of this fold's
            # training pool, purely for early stopping -- idx_test (the
            # held-out outer fold) is never touched until final evaluation
            try:
                idx_train, idx_val = train_test_split(
                    idx_train_full, test_size=0.1, stratify=y[idx_train_full], random_state=0
                )
            except ValueError:
                # a class too small to stratify the 90/10 split -- fall
                # back to a non-stratified split rather than crashing
                idx_train, idx_val = train_test_split(idx_train_full, test_size=0.1, random_state=0)

            log.info(f"[{model_name}] fold {fold_idx + 1}: train={len(idx_train)}, "
                     f"val={len(idx_val)}, test={len(idx_test)}")

            training_start = time.time()
            model = build_model(model_name, n_classes, best_params["hidden_dim"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"],
                                          weight_decay=best_params["weight_decay"])

            best_val_metric, best_state, epochs_no_improve = -1, None, 0
            history = []

            for epoch in range(args.epochs):
                train_loss, train_metrics, _, _ = run_epoch(
                    model, model_name, dataset, idx_train.copy(), device, n_classes, optimizer,
                    accum_steps=args.accum_steps, instance_loss_weight=args.instance_loss_weight,
                    max_tiles_transmil=args.max_tiles_transmil,
                )
                val_loss, val_metrics, _, _ = run_epoch(
                    model, model_name, dataset, idx_val.copy(), device, n_classes, optimizer=None,
                    max_tiles_transmil=args.max_tiles_transmil,
                )
                log.info(f"[{model_name}] fold {fold_idx + 1} epoch {epoch+1}/{args.epochs} "
                         f"train_loss={train_loss:.4f} train_top1={train_metrics['top1_accuracy']:.4f} "
                         f"val_loss={val_loss:.4f} val_top1={val_metrics['top1_accuracy']:.4f} "
                         f"val_weighted_F1={val_metrics['weighted_f1']:.4f}")
                history.append({
                    "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })

                # early stopping / best-epoch restoration criterion is
                # VALIDATION ACCURACY (top-1) -- same convention as the
                # Optuna tuning objective (make_mil_objective) and as
                # mutClassifier.py's plain validation-accuracy objective,
                # so every validation-based decision across this project
                # uses the same metric consistently
                if val_metrics["top1_accuracy"] > best_val_metric:
                    best_val_metric = val_metrics["top1_accuracy"]
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience:
                        log.info(f"[{model_name}] fold {fold_idx + 1}: early stopping at epoch {epoch+1}")
                        break

            pd.DataFrame(history).to_csv(output_dir / f"{model_name}_fold{fold_idx}_history.csv", index=False)

            if best_state is not None:
                model.load_state_dict(best_state)
            torch.save(model.state_dict(), fold_model_path)

            test_loss, test_metrics, test_preds, test_labels = run_epoch(
                model, model_name, dataset, idx_test.copy(), device, n_classes, optimizer=None,
                max_tiles_transmil=args.max_tiles_transmil,
            )
            log.info(f"[{model_name}] fold {fold_idx + 1} TEST top1={test_metrics['top1_accuracy']:.4f} "
                     f"top3={test_metrics['top3_accuracy']:.4f} top5={test_metrics['top5_accuracy']:.4f} "
                     f"weighted_F1={test_metrics['weighted_f1']:.4f} macro_F1={test_metrics['macro_f1']:.4f}")

            training_elapsed = time.time() - training_start
            n_epochs_run = len(history)
            best_epoch = int(np.argmax([h["val_top1_accuracy"] for h in history])) + 1 if history else None

            report = classification_report(test_labels, test_preds, labels=np.arange(n_classes),
                                            target_names=label_encoder.classes_, zero_division=0)
            with open(fold_report_path, "w") as f:
                f.write(f"Model: {model_name}, fold: {fold_idx}\n")
                f.write(f"Hyperparameters (grid search on outer-train folds only, nested CV): {best_params}\n")
                f.write(f"Grid search: {len(GRID_HIDDEN_DIMS) * len(GRID_LRS) * len(GRID_WEIGHT_DECAYS)} configs "
                        f"x {args.tuning_epochs} epochs, best tuning validation accuracy {tuning_val_acc:.4f}, "
                        f"tuning time {tuning_elapsed:.0f}s (full grid saved in "
                        f"{model_name}_fold{fold_idx}_grid_search.json)\n")
                f.write(f"Final training: {n_epochs_run}/{args.epochs} epochs run "
                        f"(best epoch: {best_epoch}, patience: {args.patience}), "
                        f"training time {training_elapsed:.0f}s\n")
                f.write(f"Train/val/test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}\n\n")
                f.write(f"Test top-1 accuracy: {test_metrics['top1_accuracy']:.4f}\n")
                f.write(f"Test top-3 accuracy: {test_metrics['top3_accuracy']:.4f}\n")
                f.write(f"Test top-5 accuracy: {test_metrics['top5_accuracy']:.4f}\n")
                f.write(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}\n")
                f.write(f"Test macro F1: {test_metrics['macro_f1']:.4f}\n\n")
                f.write(report)

            fold_results.append({
                "fold": fold_idx, **test_metrics,
                "tuning_val_accuracy": tuning_val_acc,
                "n_epochs_run": n_epochs_run, "best_epoch": best_epoch,
                "tuning_time_seconds": tuning_elapsed, "training_time_seconds": training_elapsed,
                **{f"param_{k}": v for k, v in best_params.items()},
            })

            if test_metrics["weighted_f1"] > best_overall_metric:
                best_overall_metric = test_metrics["weighted_f1"]
                best_overall_state = model.state_dict()
                best_overall_fold = fold_idx

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(output_dir / f"{model_name}_cv_folds.csv", index=False)

        summary = {}
        for metric_name in ["top1_accuracy", "top3_accuracy", "top5_accuracy", "weighted_f1", "macro_f1"]:
            summary[f"{metric_name}_mean"] = fold_df[metric_name].mean()
            summary[f"{metric_name}_std"] = fold_df[metric_name].std()

        log.info(f"[{model_name}] CV RESULT: "
                 f"top1={summary['top1_accuracy_mean']:.4f}+/-{summary['top1_accuracy_std']:.4f}, "
                 f"weighted_F1={summary['weighted_f1_mean']:.4f}+/-{summary['weighted_f1_std']:.4f} "
                 f"(best fold: {best_overall_fold})")

        if best_overall_state is not None:
            torch.save(best_overall_state, output_dir / f"{model_name}_best.pt")

        results.append({"model": model_name, **summary, "best_fold": best_overall_fold})

    results_df = pd.DataFrame(results).sort_values("weighted_f1_mean", ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    log.info("\n" + results_df.to_string(index=False))

    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
    log.info("Done.")


if __name__ == "__main__":
    main()