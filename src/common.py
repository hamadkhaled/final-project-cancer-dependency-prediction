"""
Shared utilities for the CBIO7215 dependency-confounder study.

Everything that both the main experiment and the ablations need lives here:
data loading, gene stratification, cross-validation splitters, a fast ridge
with GCV-selected alpha, and the metric definitions.

All randomness is seeded. See design.md for the frozen protocol.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from scipy import linalg, stats

SEED = 0
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Frozen protocol constants (design.md sections 7, 8, 11)
SELECTIVE_STD_MIN = 0.25      # std of gene effect across the cohort
SELECTIVE_DEP_MIN = 5         # lines with gene effect < DEP_THRESHOLD
DEP_THRESHOLD = -0.5
N_SELECTIVE = 1000            # cap on primary target set
N_STRATUM_SAMPLE = 200        # common-essential and flat comparison samples
N_FOLDS = 5
MIN_LINEAGE_SIZE = 10         # smaller lineages pooled into "Other"
ALPHAS = np.logspace(-1, 5, 25)
TOPK_DEFAULT = 300


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class Dataset:
    """Aligned DepMap cohort. All matrices share the same row order (cell lines).

    log_cn: copy number arrives as a relative ratio with a long right tail
    (median ~1.0, max ~254). The primary analysis uses it as supplied, matching
    how it is usually fed to models; log_cn=True applies log2(x+1) first and is
    run as a sensitivity analysis, because that skew could by itself explain a
    weak copy-number result under correlation-based screening.
    """

    def __init__(self, data_dir: str = DATA, log_cn: bool = False):
        self.Y = np.load(f"{data_dir}/Y.npy")         # dependency  (n, G_all)
        self.expr = np.load(f"{data_dir}/expr.npy")   # log2 TPM+1  (n, p_e)
        self.cn = np.load(f"{data_dir}/cn.npy")       # rel. copy # (n, p_c)
        if log_cn:
            self.cn = np.log2(self.cn + 1.0).astype(np.float32)
        self.mut = np.load(f"{data_dir}/mut.npy").astype(np.float32)
        self.meta = pd.read_csv(f"{data_dir}/meta.csv")
        names = json.load(open(f"{data_dir}/names.json"))
        self.y_genes = np.array(names["y"])
        self.feat_names = np.array(
            list(names["expr"]) + list(names["cn"]) + list(names["mut"])
        )
        self.common_essentials = set(names["common_essentials"])

        # Concatenated feature block + modality index, so a fold's correlation
        # screen is computed once and sliced per arm.
        self.X = np.hstack([self.expr, self.cn, self.mut]).astype(np.float32)
        self.modality = np.concatenate([
            np.full(self.expr.shape[1], 0, dtype=np.int8),   # 0 = expression
            np.full(self.cn.shape[1], 1, dtype=np.int8),     # 1 = copy number
            np.full(self.mut.shape[1], 2, dtype=np.int8),    # 2 = mutation
        ])

        # Lineage one-hot (arm A / F)
        lin = self.meta["OncotreeLineage"].fillna("Unknown").values
        self.lineage = lin
        self.lineage_onehot = pd.get_dummies(lin).values.astype(np.float32)

        # Gene symbol without the entrez suffix, e.g. "A1BG (1)" -> "A1BG"
        self.y_symbol = np.array([g.split(" (")[0] for g in self.y_genes])
        self.feat_symbol = np.array([f.split(" (")[0] for f in self.feat_names])

        self.n = self.Y.shape[0]

    # ---- strata (design.md §7) -------------------------------------------
    def strata(self) -> pd.DataFrame:
        sd = self.Y.std(axis=0)
        ndep = (self.Y < DEP_THRESHOLD).sum(axis=0)
        is_ce = np.array([g in self.common_essentials for g in self.y_genes])
        selective = (~is_ce) & (sd >= SELECTIVE_STD_MIN) & (ndep >= SELECTIVE_DEP_MIN)
        label = np.where(is_ce, "common_essential",
                 np.where(selective, "selective", "flat"))
        return pd.DataFrame({"gene": self.y_genes, "symbol": self.y_symbol,
                             "std": sd, "n_dependent": ndep, "stratum": label})

    def target_set(self):
        """Frozen target selection: top-variance selective genes plus stratum samples."""
        st = self.strata()
        rng = np.random.default_rng(SEED)
        sel = st.index[st.stratum == "selective"].values
        sel = sel[np.argsort(-st.loc[sel, "std"].values)][:N_SELECTIVE]
        ce = st.index[st.stratum == "common_essential"].values
        ce = rng.choice(ce, size=min(N_STRATUM_SAMPLE, len(ce)), replace=False)
        fl = st.index[st.stratum == "flat"].values
        fl = rng.choice(fl, size=min(N_STRATUM_SAMPLE, len(fl)), replace=False)
        idx = np.concatenate([np.sort(sel), np.sort(ce), np.sort(fl)])
        return idx, st.loc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------
# Cross-validation (design.md §8)
# --------------------------------------------------------------------------
def _group_kfold(groups: np.ndarray, n_folds: int = N_FOLDS):
    """Deterministic group k-fold: largest groups assigned to smallest fold first."""
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    fold_of = {}
    sizes = np.zeros(n_folds, dtype=int)
    for g in uniq[order]:
        f = int(np.argmin(sizes))
        fold_of[g] = f
        sizes[f] += int(counts[uniq == g][0])
    assign = np.array([fold_of[g] for g in groups])
    return [(np.where(assign != f)[0], np.where(assign == f)[0]) for f in range(n_folds)]


def splits_patient_grouped(ds: Dataset):
    """S1: deterministic patient-grouped CV; lines from one patient stay together."""
    return _group_kfold(ds.meta["PatientID"].fillna("NA").values)


def splits_lineage_blocked(ds: Dataset):
    """S2 (primary): whole lineages held out; rare lineages pooled into 'Other'."""
    lin = pd.Series(ds.lineage)
    counts = lin.value_counts()
    small = set(counts.index[counts < MIN_LINEAGE_SIZE])
    grp = lin.where(~lin.isin(small), "Other").values
    return _group_kfold(grp)


SCHEMES = {"patient_grouped": splits_patient_grouped,
           "lineage_blocked": splits_lineage_blocked}


# --------------------------------------------------------------------------
# Ridge with GCV-selected alpha
# --------------------------------------------------------------------------
def ridge_gcv(Xtr, ytr, Xte, alphas=ALPHAS):
    """
    Ridge regression with alpha chosen by generalized cross-validation.

    One SVD serves the whole alpha grid, which is what makes 10^5 fits tractable.
    Intercept handled by centering. Returns (test predictions, chosen alpha).
    """
    xm = Xtr.mean(axis=0)
    ym = ytr.mean()
    Xc = Xtr - xm
    yc = ytr - ym
    n = Xc.shape[0]
    U, s, _ = linalg.svd(Xc, full_matrices=False)
    d = s ** 2
    Uty = U.T @ yc
    best, best_gcv, best_filt = alphas[0], np.inf, None
    for a in alphas:
        filt = d / (d + a)                       # shrinkage factors
        resid = yc - U @ (filt * Uty)
        dof = filt.sum()
        denom = (1.0 - dof / n) ** 2
        if denom <= 1e-12:
            continue
        gcv = (resid @ resid) / n / denom
        if gcv < best_gcv:
            best_gcv, best, best_filt = gcv, a, filt
    if best_filt is None:
        best_filt = d / (d + alphas[-1])
        best = alphas[-1]
    # coef = V diag(s/(d+a)) U^T y  ; recover V from the economy SVD
    V = (Xc.T @ U) / np.where(s > 1e-12, s, 1.0)
    coef = V @ (best_filt * Uty / np.where(s > 1e-12, s, 1.0))
    return (Xte - xm) @ coef + ym, best


# --------------------------------------------------------------------------
# Metrics (design.md §9)
# --------------------------------------------------------------------------
def pooled_r2(y_true_parts, y_pred_parts, train_means):
    """
    Out-of-fold predictive skill S_mean against each fold's TRAINING mean.

    S_mean = 1 - total squared error / total squared error of the training-mean
    predictor. This is not standard test-mean R-squared. The test mean minimizes
    squared error on the test fold, so its denominator is no larger than the
    training-mean denominator and its R-squared is no higher for fixed predictions.
    Predicting the fold training mean scores exactly zero; worse predictions
    score negative. The historical function name and CSV column `r2` are retained
    for compatibility; revised prose and plots call this score S_mean.
    """
    ss_res = sum(((yt - yp) ** 2).sum() for yt, yp in zip(y_true_parts, y_pred_parts))
    ss_tot = sum(((yt - m) ** 2).sum() for yt, m in zip(y_true_parts, train_means))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def spearman(y_true, y_pred):
    if np.std(y_pred) < 1e-12:
        return np.nan
    return stats.spearmanr(y_true, y_pred).statistic


def bootstrap_ci(values, stat=np.median, n_boot=2000, seed=SEED, alpha=0.05):
    """Percentile bootstrap over genes."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return stat(v), float(np.percentile(draws, 100 * alpha / 2)), \
           float(np.percentile(draws, 100 * (1 - alpha / 2)))


# --------------------------------------------------------------------------
# Arms (design.md §5)
# --------------------------------------------------------------------------
ARMS = {
    "A_lineage":      {"modalities": [],        "lineage": True},
    "B_expression":   {"modalities": [0],       "lineage": False},
    "C_copynumber":   {"modalities": [1],       "lineage": False},
    "D_mutation":     {"modalities": [2],       "lineage": False},
    "E_multiomics":   {"modalities": [0, 1, 2], "lineage": False},
    "F_multi_plus_lin": {"modalities": [0, 1, 2], "lineage": True},
}
