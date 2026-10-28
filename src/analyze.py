"""
Analysis and figures for the dependency-confounder study.

Usage:  python3 src/analyze.py [--tag main]

Reads results/<tag>_pergene.csv, writes results/*.csv tables and figures/*.png.
Every aggregate is a median over genes with a 2000-sample percentile bootstrap CI.
The historical CSV column `r2` denotes S_mean against a training-mean baseline.
A lineage-only model cannot learn unseen categories; no confounder-removal claim is made.
Arms are compared PAIRED on identical genes and identical folds.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

ARM_LABEL = {"A_lineage": "A  lineage only", "B_expression": "B  expression",
             "C_copynumber": "C  copy number", "D_mutation": "D  mutation",
             "E_multiomics": "E  multi-omics", "F_multi_plus_lin": "F  multi-omics+lineage"}
SCHEME_LABEL = {"patient_grouped": "S1  patient-grouped CV",
                "lineage_blocked": "S2  lineage-blocked CV"}
STRATA = ["common_essential", "selective", "flat"]
STRAT_LABEL = {"common_essential": "common essential", "selective": "selective", "flat": "flat"}
C_S1, C_S2 = "#4C72B0", "#C44E52"


def load(tag="main"):
    df = pd.read_csv(f"{RES}/{tag}_pergene.csv")
    wide = df.pivot_table(index=["scheme", "gene", "symbol", "stratum"],
                          columns="arm", values="r2").reset_index()
    for a in ARM_LABEL:
        if a in wide.columns and a != "A_lineage":
            wide[f"d_{a}"] = wide[a] - wide["A_lineage"]
    return df, wide


# --------------------------------------------------------------------------
def table_main(df, tag):
    rows = []
    for sc in df.scheme.unique():
        for st in STRATA:
            for arm in ARM_LABEL:
                v = df[(df.scheme == sc) & (df.stratum == st) & (df.arm == arm)].r2
                if len(v) == 0:
                    continue
                m, lo, hi = bootstrap_ci(v.values)
                rows.append({"scheme": sc, "stratum": st, "arm": arm, "n_genes": len(v),
                             "median_r2": m, "ci_lo": lo, "ci_hi": hi,
                             "frac_r2_gt_0": float((v > 0).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/{tag}_table_main.csv", index=False)
    return t


def table_delta(wide, tag):
    """Primary endpoint: paired Delta-R2 of each arm against the lineage-only null."""
    rows = []
    for sc in wide.scheme.unique():
        for st in STRATA:
            sub = wide[(wide.scheme == sc) & (wide.stratum == st)]
            for arm in ARM_LABEL:
                col = f"d_{arm}"
                if col not in sub.columns:
                    continue
                m, lo, hi = bootstrap_ci(sub[col].values)
                rows.append({"scheme": sc, "stratum": st, "arm": arm, "n_genes": len(sub),
                             "median_delta_r2": m, "ci_lo": lo, "ci_hi": hi,
                             "frac_better_than_lineage": float((sub[col] > 0).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/{tag}_table_delta.csv", index=False)
    return t


# --------------------------------------------------------------------------
def fig_strata(df, tag):
    """Fig 1 - common essentiality inflates R2; that is the first confounder."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sc in zip(axes, ["patient_grouped", "lineage_blocked"]):
        data, labels = [], []
        for st in STRATA:
            for arm in ["A_lineage", "E_multiomics"]:
                v = df[(df.scheme == sc) & (df.stratum == st) & (df.arm == arm)].r2.dropna()
                data.append(v.values)
                labels.append(f"{STRAT_LABEL[st]}\n{arm.split('_')[0]}")
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor("#dddddd" if i % 2 == 0 else "#9ecae1")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(SCHEME_LABEL[sc], fontsize=10)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("out-of-fold $S_{mean}$ vs training mean")
    fig.suptitle("Figure 1. Predictability by gene stratum (grey = lineage only, blue = multi-omics)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_strata.png", dpi=180)
    plt.close(fig)


def fig_headline(wide, tag):
    """Fig 2 - the primary endpoint: does multi-omics beat lineage, and does it survive S2?"""
    sel = wide[wide.stratum == "selective"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for sc, c in [("patient_grouped", C_S1), ("lineage_blocked", C_S2)]:
        v = sel[sel.scheme == sc]["d_E_multiomics"].dropna()
        ax.hist(v, bins=45, alpha=0.6, color=c, label=SCHEME_LABEL[sc])
        ax.axvline(np.median(v), color=c, lw=2)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(r"$\Delta S_{mean}$  (multi-omics $-$ lineage only)")
    ax.set_ylabel("genes"); ax.legend(fontsize=8)
    ax.set_title("Primary endpoint, selective genes", fontsize=10)

    ax = axes[1]
    piv = sel.pivot_table(index="gene", columns="scheme", values="E_multiomics")
    piv = piv.dropna()
    ax.scatter(piv["patient_grouped"], piv["lineage_blocked"], s=7, alpha=0.45, color="#555555")
    lim = [min(piv.min()) - 0.05, max(piv.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.axhline(0, color="r", lw=0.8); ax.axvline(0, color="r", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$S_{mean}$ under S1 (patient-grouped)")
    ax.set_ylabel("$S_{mean}$ under S2 (lineage-blocked)")
    ax.set_title("Multi-omics: same genes, two validation schemes", fontsize=10)
    fig.suptitle("Figure 2. Transfer across evaluation schemes", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_headline.png", dpi=180)
    plt.close(fig)


def fig_arms(df, tag):
    """Fig 3 - modality comparison on selective genes, both schemes."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    arms = list(ARM_LABEL)
    w = 0.38
    for i, (sc, c) in enumerate([("patient_grouped", C_S1), ("lineage_blocked", C_S2)]):
        ms, los, his = [], [], []
        for arm in arms:
            v = df[(df.scheme == sc) & (df.stratum == "selective") & (df.arm == arm)].r2
            m, lo, hi = bootstrap_ci(v.values)
            ms.append(m); los.append(m - lo); his.append(hi - m)
        ax.bar(np.arange(len(arms)) + (i - 0.5) * w, ms, w, color=c,
               yerr=[los, his], capsize=3, label=SCHEME_LABEL[sc])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("median $S_{mean}$ (selective genes)")
    ax.legend(fontsize=8)
    ax.set_title("Figure 3. Modality contribution under two validation schemes", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_arms.png", dpi=180)
    plt.close(fig)


def fig_survivors(wide, tag, topn=25):
    """Fig 4 - which genes still beat the lineage null under the hard scheme."""
    sel = wide[(wide.scheme == "lineage_blocked") & (wide.stratum == "selective")]
    top = sel.nlargest(topn, "d_E_multiomics")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(top))
    ax.barh(y, top["d_E_multiomics"], color="#55A868")
    ax.set_yticks(y); ax.set_yticklabels(top["symbol"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta S_{mean}$ over lineage-only null (S2)")
    ax.set_title(f"Figure 4. Top {topn} genes exceeding the lineage-only baseline", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_survivors.png", dpi=180)
    plt.close(fig)
    top[["symbol", "gene", "A_lineage", "E_multiomics", "d_E_multiomics"]].to_csv(
        f"{RES}/{tag}_survivors.csv", index=False)
    return top


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main")
    ap.add_argument("--results-dir", default=None, help="Directory containing per-gene scores")
    ap.add_argument("--figures-dir", default=None, help="Where to write figures")
    a = ap.parse_args()
    RES = a.results_dir or RES
    FIG = a.figures_dir or FIG
    os.makedirs(FIG, exist_ok=True)
    df, wide = load(a.tag)
    tm = table_main(df, a.tag); td = table_delta(wide, a.tag)
    fig_strata(df, a.tag); fig_headline(wide, a.tag)
    fig_arms(df, a.tag); top = fig_survivors(wide, a.tag)

    print("\n=== median S_mean (selective genes) ===")
    print(tm[tm.stratum == "selective"][["scheme", "arm", "median_r2", "ci_lo", "ci_hi"]]
          .to_string(index=False))
    print("\n=== PRIMARY ENDPOINT: delta S_mean vs lineage-only, selective genes ===")
    print(td[(td.stratum == "selective")][
        ["scheme", "arm", "median_delta_r2", "ci_lo", "ci_hi", "frac_better_than_lineage"]]
        .to_string(index=False))
    print("\n=== largest lineage-baseline contrasts (S2) ===")
    print(top[["symbol", "A_lineage", "E_multiomics", "d_E_multiomics"]].head(15)
          .to_string(index=False))
