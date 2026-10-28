"""
Figures for the report and deck.

Written after the main run, because the figures have to serve the findings rather
than the hypothesis. Run:  python3 src/make_figures.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Dataset, SCHEMES, bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"  # Read historical results without changing them.
FIG = f"{ROOT}/work/rerun/figures"
DERIVED = f"{ROOT}/work/rerun/results"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

C_S1, C_S2, C_GREY = "#4C72B0", "#C44E52", "#999999"
ARMS = ["A_lineage", "B_expression", "C_copynumber", "D_mutation",
        "E_multiomics", "F_multi_plus_lin"]
ALAB = {"A_lineage": "A\nlineage", "B_expression": "B\nexpression",
        "C_copynumber": "C\ncopy no.", "D_mutation": "D\nmutation",
        "E_multiomics": "E\nmulti-omics", "F_multi_plus_lin": "F\nmulti+lineage"}


def _load(data_dir=None):
    df = pd.read_csv(f"{RES}/main_pergene.csv")
    ds = Dataset(data_dir=data_dir) if data_dir else Dataset()
    tidx = np.load(f"{RES}/main_target_idx.npy")
    return df, ds, tidx


# ---------------------------------------------------------------- Figure 1
def fig1_metric_inflation(df, ds, tidx):
    """Compare pooled and per-gene correlations on the SAME gene panel.

    The supplied legacy CSV used all targets for pooled correlation and only
    selective targets for its median. Its values cannot isolate aggregation.
    This corrected calculation requires prediction arrays absent from the ZIP.
    """
    Y = ds.Y[:, tidx]
    target_genes = ds.y_genes[tidx]
    info = df[(df.scheme == "lineage_blocked") & (df.arm == "E_multiomics")].set_index("gene").loc[target_genes]
    selmask = (info.stratum == "selective").values
    recs = []
    for sc in ["patient_grouped", "lineage_blocked"]:
        for arm in ["A_lineage", "E_multiomics"]:
            P = np.load(f"{RES}/main_preds_{sc}_{arm}.npy")
            if P.shape != Y.shape:
                raise ValueError(f"{sc}/{arm}: prediction shape {P.shape} differs from target shape {Y.shape}")
            arm_info = df[(df.scheme == sc) & (df.arm == arm)].set_index("gene").loc[target_genes]
            for panel, columns in [("all_targets", np.ones(len(tidx), dtype=bool)), ("selective", selmask)]:
                yp, pp = Y[:, columns], P[:, columns]
                valid = np.isfinite(yp) & np.isfinite(pp)
                pooled = stats.spearmanr(yp[valid], pp[valid]).statistic
                pg = arm_info.loc[target_genes[columns], "spearman"].median()
                recs.append({"scheme": sc, "arm": arm, "gene_panel": panel,
                             "n_genes": int(columns.sum()), "pooled": pooled, "pergene": pg})
    r = pd.DataFrame(recs)
    plot_r = r[r.gene_panel == "selective"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(4); w = 0.37
    lbl = [f"{a.split('_')[0]}\n{'S1' if s=='patient_grouped' else 'S2'}"
           for s, a in zip(plot_r.scheme, plot_r.arm)]
    ax.bar(x - w/2, plot_r.pooled, w, color=C_GREY, label="pooled over selective genes")
    ax.bar(x + w/2, plot_r.pergene, w, color=C_S2, label="per-gene median, same selective genes")
    for xi, (p, g) in enumerate(zip(plot_r.pooled, plot_r.pergene)):
        ax.text(xi - w/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
        ax.text(xi + w/2, g + (0.02 if g > 0 else -0.06), f"{g:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=9)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_ylim(-0.42, 1.22)
    ax.legend(fontsize=8.5, loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Figure 1.  The same predictions and gene panel, scored two ways\n"
                 "A = lineage only, E = multi-omics; S1 = patient-grouped CV, S2 = lineage-blocked CV",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_metric_inflation.png", dpi=190); plt.close(fig)
    r.to_csv(f"{DERIVED}/fig1_metric_inflation_matched_panels.csv", index=False)
    return r


# ---------------------------------------------------------------- Figure 2
def fig2_strata(df):
    """Outside the selective stratum, the omics models are worse than the mean."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    strata = ["common_essential", "selective", "flat"]
    slab = {"common_essential": "common\nessential", "selective": "selective", "flat": "flat"}
    for ax, sc, t in zip(axes, ["patient_grouped", "lineage_blocked"],
                         ["S1  patient-grouped CV", "S2  lineage-blocked CV"]):
        x = np.arange(len(strata)); w = 0.37
        for i, (arm, c) in enumerate([("A_lineage", C_GREY), ("E_multiomics", C_S2)]):
            ms, lo, hi = [], [], []
            for st in strata:
                v = df[(df.scheme == sc) & (df.arm == arm) & (df.stratum == st)].r2.values
                m, l, h = bootstrap_ci(v)
                ms.append(m); lo.append(m - l); hi.append(h - m)
            ax.bar(x + (i - 0.5) * w, ms, w, color=c, yerr=[lo, hi], capsize=3,
                   label="lineage only" if i == 0 else "multi-omics")
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels([slab[s] for s in strata], fontsize=9)
        ax.set_title(t, fontsize=10)
    axes[0].set_ylabel("median $S_{mean}$ vs training mean")
    axes[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Figure 2.  Multi-omics only helps on selective genes — elsewhere it is worse than the mean",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_strata.png", dpi=190); plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3_modalities(df, ds, tidx):
    """Which modality carries the signal, and what the fusion actually selects."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    x = np.arange(len(ARMS)); w = 0.37
    for i, (sc, c, lab) in enumerate([("patient_grouped", C_S1, "S1 patient-grouped"),
                                      ("lineage_blocked", C_S2, "S2 lineage-blocked")]):
        ms, lo, hi = [], [], []
        for arm in ARMS:
            v = df[(df.scheme == sc) & (df.arm == arm) & (df.stratum == "selective")].r2.values
            m, l, h = bootstrap_ci(v)
            ms.append(m); lo.append(m - l); hi.append(h - m)
        ax.bar(x + (i - 0.5) * w, ms, w, color=c, yerr=[lo, hi], capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([ALAB[a] for a in ARMS], fontsize=8.5)
    ax.set_ylabel("median $S_{mean}$ (selective genes)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("Modality contribution", fontsize=10)

    # what the fusion arm actually picks
    comp = json.load(open(f"{DERIVED}/feature_composition.json"))
    ax = axes[1]
    ax.pie([comp["expression"], comp["copy_number"], comp["mutation"]],
           labels=[f"expression\n{comp['expression']:.0f}%",
                   f"copy no.\n{comp['copy_number']:.0f}%",
                   f"mutation\n{comp['mutation']:.1f}%"],
           colors=["#4C72B0", "#DD8452", "#937860"],
           autopct=None, startangle=90, textprops={"fontsize": 9})
    ax.set_title("Arm E top-300 features\nfirst S2 training fold only", fontsize=10)
    fig.suptitle("Figure 3.  Expression dominates feature selection in the inspected fold", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_modalities.png", dpi=190); plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def fig4_two_classes(df):
    """R2 and Spearman disagree, and the disagreement is biologically meaningful."""
    sel = df[(df.scheme == "lineage_blocked") & (df.arm == "E_multiomics") &
             (df.stratum == "selective")]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(sel.r2, sel.spearman, s=16, alpha=0.4, color="#777777", edgecolors="none")
    named = {"SOX10": "lineage TF", "PAX8": "lineage TF", "TP63": "lineage TF",
             "HNF1B": "lineage TF", "KLF5": "lineage TF", "CTNNB1": "pathway",
             "FAM50A": "paralog", "DDX3X": "paralog", "RPP25L": "paralog",
             "CDS2": "paralog", "EFR3A": "paralog", "MYBL2": "paralog", "STX4": "paralog"}
    cols = {"lineage TF": "#C44E52", "paralog": "#55A868", "pathway": "#8172B3"}
    # hand-tuned label offsets so the dense top-right cluster stays readable
    off = {"FAM50A": (7, 2), "RPP25L": (7, 4), "CDS2": (-34, 6), "DDX3X": (7, -2),
           "EFR3A": (-36, -10), "MYBL2": (-38, 8), "STX4": (7, -2), "KLF5": (7, -3),
           "CTNNB1": (7, 3), "TP63": (7, -1), "HNF1B": (7, -4), "PAX8": (7, -4),
           "SOX10": (7, 1)}
    for _, r in sel[sel.symbol.isin(named)].iterrows():
        k = named[r.symbol]
        ax.scatter(r.r2, r.spearman, s=52, color=cols[k], zorder=3, edgecolors="w", lw=0.8)
        ax.annotate(r.symbol, (r.r2, r.spearman), fontsize=8.5, zorder=4,
                    xytext=off.get(r.symbol, (6, -3)), textcoords="offset points",
                    color=cols[k])
    for k, c in cols.items():
        ax.scatter([], [], s=52, color=c, label="candidate buffering" if k == "paralog" else k)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$S_{mean}$ (lineage-blocked CV)")
    ax.set_ylabel("Spearman $\\rho$ (lineage-blocked CV)")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Figure 4.  Two kinds of predictable gene\n"
                 "Skill and rank correlation emphasize different prediction properties",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_two_classes.png", dpi=190); plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig5_mechanism():
    """What the model learned: the single strongest predictor of each top gene."""
    m = pd.read_csv(f"{DERIVED}/top_predictors.csv")
    m = m.head(14).iloc[::-1]
    cols = {"paralog": "#55A868", "lineage TF": "#C44E52",
            "pathway": "#8172B3", "other": "#999999"}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    y = np.arange(len(m))
    ax.barh(y, m.r2, color=[cols[c] for c in m["class"]])
    for i, r in enumerate(m.itertuples()):
        ax.text(r.r2 + 0.008, i, f"  ← {r.top_feature} ({r.top_r:+.2f})",
                va="center", fontsize=8.6, color="#333333")
    ax.set_yticks(y); ax.set_yticklabels(m.symbol, fontsize=9)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("$S_{mean}$ under lineage-blocked CV")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label="candidate buffering" if k == "paralog" else k) for k, c in cols.items()],
              fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Figure 5.  Post hoc, full-cohort expression associations of top genes",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_mechanism.png", dpi=190); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=RES)
    parser.add_argument("--figures-dir", default=FIG)
    parser.add_argument("--derived-dir", default=DERIVED)
    args = parser.parse_args()
    RES, FIG, DERIVED = args.results_dir, args.figures_dir, args.derived_dir
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DERIVED, exist_ok=True)
    df, ds, tidx = _load(args.data_dir)
    r = fig1_metric_inflation(df, ds, tidx)
    fig2_strata(df)
    fig3_modalities(df, ds, tidx)
    fig4_two_classes(df)
    fig5_mechanism()
    print("figures written to", FIG)
    print(r.round(3).to_string(index=False))
