"""
Ablation and sensitivity summary (design.md section 11).

Every comparison is PAIRED against the main run on the identical genes, which is
why differences can be read directly. Run: python3 src/analyze_ablations.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"
SC = "lineage_blocked"


def get(tag):
    p = f"{RES}/{tag}_pergene.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def arm_r2(df, arm, scheme=SC, stratum="selective"):
    d = df[(df.scheme == scheme) & (df.arm == arm) & (df.stratum == stratum)]
    return d.set_index("gene").r2


def paired(a, b, label):
    """Median paired difference a - b over the genes both runs share."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    d = (j.a - j.b).values
    m, lo, hi = bootstrap_ci(d)
    return {"comparison": label, "n_genes": len(j), "median_delta_r2": m,
            "ci_lo": lo, "ci_hi": hi, "frac_worse": float((d < 0).mean())}


def main():
    main_df = get("main")
    rows, notes = [], []

    E = arm_r2(main_df, "E_multiomics")
    A = arm_r2(main_df, "A_lineage")
    C = arm_r2(main_df, "C_copynumber")
    D = arm_r2(main_df, "D_mutation")

    # --- 11.1 self-feature exclusion ---------------------------------------
    s = get("abl_self")
    if s is not None:
        rows.append(paired(arm_r2(s, "E_multiomics"), E,
                           "self-features removed  vs  main (arm E)"))

    # --- 11.2 lineage-permuted null N2 -------------------------------------
    p = get("abl_permlin")
    if p is not None:
        rows.append(paired(arm_r2(p, "A_lineage"), A,
                           "lineage PERMUTED (N2)  vs  real lineage (N1), arm A"))
        for sc in ["patient_grouped", "lineage_blocked"]:
            v = arm_r2(p, "A_lineage", scheme=sc).values
            m, lo, hi = bootstrap_ci(v)
            notes.append(f"  N2 permuted-lineage null, {sc}: median R2 = {m:+.4f} "
                         f"[{lo:+.4f},{hi:+.4f}]")

    # --- 11.3 feature budget ------------------------------------------------
    for k in [100, 600]:
        t = get(f"abl_k{k}")
        if t is not None:
            rows.append(paired(arm_r2(t, "E_multiomics"), E, f"top-k = {k}  vs  k = 300 (arm E)"))

    # --- 11.4 estimator ------------------------------------------------------
    r = get("abl_rf")
    if r is not None:
        rows.append(paired(arm_r2(r, "E_multiomics"), E,
                           "random forest  vs  ridge (arm E, 200-gene subsample)"))

    # --- copy-number transform ----------------------------------------------
    lg = get("abl_logcn")
    if lg is not None:
        rows.append(paired(arm_r2(lg, "C_copynumber"), C,
                           "log2(CN+1)  vs  raw CN (arm C)"))
        rows.append(paired(arm_r2(lg, "E_multiomics"), E,
                           "log2(CN+1)  vs  raw CN (arm E)"))

    # --- mutation / CN low-dimensional budgets -------------------------------
    for k in [20, 50, 100]:
        t = get(f"abl_k{k}_lowdim")
        if t is not None:
            rows.append(paired(arm_r2(t, "D_mutation"), D, f"mutation at k = {k}  vs  k = 300"))
            v = arm_r2(t, "D_mutation").values
            m, lo, hi = bootstrap_ci(v)
            notes.append(f"  arm D (mutation) at k={k}: median R2 = {m:+.4f} [{lo:+.4f},{hi:+.4f}]")

    out = pd.DataFrame(rows)
    out.to_csv(f"{RES}/ablation_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print("=== PAIRED ABLATION RESULTS (selective genes, lineage-blocked CV) ===")
    print(out.round(4).to_string(index=False))
    if notes:
        print("\n=== absolute levels ===")
        print("\n".join(notes))

    # --- stratum threshold sensitivity (no refit needed: subset of main) -----
    print("\n=== 11.5 selective-threshold sensitivity (subsets of the main run) ===")
    sub = main_df[(main_df.scheme == SC) & (main_df.stratum == "selective")]
    piv = sub.pivot_table(index="gene", columns="arm", values="r2")
    gstd = sub.drop_duplicates("gene").set_index("gene").gene_std
    for thr in [0.25, 0.30, 0.35, 0.45]:
        keep = gstd[gstd >= thr].index
        v = (piv.loc[keep, "E_multiomics"] - piv.loc[keep, "A_lineage"]).dropna().values
        m, lo, hi = bootstrap_ci(v)
        print(f"  std >= {thr:.2f}  (n={len(keep):3d}):  median dR2(E-A) = {m:.4f} "
              f"[{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()
