"""
What did the model actually learn?

Produces:
  work/rerun/results/feature_composition.json - first-S2-training-fold composition
  work/rerun/results/top_predictors.csv - full-cohort, post hoc expression associations
  work/rerun/results/symbol_root_check.csv - gene-name heuristic, NOT homology annotation

The per-gene CLASS labels are a manual annotation of the 14 best-predicted genes
(candidate buffering / lineage TF / pathway), stated as such in the report.
Symbol-root matches do not establish paralogy or a buffering mechanism. Selecting
the strongest correlation from the full cohort is exploratory; it is not an
out-of-sample mechanism test. These outputs cannot support causal claims.
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Dataset, SCHEMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"
OUT = f"{ROOT}/work/rerun/results"

# Manual annotation of the best-predicted genes (see docstring).
CLASS = {"SOX10": "lineage TF", "PAX8": "lineage TF", "TP63": "lineage TF",
         "HNF1B": "lineage TF", "KLF5": "lineage TF", "JUN": "lineage TF",
         "FAM50A": "paralog", "DDX3X": "paralog", "RPP25L": "paralog",
         "CDS2": "paralog", "EFR3A": "paralog", "STX4": "paralog", "MYBL2": "paralog",
         "CTNNB1": "pathway"}


def symbol_root(sym: str) -> str:
    """Name heuristic only: strip suffix letters and digits.

    This can join unrelated genes or miss real homologs; use an external homology
    annotation and training-fold selection for a formal paralog analysis.
    """
    s = re.sub(r"L$", "", sym) if re.search(r"\dL$", sym) else sym
    s = re.sub(r"[A-Z]$", "", s) if re.search(r"\d[A-Z]$", s) else s
    return re.sub(r"\d+$", "", s)


def main(data_dir=None):
    ds = Dataset(data_dir=data_dir) if data_dir else Dataset()
    tidx = np.load(f"{RES}/main_target_idx.npy")
    Y = ds.Y[:, tidx]
    df = pd.read_csv(f"{RES}/main_pergene.csv")
    info = df[(df.scheme == "lineage_blocked") &
              (df.arm == "E_multiomics")].set_index("gene").loc[ds.y_genes[tidx]].reset_index()

    # ---- 1. modality composition of arm E's selection (fold 1 of S2) ----
    tr, te = SCHEMES["lineage_blocked"](ds)[0]
    Xtr = ds.X[tr]; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1
    Xs = (Xtr - mu) / sd
    Ytr = Y[tr]
    yc = (Ytr - Ytr.mean(0)) / np.where(Ytr.std(0) < 1e-12, 1, Ytr.std(0))
    C = np.abs((Xs.T @ yc) / (len(tr) - 1))
    selidx = np.where((info.stratum == "selective").values)[0]
    comp = np.zeros(3)
    for gi in selidx:
        top = np.argpartition(-C[:, gi], 299)[:300]
        for m in range(3):
            comp[m] += (ds.modality[top] == m).sum()
    comp = comp / comp.sum() * 100
    os.makedirs(OUT, exist_ok=True)
    json.dump({"expression": float(comp[0]), "copy_number": float(comp[1]),
               "mutation": float(comp[2]), "scope": "first lineage-blocked training fold",
               "n_training_lines": len(tr), "n_selective_genes": len(selidx),
               "topk": 300}, open(f"{OUT}/feature_composition.json", "w"))
    print(f"arm E top-300 composition (first S2 training fold): expression {comp[0]:.1f}%, "
          f"copy-number {comp[1]:.1f}%, mutation {comp[2]:.1f}%")

    # ---- 2. strongest single expression predictor, all selective genes ----
    Xe = ds.expr
    Z = (Xe - Xe.mean(0)) / np.where(Xe.std(0) < 1e-8, 1, Xe.std(0))
    em_sym = ds.feat_symbol[ds.modality == 0]
    rows = []
    for gi in selidx:
        y = Y[:, gi]
        # Both standard deviations above use ddof=0; divide by n for Pearson r.
        r = (Z.T @ ((y - y.mean()) / y.std())) / len(y)
        j = int(np.argmax(np.abs(r)))
        tgt = info.symbol[gi]
        rows.append({"symbol": tgt, "r2": info.r2[gi], "spearman": info.spearman[gi],
                     "top_feature": em_sym[j], "top_r": float(r[j]),
                     "is_self": em_sym[j] == tgt,
                     "is_symbol_root_match": (em_sym[j] != tgt) and
                                   (symbol_root(em_sym[j]) == symbol_root(tgt))})
    pc = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    pc.to_csv(f"{OUT}/symbol_root_check.csv", index=False)

    top = pc.head(14).copy()
    top["class"] = [CLASS.get(s, "other") for s in top.symbol]
    top.to_csv(f"{OUT}/top_predictors.csv", index=False)

    # ---- 3. descriptive name-heuristic rates; no homology enrichment claim ----
    q = pc.copy()
    q["decile"] = pd.qcut(q.r2.rank(method="first"), 10, labels=False)
    print("\nAcross all {} selective genes, is the single best expression predictor "
          "a symbol-root match? (unvalidated gene-name heuristic)".format(len(q)))
    for d in [9, 8, 5, 0]:
        s = q[q.decile == d]
        print(f"  R2 decile {d+1:>2}: self {s.is_self.mean():5.1%}   "
              f"symbol-root {s.is_symbol_root_match.mean():5.1%}   (n={len(s)}, median S_mean {s.r2.median():.3f})")
    print(f"\n  overall: self {q.is_self.mean():.1%}, symbol-root {q.is_symbol_root_match.mean():.1%}")
    hi = q[q.r2 > 0.3]
    print(f"  among the {len(hi)} genes with S_mean > 0.3: "
          f"self {hi.is_self.mean():.1%}, symbol-root {hi.is_symbol_root_match.mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=RES)
    parser.add_argument("--output-dir", default=OUT)
    args = parser.parse_args()
    RES, OUT = args.results_dir, args.output_dir
    main(args.data_dir)
