"""
Main experiment: six model arms x two CV schemes x stratified target genes.

Usage:
    python3 src/run_experiment.py [--topk 300] [--tag main] [--arms A_lineage,E_multiomics]

Writes results/<tag>_pergene.csv (one row per scheme x arm x gene) and, for the
two arms that the failure analysis needs, results/<tag>_preds_<scheme>_<arm>.npy.

The report specifies the historical protocol. Seeds: common.SEED.
Submission-only changes add input/output paths; the estimator and scoring math are unchanged.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Dataset, SCHEMES, ARMS, ALPHAS, TOPK_DEFAULT, SEED,
                    ridge_gcv, pooled_r2, spearman)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SAVE_PREDS = {"A_lineage", "E_multiomics"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run(topk: int = TOPK_DEFAULT, tag: str = "main", arms=None,
        drop_self: bool = False, permute_lineage: bool = False,
        stratum: str | None = None, max_genes: int | None = None,
        estimator: str = "ridge", log_cn: bool = False,
        data_dir: str | None = None, results_dir: str | None = None,
        fold_assignments: str | None = None):
    """Fit the shared experiment; optional paths isolate toy runs from archived results."""
    ds = Dataset(log_cn=log_cn) if data_dir is None else Dataset(data_dir=data_dir, log_cn=log_cn)
    output_dir = os.path.abspath(results_dir or RES)
    tidx, tinfo = ds.target_set()
    if stratum:                            # ablations run on one stratum to bound compute
        keep = np.where(tinfo.stratum.values == stratum)[0]
        tidx, tinfo = tidx[keep], tinfo.iloc[keep].reset_index(drop=True)
    if max_genes and len(tidx) > max_genes:
        rs = np.random.default_rng(SEED).choice(len(tidx), max_genes, replace=False)
        rs = np.sort(rs)
        tidx, tinfo = tidx[rs], tinfo.iloc[rs].reset_index(drop=True)
    Y = ds.Y[:, tidx]                      # (n, G)
    G = Y.shape[1]
    arms = arms or list(ARMS.keys())
    log(f"targets={G}  features={ds.X.shape[1]}  arms={arms}  topk={topk} "
        f"drop_self={drop_self} permute_lineage={permute_lineage}")

    # Map each target gene to the feature columns that are "itself", for the
    # self-feature ablation (design.md §11.1).
    self_cols = None
    if drop_self:
        bysym = {}
        for j, s in enumerate(ds.feat_symbol):
            bysym.setdefault(s, []).append(j)
        self_cols = [np.array(bysym.get(s, []), dtype=int) for s in ds.y_symbol[tidx]]

    rows, pred_store = [], {}
    for scheme_name, splitter in SCHEMES.items():
        if fold_assignments:
            saved = pd.read_csv(fold_assignments)
            block = saved.loc[saved.scheme == scheme_name].copy()
            if len(block) != ds.n or block.ModelID.duplicated().any():
                raise ValueError(f"{scheme_name}: saved folds must contain every model exactly once")
            block = block.set_index("ModelID").reindex(ds.meta.ModelID)
            if block.fold.isna().any() or set(block.fold.astype(int)) != set(range(5)):
                raise ValueError(f"{scheme_name}: invalid or incomplete saved fold assignments")
            labels = block.fold.astype(int).to_numpy()
            splits = [(np.where(labels != f)[0], np.where(labels == f)[0]) for f in range(5)]
            for tr, te in splits:
                if set(ds.meta.PatientID.iloc[tr]) & set(ds.meta.PatientID.iloc[te]):
                    raise ValueError("Saved folds overlap patients")
                if scheme_name == "lineage_blocked" and set(ds.lineage[tr]) & set(ds.lineage[te]):
                    raise ValueError("Saved folds overlap lineages")
        else:
            splits = splitter(ds)
        # accumulators: per arm, per gene -> lists over folds
        acc = {a: {"yt": [[] for _ in range(G)], "yp": [[] for _ in range(G)],
                   "tm": [[] for _ in range(G)], "al": [[] for _ in range(G)]}
               for a in arms}
        order = np.concatenate([te for _, te in splits])

        for fi, (tr, te) in enumerate(splits):
            t0 = time.time()
            Xtr_raw, Xte_raw = ds.X[tr], ds.X[te]
            mu, sd = Xtr_raw.mean(axis=0), Xtr_raw.std(axis=0)
            sd[sd < 1e-8] = 1.0
            Xtr = (Xtr_raw - mu) / sd
            Xte = (Xte_raw - mu) / sd

            Lin_tr, Lin_te = ds.lineage_onehot[tr], ds.lineage_onehot[te]
            if permute_lineage:
                rng = np.random.default_rng(SEED + fi)
                Lin_tr = Lin_tr[rng.permutation(len(tr))]
                Lin_te = Lin_te[rng.permutation(len(te))]

            # One correlation screen per fold, sliced per arm afterwards.
            Ytr = Y[tr]
            yc = (Ytr - Ytr.mean(axis=0)) / np.where(Ytr.std(axis=0) < 1e-12, 1.0,
                                                     Ytr.std(axis=0))
            C = (Xtr.T @ yc) / (len(tr) - 1)          # (p, G)

            for arm in arms:
                spec = ARMS[arm]
                mods = spec["modalities"]
                use_lin = spec["lineage"]
                if mods:
                    mod_mask = np.isin(ds.modality, mods)
                    mod_idx = np.where(mod_mask)[0]
                for gi in range(G):
                    blocks_tr, blocks_te = [], []
                    if use_lin:
                        blocks_tr.append(Lin_tr); blocks_te.append(Lin_te)
                    if mods:
                        c = np.abs(C[mod_idx, gi])
                        if drop_self and len(self_cols[gi]):
                            ban = np.isin(mod_idx, self_cols[gi])
                            c = np.where(ban, -1.0, c)
                        k = min(topk, len(mod_idx))
                        sel = mod_idx[np.argpartition(-c, k - 1)[:k]]
                        blocks_tr.append(Xtr[:, sel]); blocks_te.append(Xte[:, sel])
                    Xa_tr = np.hstack(blocks_tr) if len(blocks_tr) > 1 else blocks_tr[0]
                    Xa_te = np.hstack(blocks_te) if len(blocks_te) > 1 else blocks_te[0]
                    ytr = Y[tr, gi]
                    if estimator == "rf":
                        from sklearn.ensemble import RandomForestRegressor
                        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                                   max_features=0.3, random_state=SEED,
                                                   n_jobs=1).fit(Xa_tr, ytr)
                        pred, alpha = rf.predict(Xa_te), np.nan
                    else:
                        pred, alpha = ridge_gcv(Xa_tr, ytr, Xa_te)
                    a = acc[arm]
                    a["yt"][gi].append(Y[te, gi]); a["yp"][gi].append(pred)
                    a["tm"][gi].append(ytr.mean()); a["al"][gi].append(alpha)
            log(f"  {scheme_name} fold {fi+1}/{len(splits)} "
                f"({len(tr)}/{len(te)}) done in {time.time()-t0:.1f}s")

        for arm in arms:
            a = acc[arm]
            for gi in range(G):
                yt = np.concatenate(a["yt"][gi]); yp = np.concatenate(a["yp"][gi])
                rows.append({
                    "scheme": scheme_name, "arm": arm,
                    "gene": tinfo.gene[gi], "symbol": tinfo.symbol[gi],
                    "stratum": tinfo.stratum[gi], "gene_std": tinfo["std"][gi],
                    "r2": pooled_r2(a["yt"][gi], a["yp"][gi], a["tm"][gi]),
                    "spearman": spearman(yt, yp),
                    "alpha_median": float(np.median(a["al"][gi])),
                })
            if arm in SAVE_PREDS and not drop_self and not permute_lineage:
                P = np.full((ds.n, G), np.nan, dtype=np.float32)
                for gi in range(G):
                    P[order, gi] = np.concatenate(a["yp"][gi])
                pred_store[f"{tag}_preds_{scheme_name}_{arm}"] = P

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out = f"{output_dir}/{tag}_pergene.csv"
    df.to_csv(out, index=False)
    for k, v in pred_store.items():
        np.save(f"{output_dir}/{k}.npy", v)
    np.save(f"{output_dir}/{tag}_target_idx.npy", tidx)
    log(f"wrote {out}  rows={len(df)}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--drop-self", action="store_true")
    ap.add_argument("--permute-lineage", action="store_true")
    ap.add_argument("--stratum", default=None, help="restrict targets to one stratum")
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "rf"])
    ap.add_argument("--log-cn", action="store_true", help="log2(x+1) the copy-number block")
    ap.add_argument("--data-dir", default=None, help="Aligned input directory; defaults to project data/")
    ap.add_argument("--results-dir", default=None, help="Output directory; defaults to project results/")
    ap.add_argument("--fold-assignments", default=None, help="Saved fold CSV; use results/fold_assignments.csv for the original real-data partition")
    args = ap.parse_args()
    run(topk=args.topk, tag=args.tag,
        arms=args.arms.split(",") if args.arms else None,
        drop_self=args.drop_self, permute_lineage=args.permute_lineage,
        stratum=args.stratum, max_genes=args.max_genes, estimator=args.estimator,
        log_cn=args.log_cn, data_dir=args.data_dir, results_dir=args.results_dir,
        fold_assignments=args.fold_assignments)
