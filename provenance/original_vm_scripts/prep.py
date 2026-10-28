import pandas as pd, numpy as np, os, json, sys, time
D   = os.path.expanduser("~/mnt/Downloads")
OUT = os.path.expanduser("~/work/out")
os.makedirs(OUT, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def truthy(s):
    if s.dtype == bool: return s
    return s.astype(str).str.lower().isin(["true","1","yes"])

def load_wide(path, id_col, meta_cols, default_col=None, chunksize=120):
    keep, ids = [], []
    n = 0
    for ch in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        n += len(ch)
        if default_col is not None and default_col in ch.columns:
            ch = ch[truthy(ch[default_col])]
        if len(ch) == 0: continue
        idv = ch[id_col].astype(str).values
        ch = ch.drop(columns=[c for c in meta_cols if c in ch.columns])
        keep.append(ch.to_numpy(dtype=np.float32))
        ids.append(idv)
        cols = list(ch.columns)
    log(f"  {os.path.basename(path)}: scanned {n} rows, kept {sum(len(i) for i in ids)}")
    return np.vstack(keep), np.concatenate(ids), cols

t0 = time.time()
log("Model.csv")
model = pd.read_csv(f"{D}/Model.csv", low_memory=False)
model = model[["ModelID","OncotreeLineage","OncotreePrimaryDisease","OncotreeSubtype","DepmapModelType"]]
log("  model", model.shape)

log("CRISPRGeneEffect.csv")
Y, ymod, ygenes = load_wide(f"{D}/CRISPRGeneEffect.csv", "Unnamed: 0", ["Unnamed: 0"])
log("  Y", Y.shape)
np.save(f"{OUT}/_Y.npy", Y); np.save(f"{OUT}/_ymod.npy", ymod)
json.dump(ygenes, open(f"{OUT}/_ygenes.json","w")); del Y

OMICS_META = ["Unnamed: 0","SequencingID","ModelConditionID","ModelID",
              "IsDefaultEntryForMC","IsDefaultEntryForModel"]
for tag, fn in [("expr","OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"),
                ("cn",  "OmicsCNGeneWGS.csv"),
                ("mut", "OmicsSomaticMutationsMatrixDamaging.csv")]:
    log(fn)
    X, xmod, xgenes = load_wide(f"{D}/{fn}", "ModelID", OMICS_META,
                                default_col="IsDefaultEntryForModel")
    log(f"  {tag}", X.shape)
    np.save(f"{OUT}/_{tag}.npy", X); np.save(f"{OUT}/_{tag}mod.npy", xmod)
    json.dump(xgenes, open(f"{OUT}/_{tag}genes.json","w")); del X

model.to_csv(f"{OUT}/_model.csv", index=False)
log("DONE in %.1f min" % ((time.time()-t0)/60))
