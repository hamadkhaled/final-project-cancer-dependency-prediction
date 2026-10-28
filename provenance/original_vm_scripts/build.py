import numpy as np, pandas as pd, json, os
OUT = os.path.expanduser("~/work/out")
DST = os.path.expanduser("~/mnt/data/cbio7215")
os.makedirs(DST, exist_ok=True)
L = lambda n: np.load(f"{OUT}/_{n}.npy", allow_pickle=True)
G = lambda n: json.load(open(f"{OUT}/_{n}genes.json"))

Y, ymod = L("Y"), L("ymod").astype(str); ygenes = np.array(G("y"))
expr, emod = L("expr"), L("exprmod").astype(str); egenes = np.array(G("expr"))
cn, cmod = L("cn"), L("cnmod").astype(str); cgenes = np.array(G("cn"))
mut, mmod = L("mut"), L("mutmod").astype(str); mgenes = np.array(G("mut"))
model = pd.read_csv(f"{OUT}/_model.csv"); model["ModelID"] = model.ModelID.astype(str)

cohort = sorted(set(ymod) & set(emod) & set(cmod) & set(mmod) & set(model.ModelID))
print("cohort n =", len(cohort))
pos = lambda ids: {m:i for i,m in enumerate(ids)}
Y   = Y  [[pos(ymod)[m] for m in cohort]]
expr= expr[[pos(emod)[m] for m in cohort]]
cn  = cn [[pos(cmod)[m] for m in cohort]]
mut = mut[[pos(mmod)[m] for m in cohort]]
meta = model.set_index("ModelID").loc[cohort].reset_index()

# --- QC: keep complete-data columns -------------------------------------
def keep_complete(M, names, tag):
    ok = ~np.isnan(M).any(axis=0)
    print(f"  {tag}: {ok.sum()}/{len(names)} genes complete")
    return M[:, ok], names[ok]
Y,  ygenes = keep_complete(Y,  ygenes, "Y(dependency)")
expr, egenes = keep_complete(expr, egenes, "expression")
cn, cgenes = keep_complete(cn, cgenes, "copy-number")
mut, mgenes = keep_complete(mut, mgenes, "mutation")

# drop zero-variance features
for nm in ["expr","cn"]:
    M = {"expr":expr,"cn":cn}[nm]; names = {"expr":egenes,"cn":cgenes}[nm]
    v = M.std(axis=0) > 1e-8
    print(f"  {nm}: dropping {(~v).sum()} zero-variance")
    if nm=="expr": expr, egenes = M[:,v], names[v]
    else:          cn, cgenes   = M[:,v], names[v]
keepm = mut.sum(axis=0) >= 10
print(f"  mutation: keeping {keepm.sum()} genes damaged in >=10 lines")
mut, mgenes = mut[:,keepm].astype(np.uint8), mgenes[keepm]

ess = pd.read_csv(os.path.expanduser("~/mnt/Downloads/CRISPRInferredCommonEssentials.csv"))
ess = set(ess.iloc[:,0].astype(str))
print("  common essentials listed:", len(ess), "| present as targets:",
      sum(g in ess for g in ygenes))

np.save(f"{DST}/Y.npy", Y);     np.save(f"{DST}/expr.npy", expr)
np.save(f"{DST}/cn.npy", cn);   np.save(f"{DST}/mut.npy", mut)
meta.to_csv(f"{DST}/meta.csv", index=False)
json.dump({"cohort":cohort,"y":list(map(str,ygenes)),"expr":list(map(str,egenes)),
           "cn":list(map(str,cgenes)),"mut":list(map(str,mgenes)),
           "common_essentials":sorted(ess)}, open(f"{DST}/names.json","w"))
print("final:", {"Y":Y.shape,"expr":expr.shape,"cn":cn.shape,"mut":mut.shape})
print("lineages:", meta.OncotreeLineage.nunique(),
      "| top:", meta.OncotreeLineage.value_counts().head(5).to_dict())
for f in sorted(os.listdir(DST)):
    print(f"  {os.path.getsize(f'{DST}/{f}')/1e6:8.1f} MB  {f}")
