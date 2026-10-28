import numpy as np, pandas as pd, json, os
OUT = os.path.expanduser("~/work/out")
L = lambda n: np.load(f"{OUT}/_{n}.npy", allow_pickle=True)
Y, ymod = L("Y"), L("ymod")
expr, emod = L("expr"), L("exprmod")
cn, cmod   = L("cn"),  L("cnmod")
mut, mmod  = L("mut"), L("mutmod")
model = pd.read_csv(f"{OUT}/_model.csv")

print("shapes:", {"Y":Y.shape,"expr":expr.shape,"cn":cn.shape,"mut":mut.shape})
print("dtypes:", Y.dtype, expr.dtype, mut.dtype)
print("sanity  Y  min/med/max: %.3f %.3f %.3f  NaN%%=%.3f" % (
      np.nanmin(Y), np.nanmedian(Y), np.nanmax(Y), 100*np.isnan(Y).mean()))
print("sanity expr min/med/max: %.3f %.3f %.3f  NaN%%=%.3f" % (
      np.nanmin(expr), np.nanmedian(expr), np.nanmax(expr), 100*np.isnan(expr).mean()))
print("sanity  cn min/med/max: %.3f %.3f %.3f  NaN%%=%.3f" % (
      np.nanmin(cn), np.nanmedian(cn), np.nanmax(cn), 100*np.isnan(cn).mean()))
print("sanity mut min/med/max: %.3f %.3f %.3f  uniq=%s" % (
      np.nanmin(mut), np.nanmedian(mut), np.nanmax(mut), np.unique(mut)[:6]))

mset = set(model.ModelID.astype(str))
s = {"Y":set(ymod),"expr":set(emod),"cn":set(cmod),"mut":set(mmod),"meta":mset}
for k,v in s.items(): print(f"  n_{k:5s} = {len(v)}")
core = s["Y"] & s["expr"] & s["meta"]
allm = core & s["cn"] & s["mut"]
print("CRISPR n expr n meta      =", len(core))
print("  + cn n mut (4 modality) =", len(allm))
