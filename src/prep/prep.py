"""Parse DepMap Public 26Q1 CSVs into arrays for build.py.

Defaults are portable project paths, not user-specific download directories.
Run with --help for the required raw file location and intermediate output path.
No training or outcome selection occurs here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RAW_DEFAULT = ROOT / "work" / "raw"
INTERMEDIATE_DEFAULT = ROOT / "work" / "prepared"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def truthy(s):
    """IsDefaultEntryForModel may arrive as bool or as a string."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_wide(path, id_col, meta_cols, default_col=None, chunksize=120):
    """Keep default entries; reject ambiguous duplicate ModelIDs instead of losing rows."""
    import numpy as np
    import pandas as pd
    keep, ids, cols = [], [], None
    n = 0
    for ch in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        n += len(ch)
        if id_col not in ch.columns:
            raise ValueError(f"{path}: missing ID column {id_col!r}")
        if default_col is not None:
            if default_col not in ch.columns:
                raise ValueError(f"{path}: missing default-entry flag {default_col!r}")
            ch = ch[truthy(ch[default_col])]
        if len(ch) == 0:
            continue
        if ch[id_col].isna().any():
            raise ValueError(f"{path}: missing model IDs")
        idv = ch[id_col].astype(str).to_numpy()
        ch = ch.drop(columns=[c for c in meta_cols if c in ch.columns])
        if cols is not None and list(ch.columns) != cols:
            raise ValueError(f"{path}: feature columns change across chunks")
        cols = list(ch.columns)
        keep.append(ch.to_numpy(dtype=np.float32))
        ids.append(idv)
    if not keep:
        raise ValueError(f"{path}: no usable rows")
    # Unicode arrays can be read without enabling pickle in build.py.
    all_ids = np.concatenate(ids).astype(str)
    duplicates = pd.Series(all_ids)[pd.Series(all_ids).duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"{path}: duplicate retained model IDs: {duplicates[:5].tolist()}")
    log(f"  {Path(path).name}: scanned {n} rows, kept {len(all_ids)}")
    return np.vstack(keep), all_ids, cols


def main(raw_dir=RAW_DEFAULT, intermediate_dir=INTERMEDIATE_DEFAULT):
    import numpy as np
    import pandas as pd
    raw_dir, intermediate_dir = Path(raw_dir), Path(intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("Model.csv")
    model = pd.read_csv(raw_dir / "Model.csv", low_memory=False)
    required = ["ModelID", "OncotreeLineage", "OncotreePrimaryDisease",
                "OncotreeSubtype", "DepmapModelType", "PatientID"]
    missing = set(required) - set(model.columns)
    if missing:
        raise ValueError(f"Model.csv: missing required columns {sorted(missing)}")
    if model.ModelID.isna().any() or model.ModelID.duplicated().any():
        raise ValueError("Model.csv must contain unique, nonmissing ModelID values")
    optional = ["Sex", "PrimaryOrMetastasis", "AgeCategory"]
    model = model[required + [c for c in optional if c in model.columns]]
    log("  model", model.shape)

    files = [("Y", "CRISPRGeneEffect.csv", "Unnamed: 0", ["Unnamed: 0"], None),
             ("expr", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", "ModelID", None, "IsDefaultEntryForModel"),
             ("cn", "OmicsCNGeneWGS.csv", "ModelID", None, "IsDefaultEntryForModel"),
             ("mut", "OmicsSomaticMutationsMatrixDamaging.csv", "ModelID", None, "IsDefaultEntryForModel")]
    omics_meta = ["Unnamed: 0", "SequencingID", "ModelConditionID", "ModelID",
                  "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
    for tag, filename, id_col, meta_cols, default_col in files:
        log(filename)
        X, ids, genes = load_wide(raw_dir / filename, id_col,
                                  meta_cols or omics_meta, default_col)
        np.save(intermediate_dir / f"_{tag}.npy", X)
        np.save(intermediate_dir / f"_{'y' if tag == 'Y' else tag}mod.npy", ids)
        with (intermediate_dir / f"_{'y' if tag == 'Y' else tag}genes.json").open("w") as handle:
            json.dump(genes, handle)
        log(f"  {tag}", X.shape)
        del X
    model.to_csv(intermediate_dir / "_model.csv", index=False)
    log("DONE in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DEFAULT)
    args = parser.parse_args()
    main(args.raw_dir, args.intermediate_dir)
