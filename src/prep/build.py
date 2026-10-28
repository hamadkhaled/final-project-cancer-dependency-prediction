"""Align parsed DepMap arrays, apply the original QC, and write analysis data.

Every model arm uses identical cell lines. Metadata (including PatientID) is
aligned once with a one-to-one ModelID join; duplicate IDs are rejected.
The saved 852-line cohort is a historical expected result, not hardcoded output.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(intermediate_dir=None, data_dir=None, raw_dir=None):
    import numpy as np
    import pandas as pd
    intermediate_dir = Path(intermediate_dir or ROOT / "work" / "prepared")
    data_dir = Path(data_dir or ROOT / "data")
    raw_dir = Path(raw_dir or ROOT / "work" / "raw")
    matrices, id_lists, names = {}, {}, {}
    for tag in ["Y", "expr", "cn", "mut"]:
        prefix = "y" if tag == "Y" else tag
        matrices[tag] = np.load(intermediate_dir / f"_{tag}.npy")
        ids = np.load(intermediate_dir / f"_{prefix}mod.npy").astype(str)
        if len(ids) != matrices[tag].shape[0] or pd.Series(ids).duplicated().any():
            raise ValueError(f"{tag}: row IDs are duplicated or do not match matrix rows")
        id_lists[tag] = ids
        with (intermediate_dir / f"_{prefix}genes.json").open() as handle:
            names[tag] = np.array(json.load(handle))
        if len(names[tag]) != matrices[tag].shape[1]:
            raise ValueError(f"{tag}: names do not match matrix columns")
    model = pd.read_csv(intermediate_dir / "_model.csv")
    if model.ModelID.isna().any() or model.ModelID.duplicated().any():
        raise ValueError("Model metadata contains duplicate or missing ModelID values")
    model["ModelID"] = model.ModelID.astype(str)
    if "PatientID" not in model:
        raise ValueError("PatientID missing: re-run revised prep.py to preserve patient metadata")
    cohort = sorted(set.intersection(*(set(v) for v in id_lists.values()), set(model.ModelID)))
    if not cohort:
        raise ValueError("No models shared by all four matrices and metadata")
    print("cohort n =", len(cohort))
    meta = pd.DataFrame({"ModelID": cohort}).merge(model, on="ModelID", how="left", validate="one_to_one", sort=False)
    if meta.PatientID.isna().any() or meta.PatientID.astype(str).str.strip().eq("").any():
        raise ValueError("Cohort has missing PatientID; resolve before patient-grouped evaluation")
    for tag in matrices:
        positions = {model_id: i for i, model_id in enumerate(id_lists[tag])}
        M = matrices[tag][[positions[m] for m in cohort]]
        complete = np.isfinite(M).all(axis=0)
        print(f"  {tag}: {complete.sum()}/{len(names[tag])} genes complete")
        M, feature_names = M[:, complete], names[tag][complete]
        if tag in ["expr", "cn"]:
            variable = M.std(axis=0) > 1e-8
            print(f"  {tag}: dropping {(~variable).sum()} zero-variance features")
            M, feature_names = M[:, variable], feature_names[variable]
        if tag == "mut":
            if not ((M >= 0) & (M <= 255) & (M == np.floor(M))).all():
                raise ValueError("Mutation counts must be nonnegative integers within uint8 range")
            keep = M.sum(axis=0) >= 10
            print(f"  mutation: keeping {keep.sum()} genes with summed damaging-mutation counts >=10")
            M, feature_names = M[:, keep].astype(np.uint8), feature_names[keep]
        matrices[tag], names[tag] = M, feature_names
    essentials = pd.read_csv(raw_dir / "CRISPRInferredCommonEssentials.csv")
    essentials = set(essentials.iloc[:, 0].astype(str))
    data_dir.mkdir(parents=True, exist_ok=True)
    for tag, M in matrices.items():
        np.save(data_dir / f"{tag}.npy", M)
    meta.to_csv(data_dir / "meta.csv", index=False)
    payload = {"cohort": cohort, "y": list(map(str, names["Y"])),
               **{tag: list(map(str, names[tag])) for tag in ["expr", "cn", "mut"]},
               "common_essentials": sorted(essentials)}
    with (data_dir / "names.json").open("w") as handle:
        json.dump(payload, handle)
    print("final:", {tag: M.shape for tag, M in matrices.items()})
    print("lineages:", meta.OncotreeLineage.nunique())
    print("distinct patients:", meta.PatientID.nunique(), "| lines sharing a patient:", int(meta.PatientID.duplicated(keep=False).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-dir", type=Path, default=ROOT / "work" / "prepared")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "work" / "raw")
    args = parser.parse_args()
    main(args.intermediate_dir, args.data_dir, args.raw_dir)
