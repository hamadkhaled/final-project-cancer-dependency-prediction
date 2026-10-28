"""Validate supplied CSV summaries and cohort metadata without raw matrices.

This checks numerical/internal consistency, NOT model training or every written
claim. Fold checks use the archived assignments in results/fold_assignments.csv,
which are the partitions used by the original experiment. The full input
matrices are supplied separately; saved out-of-fold prediction arrays are
included for the lineage-only and fusion arms but are not read here.
Optional --deck checks that every slide has notes; it does not certify layout.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import zipfile
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def group_folds(groups, n_folds=5):
    """Reproduce common._group_kfold without importing SciPy/data matrices."""
    uniq, counts = np.unique(groups, return_counts=True)
    sizes, fold_of = np.zeros(n_folds, dtype=int), {}
    for group in uniq[np.argsort(-counts)]:
        fold = int(np.argmin(sizes))
        fold_of[group] = fold
        sizes[fold] += int(counts[uniq == group][0])
    assign = np.array([fold_of[group] for group in groups])
    return [(np.flatnonzero(assign != fold), np.flatnonzero(assign == fold)) for fold in range(n_folds)]


def main(results_dir, metadata, deck=None):
    failures, checks = [], 0
    def check(label, condition):
        nonlocal checks
        checks += 1
        print(f"{'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            failures.append(label)
    def close(label, got, expected):
        check(label, bool(np.isclose(got, expected, atol=1e-10, rtol=1e-9, equal_nan=True)))

    per = pd.read_csv(results_dir / "main_pergene.csv")
    meta = pd.read_csv(metadata)
    check("cohort ModelID unique and present", not meta.ModelID.isna().any() and not meta.ModelID.duplicated().any())
    check("PatientID complete", not meta.PatientID.isna().any() and not meta.PatientID.astype(str).str.strip().eq("").any())
    check("main per-gene scheme/arm/gene keys unique", not per.duplicated(["scheme", "arm", "gene"]).any())
    check("main skill scores finite", bool(np.isfinite(per.r2).all()))
    print(f"INFO  Saved cohort: {len(meta)} lines, {meta.OncotreeLineage.nunique()} lineages, {meta.PatientID.nunique()} patients; {meta.PatientID.duplicated(keep=False).sum()} lines share a patient.")
    patients = meta.PatientID.fillna("NA").astype(str).to_numpy()
    lineage = meta.OncotreeLineage.fillna("Unknown").astype(str)
    counts = lineage.value_counts()
    blocked_groups = lineage.where(~lineage.isin(counts.index[counts < 10]), "Other").to_numpy()
    archived = pd.read_csv(results_dir / "fold_assignments.csv")
    for scheme, groups in [("patient_grouped", patients), ("lineage_blocked", blocked_groups)]:
        block = archived[archived.scheme == scheme].sort_values("row_index")
        check(f"{scheme}: archived fold rows match metadata ModelIDs",
              len(block) == len(meta) and np.array_equal(block.row_index.to_numpy(), np.arange(len(meta)))
              and np.array_equal(block.ModelID.astype(str).to_numpy(), meta.ModelID.astype(str).to_numpy()))
        labels = block.fold.astype(int).to_numpy()
        folds = [(np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)) for fold in range(5)]
        regenerated = np.full(len(meta), -1)
        for fold, (_, te) in enumerate(group_folds(groups)):
            regenerated[te] = fold
        same = np.array_equal(regenerated, labels)
        print(f"INFO  {scheme}: archived test-fold sizes {[len(te) for _, te in folds]}; "
              f"splitter regenerated in this environment {'matches' if same else 'DIFFERS from'} the archive"
              f"{'' if same else ' (tie order depends on the sorting implementation; the archive is authoritative)'}.")
        covered = np.concatenate([te for _, te in folds])
        check(f"{scheme}: test rows cover cohort exactly once", len(covered) == len(meta) and len(np.unique(covered)) == len(meta))
        check(f"{scheme}: train/test disjoint", all(not set(tr) & set(te) for tr, te in folds))
        check(f"{scheme}: patient IDs disjoint across train/test", all(not set(patients[tr]) & set(patients[te]) for tr, te in folds))
        if scheme == "lineage_blocked":
            lin = lineage.to_numpy()
            check("lineage_blocked: original lineages disjoint across train/test", all(not set(lin[tr]) & set(lin[te]) for tr, te in folds))

    expected_genes = set(per.gene)
    for (scheme, arm), block in per.groupby(["scheme", "arm"], sort=False):
        check(f"{scheme}/{arm}: same target panel", set(block.gene) == expected_genes)
    summary = pd.read_csv(results_dir / "main_table_main.csv")
    for row in summary.itertuples():
        block = per[(per.scheme == row.scheme) & (per.arm == row.arm) & (per.stratum == row.stratum)]
        label = f"{row.scheme}/{row.stratum}/{row.arm}"
        check(f"{label}: gene count", len(block) == row.n_genes)
        close(f"{label}: median S_mean", block.r2.median(), row.median_r2)
        close(f"{label}: fraction S_mean>0", (block.r2 > 0).mean(), row.frac_r2_gt_0)
        check(f"{label}: saved CI brackets median", row.ci_lo <= row.median_r2 <= row.ci_hi)
    delta = pd.read_csv(results_dir / "main_table_delta.csv")
    for row in delta.itertuples():
        block = per[(per.scheme == row.scheme) & (per.stratum == row.stratum)]
        wide = block.pivot(index="gene", columns="arm", values="r2")
        difference = (wide[row.arm] - wide["A_lineage"]).dropna()
        label = f"{row.scheme}/{row.stratum}/{row.arm}-A"
        check(f"{label}: paired count", len(difference) == row.n_genes)
        close(f"{label}: median paired skill", difference.median(), row.median_delta_r2)
        close(f"{label}: fraction beating A", (difference > 0).mean(), row.frac_better_than_lineage)

    wide = per[(per.scheme == "lineage_blocked") & (per.stratum == "selective")].pivot(index="gene", columns="arm", values="r2")
    difference = wide.E_multiomics - wide.B_expression
    print(f"INFO  Selective S2 fusion minus expression: median {difference.median():+.8f}; {int((difference > 0.05).sum())} gains >0.05; {int((difference < -0.05).sum())} losses >0.05.")
    legacy = results_dir / "fig1_metric_inflation.csv"
    if legacy.exists():
        legacy_frame = pd.read_csv(legacy)
        if "gene_panel" not in legacy_frame:
            print("SKIP  Legacy pooled-correlation comparison: mixed gene panels; not used as evidence in the report.")
    print("SKIP  Training and raw-matrix QC: full input matrices are supplied separately. Saved prediction arrays for the lineage-only and fusion arms are included but not read by this checker.")
    if deck is None:
        print("SKIP  Deck structure/notes: supply --deck PATH to check a revised PPTX.")
    else:
        ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        with zipfile.ZipFile(deck) as archive:
            slide_files = [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
            check("revised deck contains slides", bool(slide_files))
            for name in slide_files:
                number = Path(name).stem.removeprefix("slide")
                notes_file = f"ppt/notesSlides/notesSlide{number}.xml"
                if notes_file not in archive.namelist():
                    check(f"slide {number}: speaker notes exist", False)
                    continue
                xml = ET.fromstring(archive.read(notes_file))
                bodies = [shape for shape in xml.findall(".//p:sp", ns) if shape.find('.//p:ph[@type="body"]', ns) is not None]
                body_text = " ".join(text.text or "" for shape in bodies for text in shape.findall(".//a:t", ns)).strip()
                check(f"slide {number}: nonempty speaker notes", bool(body_text))
        print("INFO  Deck checks are structural. Inspect renders and rehearse timings separately.")
    print(f"{checks - len(failures)}/{checks} numerical/structural checks passed; skipped work is not verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "meta.csv")
    parser.add_argument("--deck", type=Path)
    args = parser.parse_args()
    sys.exit(main(args.results_dir, args.metadata, args.deck))
