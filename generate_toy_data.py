"""Create small, synthetic DepMap-shaped CSVs for a runnable pipeline example.

These are artificial measurements and artificial identifiers. They do not
represent real cells, genes, patients, or any result in the technical report.
The default CSVs are already included; this script regenerates them with seed 42.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def generate(output_dir: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 80
    cell_ids = [f"TOY-CELL-{i:04d}" for i in range(n)]
    lineage_id = np.repeat(np.arange(8), 10)
    metadata = pd.DataFrame({
        "ModelID": cell_ids,
        "OncotreeLineage": [f"TOY_TYPE_{i}" for i in lineage_id],
        "OncotreePrimaryDisease": "SYNTHETIC_EXAMPLE",
        "OncotreeSubtype": "SYNTHETIC_EXAMPLE",
        "DepmapModelType": "SYNTHETIC_EXAMPLE",
        "PatientID": [f"TOY-PATIENT-{i//2:04d}" for i in range(n)],
        "Sex": ["Female" if i % 2 else "Male" for i in range(n)],
    })
    metadata.to_csv(output_dir / "Model.csv", index=False)

    expression = rng.normal(5.0, 1.0, (n, 24))
    copy_number = np.exp(rng.normal(0.0, 0.3, (n, 16)))
    mutation = np.zeros((n, 12), dtype=np.uint8)
    for j in range(mutation.shape[1]):
        mutation[rng.choice(n, 16, replace=False), j] = 1

    targets = ([f"TOY_SELECTIVE_{i:02d} ({10000+i})" for i in range(10)]
               + [f"TOY_ESSENTIAL_{i:02d} ({20000+i})" for i in range(5)]
               + [f"TOY_OTHER_{i:02d} ({30000+i})" for i in range(5)])
    outcomes = np.zeros((n, len(targets)))
    for j in range(10):
        outcomes[:, j] = (-0.55 + 0.50 * (expression[:, j] - 5.0)
                          + 0.35 * (copy_number[:, j] - 1.0)
                          - 0.25 * mutation[:, j]
                          + 0.05 * lineage_id
                          + rng.normal(0.0, 0.10, n))
    outcomes[:, 10:15] = rng.normal(-1.0, 0.035, (n, 5))
    outcomes[:, 15:] = rng.normal(-0.03, 0.02, (n, 5))

    expected_selective = ((outcomes.std(axis=0) >= 0.25)
                          & ((outcomes < -0.5).sum(axis=0) >= 5))
    assert expected_selective[:10].all() and not expected_selective[10:].any()
    y = pd.DataFrame(outcomes, columns=targets)
    y.insert(0, "Unnamed: 0", cell_ids)
    y.to_csv(output_dir / "CRISPRGeneEffect.csv", index=False)
    pd.DataFrame({"gene": targets[10:15]}).to_csv(
        output_dir / "CRISPRInferredCommonEssentials.csv", index=False)

    specs = [
        ("OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", expression, "TOY_EXPR"),
        ("OmicsCNGeneWGS.csv", copy_number, "TOY_CN"),
        ("OmicsSomaticMutationsMatrixDamaging.csv", mutation, "TOY_MUT"),
    ]
    for filename, matrix, label in specs:
        frame = pd.DataFrame(matrix, columns=[f"{label}_{j:02d} ({40000+j})"
                                              for j in range(matrix.shape[1])])
        frame.insert(0, "IsDefaultEntryForModel", True)
        frame.insert(0, "ModelID", cell_ids)
        frame.to_csv(output_dir / filename, index=False)
    print(f"Synthetic CSVs: {n} cell lines, 8 toy types, 40 toy patients, 20 targets.")
    print("No data or results here are biological observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/toy/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
