"""Run the actual preparation, validation and six-arm training code on toy data.

Run: python run_toy.py
Requires the packages in requirements.txt. All generated files go to work/toy/;
the archived real-data result tables under results/ are never replaced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    work = ROOT / "work/toy"
    work.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MPLCONFIGDIR", str(work / "matplotlib_cache"))

    def run(*arguments: str | Path) -> None:
        command = [sys.executable, *map(str, arguments)]
        print("\nRUN:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    raw = ROOT / "data/toy/raw"
    if not (raw / "CRISPRGeneEffect.csv").exists():
        run(ROOT / "generate_toy_data.py")
    run(ROOT / "src/prep/prep.py", "--raw-dir", raw,
        "--intermediate-dir", work / "prepared")
    run(ROOT / "src/prep/build.py", "--raw-dir", raw,
        "--intermediate-dir", work / "prepared", "--data-dir", work / "data")
    run(ROOT / "src/test_pipeline.py", "--data-dir", work / "data")
    run(ROOT / "src/run_experiment.py", "--data-dir", work / "data",
        "--results-dir", work / "results", "--tag", "toy", "--topk", "12")

    results = pd.read_csv(work / "results/toy_pergene.csv")
    assert len(results) == 2 * 6 * 20, "Unexpected toy experiment coverage"
    assert not results.duplicated(["scheme", "arm", "gene"]).any()
    assert np.isfinite(results["r2"]).all()
    counts = results.groupby(["scheme", "arm"]).size()
    assert len(counts) == 12 and counts.eq(20).all()
    summary = (results.loc[results.stratum.eq("selective")]
               .groupby(["scheme", "arm"])["r2"].median()
               .rename("median_S_mean").reset_index())
    summary["synthetic_data"] = True
    summary.to_csv(work / "results/toy_summary.csv", index=False)
    status = {"synthetic_data": True, "cell_lines": 80, "target_genes": 20,
              "schemes": 2, "arms": 6, "outer_folds": 5, "topk": 12,
              "result_rows": len(results), "status": "passed",
              "scope": "Runnable demonstration only; does not reproduce report findings."}
    (work / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print("\nTOY PIPELINE PASSED: both evaluations, all six arms and all 20 toy targets.")
    print("Synthetic results:", work / "results/toy_summary.csv")
    print("This demonstration is separate from the real-data results in the report.")


if __name__ == "__main__":
    main()
