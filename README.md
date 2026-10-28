# Predicting Cancer Dependencies with Multimodal Molecular Data

Hamad Alhadhrami — Student ID 200010169  
CBIO7215: Multimodal AI for Therapeutic Discovery  
Instructor: Dr. Jun Wen — Fall 2026

This source-code package accompanies the separately submitted technical report.
It includes a runnable synthetic example, the model/preparation/analysis source,
the saved real-data result tables supporting the report, and links to the full
raw data. Presentation files, rehearsal notes, and previous document versions
are deliberately outside this submission package.

## 1. Install and run the small example

Use Python 3.10 or later. Open a terminal in this extracted folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_toy.py
```

On Windows, activate with `.venv\Scripts\activate` instead. If your system uses
`python3`, substitute that command for `python`.

The included data in `data/toy/raw/` are **synthetic**, not cancer observations.
The demo uses the **same source implementation as the main experiment** to:

1. Parse six small, DepMap-shaped CSV files.
2. Align cell IDs and apply the preparation/QC code.
3. Check fold grouping, score reference values and ridge against scikit-learn.
4. Train all six input arms under patient-grouped and lineage-blocked five-fold evaluation.
5. Save 240 per-gene result rows and a compact summary under `work/toy/`.

The toy cohort has 80 artificial cell lines, eight artificial cancer types,
40 artificial patient groups and 20 targets (10 selective, five essential and
five other). It has 24 expression, 16 copy-number and 12 mutation measurements.
Its molecular feature budget is 12 to keep the example small; the main study
uses up to 300. The toy data can be regenerated with:

```bash
python generate_toy_data.py
```

Expected completion message:

```text
TOY PIPELINE PASSED: both evaluations, all six arms and all 20 toy targets.
```

Outputs include `work/toy/RUN_STATUS.json`, `work/toy/results/toy_pergene.csv`
and `work/toy/results/toy_summary.csv`. **Toy outputs do not reproduce or
validate the biological findings in the report.** The demo writes separately
from the supplied real-data result tables.

Optional analysis of the newly trained toy results:

```bash
python src/analyze.py --tag toy --results-dir work/toy/results --figures-dir work/toy/figures
```

## 2. Check the real-data results reported in the paper

The supplied real-data results describe 852 cell lines, 29 lineages and
823 patients. There are 981 targets: 581 selective, 200 common-essential and
200 comparison genes. The saved CSV column `r2` is retained for compatibility,
but its meaning is training-mean predictive skill, called `S_mean` in the report.

The following commands require no real raw matrices:

```bash
python src/check_consistency.py
python src/verify_saved_results.py --source .
```

The consistency checker validates saved score summaries and checks patient and
lineage separation in the archived folds (`results/fold_assignments.csv`), the
partitions used by the original experiment. It also reports whether the splitter,
rerun in your environment, regenerates those folds; a difference there reflects
tie order, not a problem with the archived results.
The reanalysis writes `src/verified_summary.json` and the paired gene-level
fusion-versus-expression tables. It uses the recorded real-data scores, not
the toy experiment.

The four regenerated figures are included as SVG files in `figures/`. Report Figure 1 is
`fig2_fusion_distribution` and Figure 2 is `fig4_rank_vs_skill`; `fig1_selective_skill`
and `fig3_selected_cases` are supplementary views of Tables 2 and 3. To regenerate them,
install the optional dependency:

```bash
python -m pip install -r requirements-figures.txt
python analysis/make_verified_figures.py --source . --output-dir work/report_figures
```

Some inherited filenames contain words such as `r2`, `survivors` or
`metric_inflation`. They are historical names, not additional scientific claims.
The legacy pooled-correlation table compares different gene panels and is not
used as evidence for a matched-panel aggregation effect. The `paralog_check`
file contains an unvalidated gene-name heuristic, not a causal or curated
homology result. Use the report and verified analysis for interpretation.

## 3. Full raw data and a fresh real-data run

See [DATA_SOURCES.md](DATA_SOURCES.md) for the official download portal, release
reference, expected filenames and provenance limits. The source documentation
identifies **DepMap Public 26Q1**. The original processed matrices were produced on 1 October 2026 and are not distributed because of their size; the original training logs, saved predictions and fold assignments are included here. All six raw files and six processed files match the recorded SHA-256 checksums. A fresh preprocessing run reproduced all six processed files byte for byte. See VALIDATION.md for the 40-target model reruns (exact in the original software environment; close agreement in a second environment). The large raw files and processed matrices are omitted from this code ZIP; use the download links and rebuild commands below.

After obtaining the listed raw CSVs, place them in `data/raw/` and run:

```bash
python src/prep/prep.py --raw-dir data/raw --intermediate-dir work/real/prepared
python src/prep/build.py --raw-dir data/raw --intermediate-dir work/real/prepared --data-dir work/real/data
python src/test_pipeline.py --data-dir work/real/data
python src/run_experiment.py --data-dir work/real/data --results-dir work/real/results --tag rerun --topk 300 --fold-assignments results/fold_assignments.csv
python src/analyze.py --tag rerun --results-dir work/real/results --figures-dir work/real/figures
```

Use the saved fold CSV for real-data reproduction: the original splitter uses an unstable sort for equal-sized groups, so regenerating folds on a different NumPy/platform can change assignments despite identical fold sizes. The optional `--fold-assignments` argument matches ModelIDs and validates patient/lineage isolation. Toy data use their own generated groups.

Mutation inputs retain damaging-mutation counts (observed 0, 1 or 2), rather than being forced to binary. The original sparse-feature rule is a summed count of at least 10, not necessarily ten distinct cell lines.

These paths preserve the archived `results/` and `data/meta.csv`. A different
release or different processing can change the cohort and scores; compare
the constructed cohort with the recorded counts before interpreting results.
The result CSV holds every target/arm/evaluation score. Out-of-fold prediction
arrays are saved for the lineage-only and fusion arms, matching the supplied
implementation.

Sensitivity options on `src/run_experiment.py` include `--drop-self`,
`--permute-lineage`, `--topk`, `--log-cn`, `--max-genes`, `--stratum`, and
`--estimator rf`. Use a distinct `--tag` for each run. The optional sensitivity
and mechanism analysis modules are described below; they are not required for
the quick synthetic demonstration.

To repeat the 40-target verification after rebuilding the inputs:

```bash
python src/run_experiment.py --data-dir work/real/data --results-dir work/verification --tag independent_check --max-genes 40 --topk 300 --fold-assignments results/fold_assignments.csv
```

The original source notes are preserved in `provenance/ORIGINAL_REPRODUCE.md` as a historical record. The prespecified design and logged deviations cited in the report are the unchanged original files `provenance/design.md` and `provenance/deviations.md`; code comments that cite `design.md` refer to that file. Their headings use the word "preregistered"; the report says "prespecified" because no independently time-stamped registration exists. Use the portable paths and explicit-fold commands in this README for current execution.

## 4. Code map

| File | Purpose |
|---|---|
| `run_toy.py` | One-command preparation, checks and training on included toy data |
| `generate_toy_data.py` | Recreate synthetic CSVs using seed 42 |
| `src/prep/prep.py` | Parse raw files and preserve model/patient IDs |
| `src/prep/build.py` | Align rows, reject duplicate/missing IDs and apply QC |
| `src/common.py` | Dataset loading, target selection, grouped folds, ridge/GCV and metrics |
| `src/run_experiment.py` | Six input arms, two evaluations and per-gene training/scoring |
| `src/test_pipeline.py` | Grouping, metric, numerical ridge and bootstrap checks |
| `src/check_consistency.py` | Saved result-summary, metadata and archived-fold checks |
| `src/verify_saved_results.py` | Report reanalysis: paired effects, cases and conditional intervals |
| `analysis/make_verified_figures.py` | Four report figures as SVG |
| `src/analyze.py` | Per-arm summaries and basic figures for a selected run |
| `src/analyze_ablations.py` | Summarize saved sensitivity experiments |
| `src/mechanism.py` | Exploratory marginal associations; optional data/result paths |
| `src/make_figures.py` | Extended diagnostic plots; some require raw matrices/predictions |
| `results/` | Supplied real-data scores and derived historical tables |
| `data/meta.csv` | Supplied public-cell-line metadata for grouping checks |
| `figures/` | The four report figures as SVG, made by `analysis/make_verified_figures.py` |
| `logs/` | Original run logs; the two empty files are explained in `provenance/ORIGINAL_REPRODUCE.md` |

## 5. Method and interpretation

S1 groups patients; S2 holds out complete lineages, pooling rare lineages.
Scaling and selection of molecular inputs use the outer training group.
Target-panel selection uses the full cohort. Ridge's GCV penalty selection
treats the selected features as fixed; this is not fully nested, grouped
selection of the entire pipeline.

```text
S_mean = 1 - sum(held-out squared model error)
             / sum(held-out squared error of each fold's training mean)
```

Zero matches the training-mean predictor; positive is better; negative is worse.
It is not percentage accuracy or conventional test-mean R-squared. A lineage-only
model cannot learn coefficients for unseen categories, so beating it under S2
does not prove removal of tissue-associated biology. Gene-bootstrap intervals
are conditional on this selected panel and saved fits. Cases and biological
associations are exploratory. Cancer-cell-line predictions do not establish
drug efficacy, patient benefit or healthy-tissue safety.

## 6. Verification and submission changes

See [VALIDATION.md](VALIDATION.md) for executed checks and the verification
environment. Numerical model/scoring code in `src/common.py` is unchanged.
Submission additions are the synthetic data/example, documentation and explicit
input/output paths in training/testing/analysis. Analysis plot labels use
`S_mean` and avoid the original overinterpretation of lineage comparisons.
Archived real-data scores are unchanged. The technical report has updated data-availability and verification statements; the scientific conclusions are unchanged. Original historical reproduction notes, checksums and logs are retained under provenance/ and logs/.
