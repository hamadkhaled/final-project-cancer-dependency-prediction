# Executed validation

## Real inputs and original records

The complete code/results archive supplied on 14 October was inspected. Original training logs, out-of-fold predictions for lineage and fusion, target indices, explicit fold assignments and source-file checksums are retained in this submission. The original main score CSV is identical to the one previously analyzed.

- All six local raw CSVs match the original recorded SHA-256 checksums.
- All six archived processed files (produced 1 October 2026 and used for every reported result) match those recorded checksums.
- A fresh run of the portable preparation/build code on the raw CSVs reproduced Y.npy, expr.npy, cn.npy, mut.npy, names.json and meta.csv byte for byte.
- Processed dimensions: 852 cell lines; 17,092 outcomes; 19,204 expression, 19,144 copy-number and 1,074 mutation inputs. Mutation values are counts (observed 0/1/2).

## Real-data model spot-check in a second environment (14 October)

Using the supplied matrices and archived folds, a fresh run trained all six arms under both evaluation schemes on a seeded sample of 40 targets: **480 gene-arm-scheme results**. Maximum absolute differences from original main results were:

- S_mean: 2.893235831e-06 (less than 0.000003).
- Spearman: 0.001302666326. The largest rank discrepancy is in the nearly constant lineage-only baseline; molecular-arm rank differences are below 0.00014.

This is close numerical agreement in a different software environment, not byte-identical score reproduction. It is a 40-target spot-check, not a rerun of all 981 targets or all sensitivities. See `validation/real_data_40_gene_comparison.csv`, `validation/independent_verification.json` and `logs/submission_real_40_gene_rerun.log`.

**Use archived folds for reproduction.** Regenerating the original splitter under this NumPy/platform produced different assignments for tied group sizes despite identical fold sizes. The explicit `--fold-assignments results/fold_assignments.csv` option restores the original partitions and checks patient/lineage isolation.

The raw rebuild also corrected an overly strict binary-only guard in the portable build script: original mutation counts include 2 and must be retained. The original summed-count filter is preserved. Core model/scoring mathematics are unchanged.

## Exact rerun in the original software environment (15 October)

The submitted package code was rerun on the original processed matrices (SHA-256 hashes re-checked against `provenance/CHECKSUMS.sha256`) for the same seeded 40 targets, all six arms and both schemes, with `--fold-assignments results/fold_assignments.csv`. The software environment matched `results/environment.json` (Python 3.11.15, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0). All 480 S_mean and Spearman values equal the saved values exactly (maximum absolute difference 0). In this environment the seedless splitter also regenerates both archived fold assignments exactly; 799 of 823 patient groups contain one line and three lineage-blocked groups contain 40 lines, which is why tie order matters elsewhere. An earlier rerun on 11 October (lineage and fusion arms, 160 scores, `results/verify_rerun_20261011_pergene.csv`, `logs/verify_rerun_20261011.log`) also matched exactly. See `validation/rerun_20261015_40_gene_exact.csv`, `validation/rerun_20261015_40_gene_exact.json` and `logs/rerun_20261015_40_gene_exact.log`. This remains a 40-target check, not a rerun of all 981 targets or the sensitivities.

## Synthetic example and saved summaries

The included toy example successfully runs actual preparation, alignment and training on 80 artificial cell lines, both five-fold evaluations and six arms, producing 240 finite scores. Grouping, metric references and numerical ridge checks passed (ridge vs scikit-learn maximum difference approximately 5.9e-14). Saved real-result/metadata audit passed 257/257 checks (14 October version). On 15 October the checker and `src/verify_saved_results.py` were changed to test the archived fold assignments directly instead of regenerated folds; the updated checker passes 259/259, and the archived folds have no patient overlap in either scheme and no lineage overlap in S2. Paired summaries recover 109 improvements, 119 declines and 353 small changes at the 0.01 threshold. `src/verified_summary.json` was then regenerated with the updated script, so its fold section describes the archived folds; every reported number is unchanged.

Verification environment: Python 3.14.6, NumPy 2.4.6, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0, matplotlib 3.11.0. Original environment records remain in results/environment.json.

Toy data do not supply biological evidence. Input checksums establish file identity against the supplied records, not independent proof of the portal release label. The original reproduction note records a portal citation-template discrepancy. No clinical efficacy, mechanism or healthy-tissue safety claim follows from these checks.

## Final submission audit

The Desktop deliverables were independently opened and compared with the workspace copies. A fresh extraction of the actual source ZIP successfully executed `python -I run_toy.py`, producing all 240 expected results. ZIP integrity, source syntax, required raw-data links, six toy CSVs, dependency instructions and report sections were checked.

Saved main-run lineage-only and fusion predictions were used to recalculate S_mean for every one of 981 targets under both schemes: 3,924 scores. Maximum absolute discrepancy against the saved score tables was approximately 0.00000201, consistent with rounding of stored float32 predictions. Archived patient and lineage separation checks also passed. See `validation/prediction_score_audit.json`. These are score-reconstruction checks, not 3,924 newly trained models.
