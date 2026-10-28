# Preregistered Study Design

**Project:** How much of multi-omics cancer dependency prediction survives confounder control?
**Course:** CBIO7215 Multimodal AI for Therapeutic Discovery
**Status:** Written and frozen BEFORE any model was fit. Deviations are logged in `deviations.md`.

---

## 1. Discovery problem

Target identification. Genome-wide CRISPR knockout screens (DepMap) measure, for each cancer
cell line, how much its viability depends on each gene. A gene that is *selectively* essential —
lethal in tumors carrying some molecular state, tolerable elsewhere — is a candidate drug target.
Models that predict dependency from multi-omics are promoted as a way to nominate such targets in
tumors that have never been screened.

Two confounders inflate how good these models look:

- **Common essentiality.** ~1,700 of 17,092 target genes are essential in nearly every line.
  For those, predicting a constant is nearly optimal, and any model recovers it.
- **Lineage identity.** Expression is a near-perfect tissue barcode. Under random
  cross-validation a model can identify the lineage of a held-out line and emit that lineage's
  mean dependency, without learning anything cell-line-specific.

## 2. Primary research question

In predicting gene dependency from multi-omics data, how much apparent performance is attributable
to common essentiality and lineage identity, and does multi-omics fusion retain an advantage over a
**lineage-only** baseline once both are controlled?

## 3. Hypotheses

- **H1 (primary).** On selective (non-common-essential) genes, under lineage-blocked
  cross-validation, multi-omics fusion will *not* substantially outperform a lineage-only
  baseline for the majority of genes. The minority that does outperform will be biologically
  coherent rather than arbitrary.
- **H0 / alternative.** Multi-omics retains a substantial advantage over lineage-only across
  most selective genes even under lineage-blocked CV.

H1 is a prediction of a largely negative result. It is stated now so it cannot be revised later.

## 4. Nested nulls — the methodological contribution

Performance is reported against three nested nulls rather than against zero:

| Null | Definition | What it absorbs |
|---|---|---|
| **N0** | Training-set mean of the target gene | Common essentiality, overall difficulty |
| **N1** | Lineage-only ridge (one-hot `OncotreeLineage`) | Everything predictable from tissue of origin |
| **N2** | Lineage-only with lineage labels permuted | Chance performance of the N1 machinery |

The headline quantity is **ΔR² = R²(model) − R²(N1)**, not R² itself.

## 5. Model arms

All arms use an identical preprocessing, feature-selection and hyperparameter budget.

| Arm | Features |
|---|---|
| A | lineage one-hot |
| B | expression |
| C | copy number |
| D | damaging mutation |
| E | expression + CN + mutation (multi-omics fusion) |
| F | E + lineage |

Estimator: ridge regression with alpha chosen by inner CV from a fixed grid. Random forest is run
as a sensitivity analysis, not as the primary estimator.

## 6. Data

DepMap Public 26Q1, downloaded 2026-10-01. Cohort = cell lines with **all four** data types plus
metadata, so every arm sees identical samples: **n = 852 lines, 29 lineages**.

| Matrix | Shape | Notes |
|---|---|---|
| Dependency (Chronos gene effect) | 852 × 17,092 | complete-data genes only |
| Expression (log2 TPM+1) | 852 × 19,204 | non-constant genes |
| Copy number (WGS, relative) | 852 × 19,144 | complete-data genes |
| Damaging mutation | 852 × 1,074 | genes damaged in ≥10 lines |

Inclusion: `IsDefaultEntryForModel` for all omics; complete data per feature; non-constant.
Exclusion: features with any missing value in the cohort; mutation genes damaged in <10 lines.

## 7. Gene strata (fixed before results)

- **Common essential** — listed in `CRISPRInferredCommonEssentials.csv` (1,712 present).
- **Selective** — not common essential, `std(y) ≥ 0.25`, and ≥5 lines with gene effect < −0.5.
- **Flat** — everything else.

**Primary analysis is restricted to the selective stratum.** To bound compute, the primary target
set is the selective genes capped at **N = 1,000** by descending variance; 200 common-essential and
200 flat genes are sampled (seed 0) for the stratum comparison.

## 8. Validation strategy

Random splits do not reflect the discovery use case, so two schemes are run:

- **S1 — patient-grouped 5-fold CV.** Random over lines, but lines sharing a `PatientID` are kept
  in the same fold. This is the conventional setting, with the duplicate-patient leak removed.
- **S2 — lineage-blocked CV (primary).** Whole lineages are held out. This asks the question that
  matters for discovery: predict dependency in a tumor type never screened.

The S1 − S2 gap is itself a reported result.

## 9. Metrics

Per target gene, computed out-of-fold:

- **R²** against the *training* mean (not the test mean), so an uninformative model scores ≈0, and
  a model worse than the mean scores negative.
- **Spearman ρ** between predicted and observed across held-out lines.
- **ΔR²** versus N1 — the primary endpoint.

Aggregates over genes are reported as medians with bootstrap confidence intervals.

## 10. Leakage, confounding and bias register

| Risk | Control |
|---|---|
| Same patient in train and test | Patient-grouped folds in both schemes |
| Lineage leakage through expression | Lineage-blocked CV (S2); lineage-only null (N1) |
| Common essentiality inflating R² | Stratified reporting; primary analysis excludes them |
| Supervised feature selection on full data | Selection fitted inside each training fold only |
| Unsupervised variance filtering | Done on X only, cohort-wide; documented as a residual risk |
| Self-features (gene g predicting its own dependency) | Ablation excluding gene g's own features |
| Test-set centering inflating R² | R² computed against training mean |
| Lineages too small to hold out | Lineages with <10 lines pooled into an "other" block |

## 11. Ablations and sensitivity analyses

1. **Self-feature exclusion** — drop gene g's own expression/CN/mutation from its predictors.
2. **Lineage-permuted null (N2)** — calibrates chance.
3. **Feature budget** — top-k selected features k ∈ {100, 500, 2000}.
4. **Estimator** — random forest versus ridge.
5. **Stratum threshold** — selective definition at std ≥ 0.20 and ≥ 0.30.

## 12. Uncertainty

- Bootstrap over genes (2,000 resamples, seed 0) for all aggregate statistics.
- Per-gene fold-to-fold spread reported alongside point estimates.
- Arms compared **paired** on identical genes and identical folds.

## 13. Decision rules — fixed in advance

- **H1 supported** if median ΔR²(E − A) on selective genes under S2 is < 0.02 **and** its bootstrap
  95% CI contains 0, while the same quantity under S1 is materially larger.
- **H1 weakened** if median ΔR²(E − A) under S2 is > 0.05 with a 95% CI excluding 0.
- **Inconclusive** if the CI is wide and spans both thresholds, or if S1 and S2 do not separate.

## 14. Scope of claims

This is a computational study of a benchmark. Nothing here demonstrates biological or therapeutic
efficacy. Genes that survive confounder control are hypotheses for experimental follow-up, not
validated targets.

## 15. Novelty statement

Lineage confounding and common essentiality are both documented in the DepMap literature; this
project does not claim to discover either. What is new **within this project** is combining them
into a single stratified protocol with explicit nested nulls and lineage-blocked validation, and
quantifying how much each confounder accounts for on a common cohort under matched conditions.
No claim of novelty relative to the published literature is made.
