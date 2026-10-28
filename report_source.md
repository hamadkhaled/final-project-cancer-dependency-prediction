# Predicting Cancer Dependencies with Multimodal Molecular Data

## A paired evaluation across unseen cancer lineages

CBIO7215 – Multimodal AI for Therapeutic Discovery

Fall 2026

Hamad Alhadhrami  
Student ID: 200010169

Instructor: Dr. Jun Wen

<!-- Markdown source of the submitted technical report. Figures are produced by analysis/make_verified_figures.py; the Word/PDF report is the formatted version. -->

# Abstract

Predicting cancer gene dependencies from molecular profiles can help prioritize potential therapeutic targets. This study evaluated whether combining gene expression, copy number, and mutation data improves prediction across unseen cancer lineages. Six ridge-regression models were compared on 852 DepMap cell lines from 29 lineages, with a primary analysis of 581 selective dependency genes and paired, gene-level comparisons on identical cells and test folds. Under lineage-blocked evaluation, expression alone and multi-omics fusion achieved median predictive skill scores of 0.086 and 0.090, respectively, where 0 corresponds to predicting the training-set average. The median paired fusion benefit was approximately zero, but an exploratory analysis identified 42 genes with gains greater than 0.05 and 24 with losses greater than 0.05. An independent 40-gene rerun closely reproduced the recorded scores. These findings show that similar overall performance can conceal substantial differences between genes. Evaluating individual targets and validating their benefits in independent data are therefore essential before using fusion predictions to guide therapeutic discovery.

# 1. Therapeutic discovery problem and research question

Genome-wide perturbation screens reveal genes required for cancer-cell growth or survival. The Cancer Dependency Map provides a basis for identifying molecularly defined vulnerabilities and predicting them from less costly molecular measurements [1]. The therapeutic-discovery problem in this report is cancer target prioritization: which gene dependencies should be investigated in a tumor context that lacks a functional screen? A predicted dependency can nominate an experimental hypothesis, but target tractability, pharmacology, normal-tissue tolerance, and clinical relevance require separate evidence. Published target-prioritization frameworks combine fitness effects with biomarkers and tractability rather than treating prediction accuracy as target validation [3].

The prediction task maps expression, copy number, and damaging-mutation features to a continuous gene effect score. Early fusion concatenates these features before a shared selection and regression pipeline. Extra modalities may encode complementary biology, but may also alter selected predictors or worsen calibration. An average fusion score cannot distinguish these outcomes for individual targets.

**Research question.** Does early multi-omics fusion improve prediction of selective cancer dependencies across unseen lineages, and for which genes does it help or harm? The analysis compares fusion with expression on identical genes and folds, reports absolute predictive skill alongside paired gains, and examines disagreement between squared-error skill and ranking. The original prespecified design emphasized confounder baselines. The revised emphasis on gene-specific fusion gains and losses is explicitly exploratory because it was developed after inspection of the saved results.

## 1.1 Position relative to prior work

Dempster et al. compared molecular data types across cancer vulnerability datasets and found expression highly predictive relative to genomic features [4]. Shi et al. subsequently compared expression-only and multi-omics elastic-net dependency models, included a multi-variable confounder reference, and translated selected models to tumor and healthy-tissue profiles [5]. Their work already addresses expression dominance and confounder baselines. Chronos models cell-population dynamics to infer knockout fitness effects while addressing screen artifacts [2]. Cross-study integration further demonstrates that assay and dataset structure matter [6].

This project therefore does not claim that expression dominance or confounder-aware evaluation is new to the field. Its methodological contribution is a course-level, paired evaluation on a shared cohort: whole-lineage transfer is separated conceptually from within-lineage discrimination; selective targets are reported apart from control strata; and the distribution of fusion benefits is examined instead of relying on its median. The post-hoc finding that some fusion failures preserve ranking motivates a testable calibration question. Matched nonlinear models and within-lineage scoring are needed to test whether these patterns persist.

## 1.2 Why the validation unit matters

Patient-grouped evaluation asks whether a model predicts new patients from a broadly familiar tissue distribution. Holding out complete lineages asks whether learned associations transfer to tissues absent from training. Neither setting is a direct simulation of patient-tumor deployment: cell lines omit aspects of the microenvironment and differ from intact tumors [5]. Keeping these deployment questions distinct prevents a single score from carrying several unsupported interpretations.

## 1.3 Major challenges and the selected focus

Major challenges include CRISPR measurement artifacts and cross-study batch effects [2,6]; many inputs relative to samples and uneven cancer-type coverage; tissue-related confounding and transfer to new tumors [5]; redundant or noisy modalities; and separating prediction from causal biology, druggability and healthy-tissue safety [3,5]. This project focuses on unseen-lineage transfer and gene-specific fusion benefits. Screening correction and clinical validation remain separate tasks.

# 2. Data and analysis design

The recorded source is DepMap Public 26Q1. The cohort was constructed as the intersection of models with dependency, expression, copy-number, damaging-mutation, and metadata records. Default profiling entries were retained. Columns with missing values were excluded from every matrix, and constant expression and copy-number features were removed. Mutation features were retained when their summed damaging-mutation count was at least ten. All six raw inputs and the six processed analysis files used for every reported result match recorded checksums; fresh preprocessing reproduced the processed files byte for byte.

| **Data component** | **Recorded dimensions or count** | **Representation**                  |
|--------------------|----------------------------------|-------------------------------------|
| Shared cohort      | 852 cell lines                   | 29 lineage labels and 823 patients  |
| Dependency         | 852 × 17,092                     | Chronos gene effect                 |
| Expression         | 852 × 19,204                     | log2 TPM plus 1                     |
| Copy number        | 852 × 19,144                     | Relative copy-number ratio          |
| Damaging mutation  | 852 × 1,074                      | Damaging-mutation count (0, 1 or 2) |

**Table 1.** Cohort and feature specification. Metadata independently confirms the sample, lineage, and patient counts. Matrix dimensions and preprocessing were checked against the archived inputs and a fresh raw-data rebuild. More negative gene effect denotes stronger dependency; the conventional anchors are approximately zero for no effect and minus one for the median common-essential effect.

## 2.1 A fixed target benchmark

The selective panel excludes genes on the common-essential list, requires cohort-wide gene-effect standard deviation of at least 0.25, and requires at least five lines with gene effect below −0.5. All 581 qualifying genes were included because the planned 1,000-gene cap did not bind. The recorded full target universe contained 1,712 common-essential genes and 14,799 remaining flat or nonselective genes, which fail at least one selective criterion. Seeded samples of 200 from each comparison stratum produced 981 evaluated targets.

This target panel is fixed using outcomes from the entire cohort. That is a valid way to describe a benchmark, but performance is conditional on the selected panel. It is not a prospective evaluation of a target-selection procedure trained without test outcomes. A panel frozen from an earlier release or a training-only selection sensitivity would strengthen prospective claims. Common-essential status is also not a direct measurement of normal-tissue toxicity; the focus on selective genes narrows the question rather than establishing that other genes can never be useful targets.

## 2.2 Prespecified and exploratory analyses

The design file (provenance/design.md) states that the protocol was frozen before fitting. No independently time-stamped registration accompanies the archive, so this report uses the term prespecified protocol. Its primary hypothesis, H1, predicted that for selective genes under lineage-blocked validation, fusion would not substantially outperform the lineage-only baseline for most genes. The six arms, target strata, two outer validation schemes, self-feature removal, feature-budget sensitivity, permutation control, and estimator sensitivity appear in that protocol. The saved implementation changed penalty selection to generalized cross-validation and narrowed the feature-budget grid (provenance/deviations.md).

The new fusion difference distribution and ranking-versus-skill case analysis are post-hoc reanalyses of saved results. Full-cohort biological associations, copy-number transformation, and additional low-dimensional mutation sensitivities are exploratory. They generate follow-up questions and are not presented as independent confirmation of a hypothesis selected beforehand. The planned selectivity sensitivity at standard deviation 0.20 is absent; stricter thresholds were evaluated as subsets of the fixed panel (Section 7).

# 3. Models, validation and scoring

Six arms share the same outer samples. Arm A uses one-hot lineage labels; B uses expression; C uses copy number; D uses damaging mutation; E concatenates B, C, and D; F adds lineage to E. For each target and training fold, omics features are standardized with training statistics and the top 300 eligible features are selected by absolute training-set Pearson correlation with the response. The lineage columns are appended where specified. Feature counts are matched across omics arms, but feature-selection outcomes differ.

Ridge penalties are chosen by generalized cross-validation over 25 logarithmically spaced values from 0.1 to 100,000. Selection is refitted in each outer training fold. However, the GCV objective is applied after supervised screening and does not refit the entire screening procedure in grouped inner folds. It therefore does not reproduce the unseen-lineage deployment task during tuning. Cohort-wide missingness, variance, and mutation-frequency filtering also use information about test covariates, although they do not use test response values for feature screening.

## 3.1 Outer partitions and reference models

S1 uses five patient-grouped folds, assigning larger groups to smaller folds. It is not a random split. S2 uses five lineage-blocked folds. Lineages with fewer than ten lines share an Other group, which is held out together. The archived fold assignments have no patient overlap in either scheme and no lineage overlap in S2. S1 test-fold sizes were 171, 171, 170, 170, and 170; S2 sizes were 175, 169, 170, 164, and 174. Pooling rare tissues reduces resolution and creates a heterogeneous block, but does not introduce tissue overlap. The splitter uses no random seed, but 799 of 823 patient groups contain one line and three S2 blocks contain 40 lines, so the order of tied groups depends on the sorting implementation. It regenerates the archived folds in the original software environment, but a second environment produced different assignments; exact reproduction therefore uses the archived fold assignments.

The training mean is a deployable baseline. The lineage-only model is a baseline with learned tissue effects in S1. A lineage-permuted fit is a control for the lineage-model machinery. These are not increasingly strong nested null models. In S2, the effect of a test lineage cannot be learned from its one-hot column because that lineage is absent from training. The resulting prediction is approximately constant, with small shifts induced by centering and ridge regularization. Its score near zero does not validate the removal of lineage-associated expression patterns.

## 3.2 Training mean predictive skill

For each gene, the score sums squared held-out errors over all test folds. Each fold's reference is that gene's training mean. The primary predictive skill is

$$S_{mean,g} = 1 - \frac{\sum_{f}^{}{\sum_{i \in T_{f}}^{}{(y_{i,g} - {\widehat{y}}_{i,g})}^{2}}}{\sum_{f}^{}{\sum_{i \in T_{f}}^{}{(y_{i,g} - {\overline{y}}_{train,f,g})}^{2}}}$$

Zero matches the fold-specific training-mean predictor; negative values are worse. This score is called S_(mean) throughout because it is not conventional test-set R². For a single fold,

$$\sum_{i \in T_{f}}^{}{(y_{i} - {\overline{y}}_{train})}^{2} = \sum_{i \in T_{f}}^{}{(y_{i} - {\overline{y}}_{test})}^{2} + n_{f}{({\overline{y}}_{test} - {\overline{y}}_{train})}^{2}$$

The training-mean denominator is at least as large as the fold test-mean denominator. For identical predictions, S_(mean) is therefore at least as high as the corresponding test-mean-normalized score. Test means may be used to define an evaluation statistic without being supplied to a trained predictor; that is not itself training leakage.

Per-gene Spearman assesses ranking across held-out samples, not calibration, and can reflect between-tissue differences. Medians and paired differences use 2,000 percentile gene-bootstrap resamples, seed zero. Intervals are conditional on this panel, fits, cohort and folds; they exclude uncertainty over new patients, new lineages or retraining. Gene dependence further limits population interpretation.

# 4. Results of the matched model comparison

Both expression and early fusion retained modest positive median predictive skill under S2. Of the 581 selective genes, 468 had positive fusion skill and 113 had negative skill. The lineage-only median approached zero, as expected from the unavailable test-lineage coefficients. Standalone copy number and damaging mutation had negative median skill under the implemented representation and estimator.

| **Arm**               | **S1 median S_mean**      | **S2 median S_mean**      | **S2 median Spearman** |
|-----------------------|---------------------------|---------------------------|------------------------|
| A Lineage only        | 0.050 [0.045, 0.054]    | 0.002 [0.001, 0.002]    | −0.116                 |
| B Expression          | 0.133 [0.125, 0.144]    | 0.086 [0.080, 0.098]    | 0.300                  |
| C Copy number         | 0.021 [0.016, 0.027]    | −0.008 [−0.014, −0.005] | 0.126                  |
| D Damaging mutation   | −0.060 [−0.064, −0.055] | −0.110 [−0.121, −0.105] | 0.009                  |
| E Early fusion        | 0.133 [0.123, 0.141]    | 0.090 [0.079, 0.100]    | 0.300                  |
| F Fusion plus lineage | 0.133 [0.123, 0.141]    | 0.089 [0.079, 0.098]    | 0.300                  |

**Table 2.** Selective-gene results on identical outer partitions. Brackets denote 95% percentile gene-bootstrap intervals. S1 is patient-grouped; S2 is lineage-blocked. Spearman values are rounded summaries of saved per-gene correlations.

## 4.1 Interpretation of the original endpoint

The prespecified decision rule labels H1 weakened if the S2 paired median fusion-minus-lineage skill exceeds 0.05 and its gene-bootstrap interval excludes zero. The saved result, 0.086 [0.077, 0.097], meets that operational rule. It does not constitute a formal hypothesis rejection, and the baseline degeneracy prevents the originally intended interpretation as prediction beyond learned tissue effects. The more defensible result is that omics models can outperform an almost constant reference when predicting held-out tissues on this benchmark.

Fusion's paired median S2-minus-S1 change was −0.035 [−0.039, −0.031]; 83.5% of selective genes declined. This demonstrates sensitivity to the partitioning task. Dividing the two fusion medians produces a descriptive ratio of approximately 0.67, but the scores use different training means and test contexts. That ratio is not a fraction of biological signal shown to be free from lineage effects.

## 4.2 Why positive transfer skill is an incomplete discovery criterion

Out-of-fold per-gene predictions from S2 span multiple held-out tissues. They can score well by capturing tissue-level differences even without ranking models within each tissue. Expression may transfer such tissue programs across related lineages. The present results establish predictive transfer on the fixed benchmark, while the stronger claim of within-lineage discrimination remains untested. Target nomination should examine both the error-based score and ranking at the unit relevant to the intended experiment.

The S2 standalone mutation arm has positive skill for six genes, with WRN its strongest example at approximately 0.115. Its poor median is therefore a pipeline-level result, not evidence that damaging mutation is uninformative for every dependency or that genomic information can never help fusion.

# 5. Fusion helps and harms different genes

The paired median fusion-minus-expression difference under S2 was −3.7 × 10⁻⁸, with a gene-bootstrap interval from −1.1 × 10⁻⁷ to 3.5 × 10⁻⁸. This is practically zero. It does not establish equivalence for individual genes. In the new post-hoc analysis, 42 of 581 genes gained more than 0.05 skill and 24 lost more than 0.05. At the 0.01 threshold, 109 improved, 119 worsened, and 353 differed by no more than 0.01. These are effect-size categories, not gene-level significance tests.

**Figure 1.** Gene-specific early-fusion benefits under lineage-blocked validation. Each difference compares fusion and expression on the same gene and outer folds. Thresholds highlight descriptive changes; no per-gene significance claim is implied. The plot is a new analysis of archived gene-level scores rather than a new training run.

| **Gene** | **Expression S_mean** | **Fusion S_mean** | **Paired change** | **Fusion Spearman** |
|----------|-----------------------|-------------------|-------------------|---------------------|
| WRN      | −0.661                | 0.148             | +0.809            | 0.160               |
| NDE1     | −0.104                | 0.119             | +0.223            | 0.350               |
| EPC1     | −1.323                | −0.015            | +1.308            | 0.183               |
| IGF1R    | 0.055                 | −0.313            | −0.368            | 0.280               |
| VPS4A    | 0.205                 | −0.042            | −0.247            | 0.446               |

**Table 3.** Exploratory examples selected to illustrate improvement, residual failure, and harm. Absolute skill must accompany the paired change: EPC1 has the largest gain among these cases but still fails to outperform the mean baseline.

WRN is a useful therapeutic reference because prior experiments established selective WRN essentiality in microsatellite-unstable cancer models [7]. The standalone mutation arm also has positive skill for WRN (Section 4.2), but the present prediction gain does not establish recovery of that mechanism. Added modalities, altered feature competition, or a changed penalty may produce the gain; a modality-addition ablation and MSI-aware held-out analysis would distinguish these possibilities. IGF1R and VPS4A illustrate a different concern: fusion loses positive expression-based skill even though rank correlation rises slightly, from 0.262 to 0.280 and from 0.436 to 0.446. Raw predictions are needed to test whether this reflects calibration shifts or extreme errors.

# 6. Scores and biological associations require different interpretations

The S2 median fusion skill was −0.021 for the 200 common-essential controls and −0.161 for the 200 flat controls, compared with 0.090 for selective genes. Fusion had negative skill for 113 common-essential and 181 flat genes, rather than every gene in those strata. A negative median indicates a typical loss against the mean predictor, not a universal property. Low dependency variation can make error-based prediction difficult and emphasizes why the target population must accompany each score. Among the 113 selective genes with negative fusion skill, 108 still had positive Spearman correlation, so positive ranking alone does not establish squared-error value.

The original pooled-correlation comparison changed both aggregation and gene population, using 981 pooled targets versus 581 selective per-gene targets. It is excluded because the target panels differ; a matched-panel analysis of the archived predictions remains a follow-up. Per-gene scores address a target's behavior across models; pooled correlation may instead reflect differences between gene means.

**Figure 2.** Predictive skill and ranking for the same 581 selective targets under S2 fusion. Both are per-gene quantities. Highlighted examples are exploratory; disagreement between the axes motivates further analysis rather than assigning a biological mechanism from the score pattern alone.

## 6.1 Associations generate biological hypotheses

SOX10 had fusion skill 0.651 with Spearman correlation 0.197, whereas FAM50A had 0.516 and 0.723. High error-based skill with weaker ranking can arise when a small set of large responses dominates error reduction. High values for both metrics can be compatible with graded predictability. Confirming the explanation for either gene requires prediction plots and evaluation within held-out tissues.

In a full-cohort exploratory screen, FAM50B was the strongest expression correlate of FAM50A dependency, with correlation approximately 0.78. DDX3Y was the corresponding feature for DDX3X, with correlation approximately 0.72. Higher expression accompanies less negative dependency, consistent with buffering. This does not establish ridge-model reliance or causation. Sex and tissue structure are alternative explanations for the DDX3X and DDX3Y association.

The original symbol-root rule is a naming heuristic, so its enrichment claim is not retained as biological evidence. A follow-up should use versioned Ensembl Compara relationships inferred from duplication events [8], training-fold feature selection, and an annotated small-feature model evaluated against the full predictor on identical held-out samples. Perturbation evidence is needed to establish buffering.

# 7. Sensitivity analyses constrain the fusion conclusion

The archived sensitivities compare genes shared with the main run under S2; the permutation control is also shown under S1, where lineage effects can be learned. They are summarized as paired skill changes. Except for the 200-gene random-forest experiment, the listed comparisons use all 581 selective genes. Gene-bootstrap intervals have the conditional interpretation stated in Section 3. Under S1, permuting lineage labels reduced the lineage-only median from 0.050 to −0.0007, so this baseline learns tissue effects when test lineages appear in training. Restricting the fixed panel to stricter selectivity thresholds, without retraining, raised the S2 fusion-minus-lineage median to 0.128 [0.119, 0.140] at standard deviation 0.30 (300 genes) and 0.155 [0.134, 0.182] at 0.35 (149 genes); the fusion-minus-expression median remained approximately zero in both subsets.

| **Comparison with main specification**     | **Median paired change** | **95% interval**       |
|--------------------------------------------|--------------------------|------------------------|
| Fusion without the target's own features   | Approximately 0          | Approximately [0, 0] |
| Fusion top 100 versus top 300 features     | −0.0023                  | [−0.0046, 0.0005]    |
| Fusion top 600 versus top 300 features     | +0.0012                  | [−0.0011, 0.0039]    |
| Permuted versus true lineage baseline, S2  | −0.0019                  | [−0.0023, −0.0016]   |
| Permuted versus true lineage baseline, S1  | −0.0520                  | [−0.0554, −0.0476]   |
| Random-forest fusion versus ridge fusion   | +0.0198                  | [0.0137, 0.0288]     |
| Log-transformed versus raw copy-number arm | +0.0085                  | [0.0069, 0.0096]     |
| Fusion with log-transformed copy number    | Approximately 0          | Approximately [0, 0] |
| Mutation top 20 versus top 300 features    | +0.0236                  | [0.0134, 0.0315]     |

**Table 4.** Saved sensitivity results recomputed on matched genes. Values rounded to four decimals can appear exactly zero despite small floating-point changes. These sensitivities restrict conclusions to the tested pipeline rather than excluding all alternative explanations.

## 7.1 Estimator and preprocessing matter

On the matched 200-gene subset, median random-forest fusion skill was 0.106 and ridge fusion skill was 0.091. The median of their paired differences was 0.020, a distinct statistic from the difference of their medians. Random forest improved 148 of the 200 genes. Its recorded configuration used 100 trees, minimum leaf size five, maximum feature fraction 0.3, and seed zero. The archive contains only lineage and fusion random-forest arms. It therefore cannot answer whether nonlinear fusion beats nonlinear expression. The result measures an improvement within this sensitivity, not a known fraction of all attainable predictive signal.

Log-transforming copy number improved the standalone arm toward zero but did not materially change the fusion median. The primary feature-selection audit found 85.1% expression, 14.2% copy number, and 0.7% mutation features. This was counted across the 581 selective targets in the first S2 training fold only. Selection counts do not measure model reliance or the fraction of predictive signal supplied by each modality.

## 7.2 Negative findings remain useful

Reducing the mutation budget to 20 improved its median to approximately −0.086, still below the mean baseline. This does not rule out overfitting, encoding limitations, unstable selection, or insufficient regularization. The implementation covers damaging mutations above a frequency cutoff, not all mutation information. Similarly, the near-zero fusion median gain is evidence about ridge with global correlation screening. It does not establish that other fusion architectures or interactions are ineffective. The appropriate follow-up is a matched estimator comparison and targeted modality additions, not a more categorical interpretation of these negative medians.

# 8. Discussion and next experiments

The near-zero median fusion advantage coexists with appreciable target-specific gains and losses. Selecting a representation from that median alone could discard useful genomic context for some targets or introduce error for others. Conversely, a large improvement can rescue a failed predictor without producing positive skill, as EPC1 illustrates. Both paired change and absolute skill are required.

Lineage blocking changes the deployment question while leaving tissue programs in molecular features. A reference model unable to represent a novel category cannot quantify the omics model's use of those programs. The current evidence supports transfer to a fixed panel of unseen lineages. Within-lineage scoring and a broader confounder reference are needed to assess discrimination within a tissue.

## 8.1 Prioritized tests

First, compare expression and fusion using the same nonlinear estimator, target panel, folds, tuning budget, and training-local screening. A grouped inner validation should refit the complete selector and model. Report paired S_(mean) and Spearman differences, the number of targets crossing zero skill, and calibration-sensitive errors. This tests whether the zero-median ridge finding is a limitation of linear early fusion.

Second, compare expression with expression plus copy number and expression plus mutation on the same design. Include a modality-balanced early selector at a fixed total budget and a simple late-fusion reference whose weights are fitted inside training data. WRN, NDE1, IGF1R, and VPS4A provide concrete follow-up cases, but any case-specific choices must be labeled exploratory. Record selected-feature stability across all training folds and use held-out ablations to assess model reliance.

Third, report within-lineage metrics for tissues with adequate sample size and target variation. An evaluation-only centering of observations and predictions within each held-out lineage can separate between-tissue structure from within-tissue discrimination; no test outcomes should enter the predictor. Summaries should show lineage-specific results and repeated lineage-grouped partitions. Paired patient- or lineage-level resampling would better address the deployment claim, with explicit reporting of whether models are refitted.

Fourth, test the biological hypotheses with externally annotated paralogs, training-fold association selection, and a small interpretable predictor on identical held-out samples. Audit sex and lineage as alternative explanations. Use experimental perturbations to test the proposed mechanism before nominating an association as causal evidence.

## 8.2 Limits of the current evidence

The study uses one recorded release and a complete-modality intersection that may favor well-characterized models. The panel is selected globally, each scheme uses a single fixed partition rather than repeated partitions, and gene-level multiplicity is not addressed with hypothesis tests. The original inputs, training logs, and lineage/fusion predictions are archived and checksum-verified. Reruns of 40 targets reproduced the saved scores exactly in the original software environment and closely in a second environment; the full 981-target experiment and the sensitivities were not rerun. Within-lineage scoring and a complete feature-reliance audit remain follow-up analyses.

## 8.3 Conclusion

Expression and early fusion show modest transfer skill for selective cancer dependencies on this fixed benchmark. Fusion has essentially no median advantage over expression, while a subset of genes gains and another subset loses substantially. The methodological insight is to evaluate absolute accuracy, paired modality benefit, and ranking together at the relevant deployment unit. These results prioritize follow-up experiments; they do not establish tissue-independent mechanisms or validated therapeutic targets.

# References

[1] Tsherniak A, Vazquez F, Montgomery PG, et al. Defining a Cancer Dependency Map. Cell. 2017;170:564-576.e16. <https://doi.org/10.1016/j.cell.2017.06.010>

[2] Dempster JM, Boyle I, Vazquez F, et al. Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects. Genome Biology. 2021;22:343. <https://doi.org/10.1186/s13059-021-02540-7>

[3] Behan FM, Iorio F, Picco G, et al. Prioritization of cancer therapeutic targets using CRISPR–Cas9 screens. Nature. 2019;568:511-516. <https://doi.org/10.1038/s41586-019-1103-9>

[4] Dempster JM, Krill-Burger JM, McFarland JM, et al. Gene expression has more power for predicting in vitro cancer cell vulnerabilities than genomics. bioRxiv preprint. 2020. <https://doi.org/10.1101/2020.02.21.959627>

[5] Shi X, Gekas C, Verduzco D, et al. Building a translational cancer dependency map for The Cancer Genome Atlas. Nature Cancer. 2024;5:1176-1194. <https://doi.org/10.1038/s43018-024-00789-y>

[6] Pacini C, Dempster JM, Boyle I, et al. Integrated cross-study datasets of genetic dependencies in cancer. Nature Communications. 2021;12:1661. <https://doi.org/10.1038/s41467-021-21898-7>

[7] Chan EM, Shibue T, McFarland JM, et al. WRN helicase is a synthetic lethal target in microsatellite unstable cancers. Nature. 2019;568:551-556. <https://doi.org/10.1038/s41586-019-1102-x>

[8] Ensembl. Homology types. Documentation of Compara evolutionary relationships. Accessed 11 October 2026. <https://mart.ensembl.org/info/genome/compara/homology_types.html>

# Appendix Reproducibility and analysis record

## A.1 What is reproducible from the archive

The main result table contains 11,772 rows covering two schemes, six arms, and 981 targets, with no duplicated scheme-arm-gene records. The revision independently recomputed medians, paired differences, gene-bootstrap intervals, effect-size counts, case comparisons, and sensitivity summaries from the archived per-gene tables. It checked both archived outer partitions against the metadata, verifying zero patient overlap in either scheme and zero lineage overlap in S2; in the original software environment, the splitter regenerates both partitions exactly.

The reanalysis script, machine-readable summary, regenerated figures, and input table hashes accompany the revised package. CSV fields named r2 are retained in source outputs for compatibility, but represent S_(mean) as defined in this report. A gene-bootstrap interval is a panel-conditional summary. It does not provide a gene-specific error bar, and the archive does not include per-gene fold scores sufficient to reconstruct all planned fold-spread summaries.

## A.2 Archived inputs and verification reruns

The aligned Y.npy, expr.npy, cn.npy, mut.npy, names.json, and meta.csv files were produced from the six raw CSVs on 1 October 2026 and underlie every reported result. All twelve files match recorded SHA-256 checksums, and a fresh raw-data rebuild reproduced all six processed files byte for byte. The complete code/results archive also supplies training logs, explicit folds, target indices, environment records, and lineage/fusion predictions. The portal is recorded as DepMap Public 26Q1; an original citation-template discrepancy remains documented. Checksums identify the files but do not independently prove the release label.

Three reruns used the original matrices and the same seeded sample of 40 targets. On 11 October, the lineage and fusion arms (160 scores) matched the saved scores exactly. On 14 October, all six arms under both schemes (480 scores) were retrained in a second software environment with the archived folds. Maximum absolute differences were 0.00000290 for S_(mean) and 0.00131 for Spearman; the largest rank discrepancy was in the nearly constant lineage-only baseline. On 15 October, the submitted code was rerun for the same 480 scores with the archived folds in the original software environment (results/environment.json) and matched the saved scores exactly. These checks cover 40 of 981 targets; the full experiment and sensitivities were not rerun. The source ZIP includes a runnable synthetic example, checksum records, logs, verification tables and full-data download/rebuild instructions. Future runs should save expression predictions as well.

## A.3 Changes from the recorded protocol

The recorded deviations narrow the top-feature sensitivity from 100, 500, and 2,000 to 100 and 600 around a main budget of 300; note the inactive selective cap; and replace inner cross-validation with GCV. Patient grouping already appears in the design and is not described here as an unplanned control. The design describes S1 as random over lines; the implemented S1 assignment uses the seedless greedy splitter instead (Section 3.1). Rare-lineage pooling preserves strict blocking. Stricter selectivity thresholds subset existing fits and do not retrain models; the planned looser threshold of 0.20 was not available.

The present report additionally corrects the training-mean metric interpretation, labels baselines and the permutation control separately, limits the hypothesis conclusion to operational weakening, and replaces an aggregate equivalence claim with a post-hoc fusion heterogeneity analysis. It removes unmatched pooled-metric evidence and quantitative claims based on the gene-symbol homology heuristic. Full-cohort expression associations are retained as hypothesis generation. Nonlinear expression-versus-fusion comparison, within-lineage metrics, repeated partitions, annotation-based biology, and full-cohort retraining beyond the verification sample remain future analyses.
