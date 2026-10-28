# Full raw data and provenance

## Official source

- [DepMap download portal](https://depmap.org/portal/data_page/?tab=allData)
- [Alternative official download page](https://depmap.org/portal/download/all/)
- [Official DepMap Public 26Q1 release announcement](https://forum.depmap.org/t/announcing-the-26q1-release/4606)

The supplied study documentation identifies **DepMap Public 26Q1**. The release
announcement confirms that this public release exists; this does not independently
confirm which exact exports produced the archived experiment scores. The original download hashes are included in `provenance/CHECKSUMS.sha256` and were checked against the six local raw files. A matching hash establishes file identity; identifying the release independently still requires the portal metadata.

Select that release rather than a newer release when attempting to reconstruct
the reported study. The preprocessing code expects the following raw filenames:

| File | Role |
|---|---|
| `CRISPRGeneEffect.csv` | Model-level CRISPR dependency/gene-effect outcomes |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | Expression inputs |
| `OmicsCNGeneWGS.csv` | Copy-number inputs |
| `OmicsSomaticMutationsMatrixDamaging.csv` | Damaging-mutation counts (observed 0/1/2) |
| `Model.csv` | Cell-line, lineage and patient metadata |
| `CRISPRInferredCommonEssentials.csv` | Common-essential target annotation |

Put these downloads in `data/raw/` and follow the README. The parsers expect
`ModelID` for omics/model metadata, `Unnamed: 0` as the gene-effect row-ID
column, and `IsDefaultEntryForModel` to choose the model-level omics entry.
Model metadata must contain `PatientID`, `OncotreeLineage`,
`OncotreePrimaryDisease`, `OncotreeSubtype`, and `DepmapModelType`. Inspect
schema changes before adapting a different release; do not silently merge
duplicate model rows or treat screen-level IDs as model-level IDs.

## Data included in this package

- `data/meta.csv`: the supplied real cohort's metadata, for saved-result checks.
- `results/`: supplied real-data result tables supporting the report.
- `data/toy/raw/`: tiny **synthetic** CSVs in the expected schema, created with
  `generate_toy_data.py`. Identifiers beginning `TOY` are artificial.

The full raw CSVs and processed matrices are not included in this code ZIP because of their size. They were located locally and checked; the processed inputs can be reconstructed from the portal files using the documented commands. The processed matrices are not distributed; their SHA-256 checksums are listed in `provenance/CHECKSUMS.sha256`, so a rebuilt copy can be checked against the originals. The code/results archive supplies original training logs, saved predictions and fold assignments, all retained here.
The synthetic example exists to demonstrate runnable code; its results must
not be interpreted as biological evidence or substituted for the report's
recorded real-data results.
