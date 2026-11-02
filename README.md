# Overlapping community detection on fused patient-similarity graphs

Locked multi-cohort evaluation of hard and soft community detection on breast-cancer expression graphs. The lead is a pre-specified policy. Classical BIGCLAM is an a priori baseline and cannot become the lead, even when its LumA odds ratio, IntClust enrichment, or simulation correlation is higher.

- Discovery: GSE96058 / SCAN-B (n=3,273)
- Validation: TCGA-BRCA (n=1,093)
- External survival: METABRIC (n=1,980)
- Treatment-response external (added later; null): GSE25066 (n=508)

Manuscript: `manuscript/manuscript.tex` (and `manuscript/manuscript.docx`). Locked hyperparameters: `manuscript/SUPPLEMENTARY_METHODS.md`. Bake-off notes: `docs/STUDY_NOTES.md`. Ceiling call: `results/ceiling/decision_note.md`.

Do **not** retune `K`, `k`, mixed ratios, epochs, restarts, or the gene list.

## What a reviewer can do without re-fitting

Clone the repository. Published tables and decision notes under `Results/` and `manuscript/tables/` are tracked. Expression matrices are **not** in git (they exceed GitHub size limits). Inspecting those CSV/JSON/Markdown files is enough to check the reported numbers.

```bash
git clone https://github.com/saeedsamimi995/snf-overlap-evaluation.git
cd snf-overlap-evaluation
```

## Full re-run (reproduce the analysis)

### 1. Environment

```bash
conda env create -f environment.yml
conda activate snf-dual-output
pip install -r requirements.txt
```

Python 3.11. A CPU PyTorch build works; a GPU speeds NOCD only. Disk: several GB free for SCAN-B plus the other cohorts. RAM: plan on ≥16 GB for the discovery matrix.

### 2. Place public data under `data/`

Edit only the file paths in the `q1:` block of `config/config.yaml` if your filenames differ. Defaults expect:

```text
data/
  GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv
  gse96058_clinical.csv
  Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct
  brca_tcga_clinical_data.tsv
  tcga/brca_pancan_clinical_patient.txt      # auto-downloaded if missing
  metabric/data_mrna_illumina_microarray.txt
  metabric/data_clinical_patient.txt
  metabric/data_clinical_sample.txt
  gse25066/GSE25066_series_matrix.txt.gz
  gse25066/GPL96.annot.gz
```

Sources (each under its own access terms):

| Cohort | Typical source |
|--------|----------------|
| GSE96058 | GEO / SCAN-B release; expression as genes × samples CSV. Clinical must be wide (samples as columns) with rows that include `PAM50`, `OS_time`, `OS_event`, and preferably `age`. |
| TCGA-BRCA | Broad Firehose RSEM log2 CCT + a TCGA clinical TSV. PanCancer `SUBTYPE` is fetched automatically if the local PAM50 file is missing. |
| METABRIC | cBioPortal BRCA METABRIC microarray + clinical patient/sample tables. |
| GSE25066 | GEO series matrix and GPL96 annotation (Hatzis et al., JAMA 2011). Expression is already ~log; do not re-log. |

`data/harmonized/` is created by the first command below and is gitignored.

### 3. Run commands in order

From the project root, with the conda env active:

```bash
# A. Lock the top-2,000 discovery genes, write data/harmonized/, fit hard partitions
python run_q1_study.py --config config/config.yaml

# B. Classical BIGCLAM baseline on the fused SNF graph
python run_snf_bigclam_study.py \
  --harmonized-dir data/harmonized \
  --results-dir results/snf_bigclam

# C. Soft bake-off (NOCD-X, NOCD-G, SLPA, CESNA-lite, BIGCLAM)
python run_snf_overlap_sota.py \
  --harmonized-dir data/harmonized \
  --results-dir results/snf_overlap_sota \
  --reuse-bigclam-dir results/snf_bigclam
python scripts/rank_overlap_sota.py

# D. Ceiling / boundary checks (do not reverse the lead)
python -m scripts.run_ceiling_experiments

# E. GSE25066 pCR tests (null; no treatment-resistance claim)
python -m src.analysis.gse25066_response
```

Optional earlier soft screen (not the locked lead table):

```bash
python run_snf_soft_bakeoff.py \
  --harmonized-dir data/harmonized \
  --results-dir results/snf_soft \
  --reuse-bigclam-dir results/snf_bigclam
```

Step A writes `results/q1/locked_config.json`. Steps B–D refuse to invent settings: they load that lock and `data/harmonized/`. Step E needs the two GSE25066 GEO files and the discovery gene list under `data/harmonized/gene_list.txt`.

### 4. Where to look after a run

| Path | Contents |
|------|----------|
| `data/harmonized/` | Locked expression matrices and clinical sidecars |
| `results/q1/` | Hard partitions, `locked_config.json`, `stability.csv` |
| `results/snf_bigclam/` | BIGCLAM baseline on fused affinity |
| `results/snf_overlap_sota/` | Soft bake-off assignments and ranking inputs |
| `results/ceiling/` | Density sweep, view ablation, simulation, `decision_note.md` |
| `results/gse25066/` | `pcr_tests.csv`, `assignments.csv`, gene-overlap sensitivity |
| `manuscript/tables/` | Paper tables copied for the manuscript |

### Common failures

- **FileNotFoundError on prepare** — a path in `config/config.yaml` still points at a missing file. Keep paths relative to the repo root under `data/`.
- **Missing `locked_config.json`** — run step A before B–D.
- **Missing `data/harmonized`** — same; step A must finish `prepare`.
- **GSE25066 import errors** — place both `.gz` files under `data/gse25066/` with the exact names above.
- **Out of memory on SCAN-B** — the transformed GSE96058 matrix is ~1.8 GB on disk; use a machine with enough RAM.

## Layout

```text
config/config.yaml              paths + locked q1 settings
src/analysis/                   SNF, soft engines, ceiling, GSE25066
src/preprocessing/              cohort loaders and gene-list lock
src/bigclam/                    classical BIGCLAM baseline
scripts/                        ranking and ceiling driver
manuscript/                     tex, docx, figures, tables
results/                        tracked study outputs (matrices not included)
docs/STUDY_NOTES.md             bake-off and related-work notes
```

## Citation

Code: https://github.com/saeedsamimi995/snf-overlap-evaluation

Samimi S. Overlapping community detection on fused patient-similarity graphs: a locked multi-cohort evaluation with a pre-specified lead policy. 2026.

Yang J, Leskovec J. Overlapping community detection at scale. WSDM. 2013:587–596.  
Wang B, et al. Similarity network fusion. Nat Methods. 2014;11:333–337.

## License

MIT. See `LICENSE`.
