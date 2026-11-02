# Study notes

One note for the locked evaluation. The manuscript is the result. This file keeps the bake-off, the earlier soft screen, the comparison set, and the commands that used to live in separate notes.

## What the paper is

Hard subtypes come from multi-view SNF (Euclidean and correlation kNN, k=15, T=15) and a spectral cut (K=5, assign_labels="kmeans", seed 42). Soft engines run on the symmetrized kNN of that fused graph: NOCD-X, NOCD-G, SLPA, CESNA-lite, and classical BIGCLAM.

BIGCLAM is `baseline_comparison`. It is not lead-eligible. Quality filters on METABRIC are mixed fraction in [0.10, 0.40], median second/first affiliation below 0.85, and LumA odds ratio above 1. NOCD-X is the lead because it is the only eligible engine that passes those filters.

Do not claim that BIGCLAM is the best clusterer, that mixed patients are Claudin-low, that the GSE25066 null is a treatment-resistance result, or that classifier accuracy on BIGCLAM labels is evidence.

## Locked soft bake-off (METABRIC)

Source: `run_snf_overlap_sota.py`, ranked by `scripts/rank_overlap_sota.py`.

| Method | Own hard+mixed p | Hard Cox p | LumA OR | Mixed | Peak | Role |
|--------|-----------------:|-----------:|--------:|------:|-----:|------|
| SLPA | 0.0055 | 0.083 | 0.44 | 69% | 1.00 | Fails quality filters |
| BIGCLAM | 0.036 | 0.022 | 3.00 | 25% | 0.07 | Passes filters; baseline, not the lead |
| CESNA-lite | 0.114 | 0.135 | 0.53 | 20% | 0.33 | LumA depleted |
| NOCD-G | 0.183 | 0.058 | 0.67 | 19% | 0.74 | Graph features only |
| NOCD-X | 0.417 | 7e-6 | 1.83 | 17% | 0.20 | Policy lead |

BIGCLAM's residual p is about 0.036 and its BH q in the five-engine family is about 0.091, so that residual does not survive FDR. Hard SNF on METABRIC separates survival after age and PAM50 (p about 1.7e-4; BH q about 8.3e-4 within five hard methods). Bootstrap 95% intervals for the nested concordance gain include zero on GSE96058, TCGA, and METABRIC.

```bash
python run_snf_overlap_sota.py --harmonized-dir data/harmonized --results-dir results/snf_overlap_sota --reuse-bigclam-dir results/snf_bigclam
python scripts/rank_overlap_sota.py
```

## Ceiling checks

These do not reverse the lead. Numbers are in `results/ceiling/decision_note.md`.

- Simulation (n=300): mean correlation with planted soft membership is 0.53 (BIGCLAM), 0.17 (SLPA), 0.10 (graph NOCD). Mixed-label ARI is below 0.10 for every engine.
- IntClust enrichment is stronger for BIGCLAM (IntClust 8 OR 3.24) than for NOCD-X.
- View ablation (`results/ceiling/view_ablation.csv`), reduced budget: BIGCLAM LumA OR is 3.00 fused, 1.35 Euclidean, 0.91 correlation. NOCD-X on the same fits is 0.97, 0.86, and 0.86.
- SLPA peak ratio stays 1.0 while mixed fraction rises from 0.51 to 0.83 as k falls from 30 to 5.
- GSE25066 pCR is null in LumA (5 events) and in the hard cluster with the most mixed patients (16 events).
- SCAN-B programme contrasts for BIGCLAM hard cluster 0 are in `results/ceiling/program_contrasts.csv` (estrogen up, HER2 down; proliferation does not repeat the METABRIC direction).

```bash
python -m scripts.run_ceiling_experiments
python -m src.analysis.gse25066_response
```

## Earlier soft screen

`run_snf_soft_bakeoff.py` compared prototype, NMF, GMM, fuzzy c-means, and BIGCLAM on the same fused affinity, before the overlapping-engine policy above. It is not the locked lead table. On that screen BIGCLAM had the strongest residual (p about 0.036, LumA OR about 3) and GMM collapsed toward a hard cut.

```bash
python run_snf_soft_bakeoff.py --harmonized-dir data/harmonized --results-dir results/snf_soft --reuse-bigclam-dir results/snf_bigclam
```

## Comparison set

The fair expression-only comparisons are the methods subtype papers actually report: SNF plus a spectral cut (Wang et al., Nat Methods 2014), consensus clustering, NMF, and Leiden. polyClustR, DSCC, and 3Mont are network precedents, not drop-in baselines. SUPREME, MOFA+, and MSFN use extra omics and are out of scope. No 2023–2026 breast-cancer expression paper applies BIGCLAM itself. Biological anchors remain Perou 2000, Parker PAM50 2009, TCGA BRCA 2012, and Curtis METABRIC 2012.

## Rebuild the manuscript

```bash
python manuscript/revise_figures_and_tables.py
python manuscript/export_figures.py
python manuscript/build_docx.py
```

`revise_figures_and_tables.py` redraws the Kaplan–Meier risk tables and the decision figure. `export_figures.py` writes the pipeline and bar charts and does not overwrite those revised Kaplan–Meier files. Figures live in `manuscript/paper_figures/includegraphics/`. Tables live in `manuscript/tables/`.

## Legacy path

`main.py` and `python run_q1_study.py --config config/config.yaml` belong to the earlier cosine-graph BIGCLAM plus MLP/SVM pipeline. An absolute variance cutoff of 13 is not the locked gene rule. Classifier accuracy repredicts labels built from the same expression profiles and is not used in the manuscript.
