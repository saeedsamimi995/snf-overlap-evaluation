# Supplementary Methods: hyperparameters and locks

## Gene selection (locked on discovery)
1. Intersect gene symbols across GSE96058, TCGA-BRCA, and METABRIC.  
2. On **GSE96058 only**, rank intersecting genes by variance.  
3. Keep the top **2,000** genes (`locked_config.json` → `n_genes`).  
4. Apply that **same gene list** to TCGA and METABRIC.  
5. Z-score **within each cohort** (gene-wise).

## SNF (hard backbone)
| Parameter | Value |
|-----------|-------|
| Views | Euclidean distance kNN; correlation distance kNN |
| kNN `k` | 15 |
| Local affinity | KNN affinity from pairwise distances (`_knn_affinity_from_distance`) |
| Fusion | Classic SNF iterative message passing (`snf_fuse`) |
| SNF iterations | **15** |
| Hard cut | Spectral clustering on fused affinity, `assign_labels=kmeans`, `K=5`, `seed=42` |
| Graph sparsification for soft engines | Keep mutual top-`k=15` neighbors of fused affinity; symmetrize; bridge disconnected components by strongest fused similarity |

## Soft / overlapping engines (on sparsified SNF graph)

### NOCD-X / NOCD-G (Shchur & Günnemann–style)
| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer GraphConv (GCN-style normalized adjacency) |
| Hidden size | 64 |
| Output | ReLU affiliations, `K=5` |
| Dropout | 0.2 |
| Epochs | 250 |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-2` |
| Loss | Bernoulli–Poisson edge likelihood with **8** negative samples per edge + mild L1 on `F` |
| NOCD-X features | Expression → StandardScaler → PCA (`≤32` PCs) |
| NOCD-G features | Degree + random projection of adjacency (no expression) |

### SLPA
| Parameter | Value |
|-----------|-------|
| Iterations | 40 |
| Soft `F` | Relative frequency of the **global top-K** labels in each node’s memory |
| Seed | 42 |

### CESNA-lite
Attribute-aware affiliation approximating CESNA without discrete attributes: non-negative `F` with graph Bernoulli–Poisson loss **plus** MSE reconstruction of expression PCA features (`F @ B`), `attr_weight=0.5`, 200 Adam steps, `lr=0.05`, 8 neg samples.

### BIGCLAM (baseline only)
Neg-sampling / balanced likelihood, NMF init, L1=`0.01`, `lr=0.05`, 250 iterations, 5 restarts, 8 neg samples — reused from `results/snf_bigclam/*_F.npy` when available.

## Mixed-membership lock (per engine, discovery → validation)
A patient is **mixed** if (2nd strongest affiliation) / (strongest) ≥ `ratio`.

**Why ratios differ across engines:** they are **not** fixed a priori to the same number. On GSE96058 discovery, for each engine independently we scan `ratio ∈ {0.50, 0.60, …, 0.95}` and pick the value whose mixed fraction is closest to the centre of the usable window **[0.10, 0.40]** (same rule as `choose_overlap_ratio` in code). Engines with flatter `F` (e.g. SLPA, NOCD-G) need a higher ratio to avoid marking almost everyone mixed; peaked engines (BIGCLAM, NOCD-X) lock lower. The chosen `ratio` is then **frozen** for TCGA/METABRIC.

| Engine | Locked ratio | Discovery rationale |
|--------|--------------|---------------------|
| NOCD-X | 0.60 | enters 10–40% window |
| NOCD-G | 0.90 | flatter `F`; high ratio needed |
| SLPA | 0.80 | still ~76% mixed on discovery (cannot fully enter window) |
| CESNA-lite | 0.70 | enters window |
| BIGCLAM | 0.50 | peaked `F`; lower ratio suffices |

## Soft lead decision policy (a priori)
1. **BIGCLAM** is designated a **classical baseline** and is **not eligible** to be the paper lead, even if it passes quality filters.  
2. Quality filters on METABRIC: mixed ∈ [0.10, 0.40]; peak ratio (median 2nd/1st) < 0.85; LumA OR > 1.  
3. Among **lead-eligible** engines that pass all filters, select by hard Cox then residual (see `tables/lead_decision_table.csv`).  
4. Engines not excluded before the bake-off: all five listed above were fit; SLPA/CESNA/NOCD-G may fail filters **after** seeing results.

## Multiple testing
Primary tables report **uncorrected** Cox LR *p*-values. With five hard methods × three cohorts, and five soft engines on METABRIC residuals, we discuss Benjamini–Hochberg FDR in the text; METABRIC SNF hard Cox and NOCD-X hard Cox remain significant under BH within the METABRIC hard-method family.
