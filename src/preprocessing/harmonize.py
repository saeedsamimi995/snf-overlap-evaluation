"""Lock a discovery gene list and z-score each cohort on that list.

Variance is computed on the raw discovery matrix. Z-scoring first would give
every gene variance 1, so it cannot rank genes. Each cohort is then z-scored
on its own samples so RNA-seq and microarray graphs share a scale without
sharing a variance cutoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.cohorts import PROJECT_ROOT

MIN_FINITE_FRACTION = 0.8


def shared_gene_symbols(expression_by_cohort):
    """Genes present, and mostly measured, in every cohort."""
    keep = None
    for frame in expression_by_cohort.values():
        finite = np.isfinite(frame.to_numpy(dtype=np.float64)).mean(axis=1) >= MIN_FINITE_FRACTION
        symbols = set(frame.index[finite].astype(str))
        keep = symbols if keep is None else keep & symbols
    return sorted(keep or [])


def rank_genes_by_variance(expression, genes):
    """Return ``genes`` ordered from highest discovery variance to lowest."""
    block = expression.reindex(genes)
    values = block.to_numpy(dtype=np.float64)
    variance = np.nanvar(values, axis=1)
    order = np.argsort(-np.nan_to_num(variance, nan=-np.inf))
    return [genes[int(idx)] for idx in order]


def zscore_genes(values):
    """Column-wise z-score. ``values`` is samples by genes."""
    array = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(array, axis=0, keepdims=True)
    std = np.nanstd(array, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    scored = (array - mean) / std
    return np.nan_to_num(scored, nan=0.0).astype(np.float32)


def harmonize_loaded(loaded, n_genes, discovery="gse96058", min_shared_genes=200):
    """Intersect symbols, lock the discovery gene list, and z-score each cohort."""
    expression = {name: cohort["expression"] for name, cohort in loaded.items()}
    if discovery not in expression:
        raise KeyError(f"Discovery cohort {discovery} was not loaded")
    shared = shared_gene_symbols(expression)
    print(f"  Shared genes with measurements in every cohort: {len(shared):,}")
    if len(shared) < min_shared_genes:
        raise RuntimeError(
            f"Only {len(shared)} genes are shared across cohorts; need at least {min_shared_genes}"
        )
    ranked = rank_genes_by_variance(expression[discovery], shared)
    gene_list = ranked[: min(int(n_genes), len(ranked))]
    print(f"  Locked {len(gene_list)} genes by {discovery} variance")

    harmonized = {"genes": gene_list, "discovery": discovery, "cohorts": {}}
    for name, cohort in loaded.items():
        block = cohort["expression"].reindex(gene_list).T  # samples x genes
        block.index = block.index.astype(str)
        clinical = cohort["clinical"].copy()
        clinical.index = clinical.index.astype(str)
        clinical = clinical.reindex(block.index)
        scored = zscore_genes(block.to_numpy(dtype=np.float64))
        harmonized["cohorts"][name] = {
            "X": scored,
            "samples": list(block.index),
            "clinical": clinical,
        }
        print(f"  {name}: {scored.shape[0]} samples x {scored.shape[1]} z-scored genes")
    return harmonized


def save_harmonized(harmonized, harmonized_dir, metabric_dir="data/metabric"):
    """Write the locked matrices. METABRIC also gets a copy under data/metabric."""
    out = PROJECT_ROOT / harmonized_dir
    out.mkdir(parents=True, exist_ok=True)
    gene_path = out / "gene_list.txt"
    gene_path.write_text("\n".join(harmonized["genes"]) + "\n", encoding="utf-8")
    manifest = {
        "discovery": harmonized["discovery"],
        "n_genes": len(harmonized["genes"]),
        "cohorts": {},
    }
    for name, cohort in harmonized["cohorts"].items():
        np.save(out / f"{name}_expression.npy", cohort["X"])
        (out / f"{name}_samples.txt").write_text(
            "\n".join(cohort["samples"]) + "\n", encoding="utf-8"
        )
        clinical_path = out / f"{name}_clinical.csv"
        cohort["clinical"].to_csv(clinical_path)
        manifest["cohorts"][name] = {
            "n_samples": int(cohort["X"].shape[0]),
            "n_genes": int(cohort["X"].shape[1]),
            "n_pam50": int(cohort["clinical"]["PAM50"].notna().sum()),
            "n_survival": int(
                (cohort["clinical"]["OS_time"].notna() & cohort["clinical"]["OS_event"].notna()).sum()
            ),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if "metabric" in harmonized["cohorts"]:
        meta_out = PROJECT_ROOT / metabric_dir
        meta_out.mkdir(parents=True, exist_ok=True)
        meta = harmonized["cohorts"]["metabric"]
        np.save(meta_out / "metabric_expression.npy", meta["X"])
        (meta_out / "metabric_genes.txt").write_text(
            "\n".join(harmonized["genes"]) + "\n", encoding="utf-8"
        )
        (meta_out / "metabric_samples.txt").write_text(
            "\n".join(meta["samples"]) + "\n", encoding="utf-8"
        )
        meta["clinical"].to_csv(meta_out / "metabric_clinical.csv")
    print(f"  Harmonized matrices written to {out}")
    return out


def load_harmonized(harmonized_dir):
    """Reload matrices written by :func:`save_harmonized`."""
    root = PROJECT_ROOT / harmonized_dir
    genes = [
        line.strip()
        for line in (root / "gene_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    harmonized = {"genes": genes, "discovery": manifest["discovery"], "cohorts": {}}
    for name in manifest["cohorts"]:
        samples = [
            line.strip()
            for line in (root / f"{name}_samples.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        clinical = pd.read_csv(root / f"{name}_clinical.csv", index_col=0)
        clinical.index = clinical.index.astype(str)
        clinical = clinical.reindex(samples)
        harmonized["cohorts"][name] = {
            "X": np.load(root / f"{name}_expression.npy"),
            "samples": samples,
            "clinical": clinical,
        }
    return harmonized
