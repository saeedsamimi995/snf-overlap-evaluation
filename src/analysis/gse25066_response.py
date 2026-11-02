"""Locked-pipeline test of pathological complete response on GSE25066.

Settings are copied from the discovery lock and are not retuned on this cohort:
K=5, kNN=15, SNF iterations=15, NOCD-X mixed ratio 0.60, BIGCLAM mixed ratio 0.50.
The pre-stated primary family is two NOCD-X tests: pCR in mixed versus single
patients inside PAM50 LumA, and inside the hard SNF cluster that contains the
most mixed patients. BIGCLAM is reported as the baseline and is not eligible
to replace the lead.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.analysis.ceiling_mechanistic import LOCKED_RATIO, bh_qvalues
from src.analysis.q1_clustering import membership_summary
from src.analysis.snf_bigclam import (
    build_snf_affinity,
    snf_affinity_to_graph,
    snf_hard_labels,
)
from src.analysis.overlapping_sota import fit_nocd
from src.bigclam.bigclam_model import fit_bigclam_c
from src.preprocessing.cohorts import PROJECT_ROOT

K = 5
KNN_K = 15
SNF_ITERATIONS = 15
SEED = 42
PRIMARY_ENGINE = "nocd_x"
BASELINE_ENGINE = "bigclam"


def _split_geo_row(line):
    parts = line.rstrip("\n").split("\t")
    return [part.strip().strip('"') for part in parts[1:]]


def load_clinical(matrix_path):
    """Sample-level fields from the GEO series-matrix header.

    Characteristics are parsed per sample. A row-wise key taken from the first
    sample mis-assigns later columns when a sample is missing a field.
    """
    columns = {}
    characteristic_rows = []
    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            key = line.split("\t", 1)[0][len("!Sample_") :]
            values = _split_geo_row(line)
            if key == "characteristics_ch1":
                characteristic_rows.append(values)
            elif key in {"title", "geo_accession"}:
                columns[key] = values
    n = len(columns["geo_accession"])
    records = []
    for index in range(n):
        record = {
            "title": columns["title"][index],
            "geo_accession": columns["geo_accession"][index],
        }
        for row in characteristic_rows:
            if index >= len(row) or ":" not in row[index]:
                continue
            name, value = row[index].split(":", 1)
            record[name.strip()] = value.strip()
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    frame.index = frame["geo_accession"].astype(str)
    response = frame["pathologic_response_pcr_rd"].astype(str)
    frame["pcr"] = np.where(response.eq("pCR"), 1, np.where(response.eq("RD"), 0, np.nan))
    frame["PAM50"] = frame["pam50_class"].astype(str)
    return frame


def load_probe_symbols(annot_path):
    """Map Affymetrix probe ids to a single Hugo symbol."""
    mapping = {}
    with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] in "#!^":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            symbol = parts[2].split("///")[0].strip().upper()
            if symbol and symbol not in {"---", "NA", "NAN"}:
                mapping[parts[0]] = symbol
    return mapping


def load_expression(matrix_path, symbol_of):
    """Probe matrix collapsed to one max-variance probe per gene symbol."""
    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
        table = pd.read_csv(handle, sep="\t", index_col=0, comment="!")
    table = table.apply(pd.to_numeric, errors="coerce")
    table.index = table.index.astype(str).str.strip().str.strip('"')
    symbols = pd.Series({probe: symbol_of.get(probe) for probe in table.index})
    table = table.loc[symbols.notna()]
    symbols = symbols.loc[table.index]
    variances = table.var(axis=1)
    keep = variances.groupby(symbols).idxmax()
    collapsed = table.loc[keep.to_numpy()]
    collapsed.index = keep.index.astype(str)
    return collapsed.T


def align_locked_genes(expression, clinical, gene_list):
    """Keep the locked genes that the array measures, in locked order, then z-score."""
    present = [gene for gene in gene_list if gene in expression.columns]
    block = expression.loc[clinical.index, present].to_numpy(dtype=np.float64)
    center = np.nanmean(block, axis=0)
    scale = np.nanstd(block, axis=0)
    scale[scale < 1e-8] = 1.0
    filled = np.where(np.isnan(block), center, block)
    scored = (filled - center) / scale
    return scored, present


def _pcr_test(pcr, mixed, scope, engine):
    pcr = np.asarray(pcr, dtype=float)
    mixed = np.asarray(mixed, dtype=bool)
    valid = np.isfinite(pcr)
    pcr = pcr[valid]
    mixed = mixed[valid]
    n_mixed = int(mixed.sum())
    n_single = int((~mixed).sum())
    pcr_mixed = int(pcr[mixed].sum()) if n_mixed else 0
    pcr_single = int(pcr[~mixed].sum()) if n_single else 0
    empty = {
        "engine": engine,
        "scope": scope,
        "n": int(valid.sum()),
        "n_mixed": n_mixed,
        "n_single": n_single,
        "n_pcr_mixed": pcr_mixed,
        "n_pcr_single": pcr_single,
        "rate_mixed": np.nan,
        "rate_single": np.nan,
        "odds_ratio": np.nan,
        "p": np.nan,
        "note": "",
    }
    if n_mixed < 15 or n_single < 15:
        empty["note"] = "fewer than 15 patients in one class"
        return empty
    table = np.array(
        [
            [pcr_mixed, n_mixed - pcr_mixed],
            [pcr_single, n_single - pcr_single],
        ]
    )
    odds, p_value = fisher_exact(table)
    empty.update(
        {
            "rate_mixed": pcr_mixed / n_mixed,
            "rate_single": pcr_single / n_single,
            "odds_ratio": float(odds),
            "p": float(p_value),
            "note": "Fisher exact, pCR in mixed versus single; OR<1 means lower pCR in mixed",
        }
    )
    return empty


def _largest_mixed_cluster(hard, mixed):
    best = None
    for community in np.unique(hard):
        in_c = hard == community
        n_mixed = int((in_c & mixed).sum())
        n_single = int((in_c & ~mixed).sum())
        if n_mixed >= 15 and n_single >= 15 and (best is None or n_mixed > best[0]):
            best = (n_mixed, int(community))
    return None if best is None else best[1]


def run(matrix_path, annot_path, gene_list_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Reading GSE25066 clinical fields")
    clinical = load_clinical(matrix_path)
    print("Reading GPL96 symbols")
    symbols = load_probe_symbols(annot_path)
    print("Collapsing probes")
    expression = load_expression(matrix_path, symbols)
    expression.index = expression.index.astype(str).str.strip().str.strip('"')
    clinical = clinical.loc[clinical.index.intersection(expression.index)]
    expression = expression.loc[clinical.index]
    genes = [
        line.strip().upper()
        for line in Path(gene_list_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    X, used = align_locked_genes(expression, clinical, genes)
    print(f"Locked genes on array: {len(used)} / {len(genes)}; n={X.shape[0]}")
    print("Building SNF")
    fused = build_snf_affinity(X, knn_k=KNN_K, snf_iterations=SNF_ITERATIONS)
    hard = snf_hard_labels(fused, K, seed=SEED)
    graph = snf_affinity_to_graph(fused, knn_k=KNN_K)
    print("Fitting NOCD-X")
    nocd, _ = fit_nocd(graph, X=X, n_communities=K, epochs=250, seed=SEED, feature_mode="attributes")
    print("Fitting BIGCLAM")
    bigclam, _, _ = fit_bigclam_c(
        graph,
        num_communities=K,
        iterations=250,
        restarts=5,
        seed=SEED,
        l1=0.01,
        n_neg=8,
    )
    memberships = {PRIMARY_ENGINE: nocd, BASELINE_ENGINE: bigclam}
    rows = []
    assignment = clinical.copy()
    assignment["snf_community"] = hard
    for engine, membership in memberships.items():
        _, mixed, _ = membership_summary(membership, ratio=LOCKED_RATIO[engine])
        assignment[f"{engine}_mixed"] = mixed.astype(int)
        family = "primary" if engine == PRIMARY_ENGINE else "baseline"
        luma = assignment["PAM50"].astype(str).str.lower().eq("luma").to_numpy()
        luma_row = _pcr_test(assignment["pcr"].to_numpy()[luma], mixed[luma], "LumA", engine)
        luma_row["family"] = family
        rows.append(luma_row)
        cluster = _largest_mixed_cluster(hard, mixed)
        if cluster is None:
            rows.append(
                {
                    "engine": engine,
                    "scope": "hard_cluster",
                    "p": np.nan,
                    "note": "no hard cluster with at least 15 mixed and 15 single",
                    "family": family,
                }
            )
        else:
            in_cluster = hard == cluster
            cluster_row = _pcr_test(
                assignment["pcr"].to_numpy()[in_cluster],
                mixed[in_cluster],
                f"hard_{int(cluster)}",
                engine,
            )
            cluster_row["family"] = family
            rows.append(cluster_row)
    tests = pd.DataFrame(rows)
    tests["q"] = np.nan
    for family in ("primary", "baseline"):
        mask = tests["family"].eq(family)
        tests.loc[mask, "q"] = bh_qvalues(tests.loc[mask, "p"].to_numpy())
    primary = tests[tests["family"].eq("primary")]
    hit = bool((primary["q"] < 0.05).any())
    summary = {
        "cohort": "GSE25066",
        "regimen": "neoadjuvant taxane plus anthracycline",
        "endpoint": "pathologic complete response versus residual disease",
        "n": int(len(assignment)),
        "n_pcr": int(assignment["pcr"].eq(1).sum()),
        "n_rd": int(assignment["pcr"].eq(0).sum()),
        "n_luma": int(assignment["PAM50"].astype(str).str.lower().eq("luma").sum()),
        "locked_genes_available": int(len(used)),
        "locked_genes_requested": int(len(genes)),
        "locked_ratios": {PRIMARY_ENGINE: LOCKED_RATIO[PRIMARY_ENGINE], BASELINE_ENGINE: LOCKED_RATIO[BASELINE_ENGINE]},
        "pre_stated_family": "NOCD-X Fisher tests inside LumA and inside the largest mixed hard cluster",
        "primary_q_lt_0.05": hit,
        "decision": (
            "pCR differs by mixed status inside the pre-stated NOCD-X family"
            if hit
            else "null: mixed status does not change pCR after BH inside the pre-stated NOCD-X family"
        ),
    }
    tests.to_csv(out / "pcr_tests.csv", index=False)
    assignment.to_csv(out / "assignments.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "genes_used.txt").write_text("\n".join(used) + "\n", encoding="utf-8")
    print(tests.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


def gene_overlap_sensitivity(out_dir=None):
    """Refit discovery SNF on the genes GSE25066 can measure.

    Compares that graph with the locked 2,000-gene discovery graph. Does not
    refit soft engines and does not change the locked gene list.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score

    root = PROJECT_ROOT
    out = Path(out_dir) if out_dir else root / "results" / "gse25066"
    genes = [
        line.strip()
        for line in (root / "data" / "harmonized" / "gene_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    used = [
        line.strip()
        for line in (out / "genes_used.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    position = {gene: index for index, gene in enumerate(genes)}
    columns = [position[gene] for gene in used if gene in position]
    expression = np.load(root / "data" / "harmonized" / "gse96058_expression.npy")
    block = np.asarray(expression[:, columns], dtype=np.float64)
    center = np.nanmean(block, axis=0)
    scale = np.nanstd(block, axis=0)
    scale[scale < 1e-8] = 1.0
    scored = (np.where(np.isnan(block), center, block) - center) / scale
    print(f"Discovery sensitivity: {scored.shape[0]} samples, {scored.shape[1]} genes")
    fused = build_snf_affinity(scored, knn_k=KNN_K, snf_iterations=SNF_ITERATIONS)
    hard = snf_hard_labels(fused, K, seed=SEED)
    locked_labels = np.load(root / "Results" / "snf_overlap_sota" / "gse96058_snf_labels.npy")
    locked_affinity = np.load(root / "results" / "ceiling" / "graphs" / "gse96058_fused_affinity.npy")
    tri = np.triu_indices(fused.shape[0], k=1)
    rho = spearmanr(locked_affinity[tri], fused[tri]).statistic
    summary = {
        "cohort": "gse96058",
        "n": int(scored.shape[0]),
        "n_genes_locked": int(len(genes)),
        "n_genes_on_array": int(scored.shape[1]),
        "fraction_of_locked_genes": float(scored.shape[1] / len(genes)),
        "hard_label_ari_vs_locked": float(adjusted_rand_score(locked_labels, hard)),
        "fused_affinity_spearman": float(rho),
        "note": "Same K, k, T, and seed. Soft engines were not refit. The locked gene list is unchanged.",
    }
    (out / "gene_overlap_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    root = PROJECT_ROOT
    run(
        root / "data" / "gse25066" / "GSE25066_series_matrix.txt.gz",
        root / "data" / "gse25066" / "GPL96.annot.gz",
        root / "data" / "harmonized" / "gene_list.txt",
        root / "results" / "gse25066",
    )


if __name__ == "__main__":
    main()
