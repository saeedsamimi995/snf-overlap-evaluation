"""SNF-BIGCLAM: overlapping affiliation on fused patient-similarity graphs.

Novelty hook: SNF supplies the multi-metric patient network that recent BC
papers already trust for hard subtypes; BIGCLAM supplies sparse overlapping
membership on that same fused graph. Hard labels come from SNF (prognosis) or
from BIGCLAM argmax on the SNF graph (network communities). Soft scores
(mixed flag, membership entropy) quantify subtype-boundary patients.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

from src.analysis.q1_clustering import (
    _connect_components,
    _knn_affinity_from_distance,
    choose_overlap_ratio,
    membership_summary,
    snf_fuse,
)
from src.bigclam.bigclam_model import fit_bigclam_c


def build_view_affinities(X, knn_k=15, snf_iterations=15):
    """Euclidean, correlation, and fused SNF affinities on one expression matrix."""
    features = np.asarray(X, dtype=np.float64)
    euclidean = euclidean_distances(features)
    correlation = pairwise_distances(features, metric="correlation")
    w_euclid = _knn_affinity_from_distance(euclidean, knn_k)
    w_corr = _knn_affinity_from_distance(correlation, knn_k)
    fused = snf_fuse([w_euclid, w_corr], k=knn_k, iterations=snf_iterations)
    return {"euclidean": w_euclid, "correlation": w_corr, "fused": fused}


def build_snf_affinity(X, knn_k=15, snf_iterations=15):
    """Fuse Euclidean and correlation KNN affinities (single-omics SNF)."""
    return build_view_affinities(X, knn_k=knn_k, snf_iterations=snf_iterations)["fused"]


def snf_affinity_to_graph(fused, knn_k=15):
    """Sparsify the fused affinity into an undirected kNN graph for BIGCLAM."""
    similarity = np.asarray(fused, dtype=np.float64).copy()
    np.fill_diagonal(similarity, -np.inf)
    n_samples = similarity.shape[0]
    neighbors = int(min(knn_k, n_samples - 1))
    nearest = np.argpartition(-similarity, kth=neighbors - 1, axis=1)[:, :neighbors]
    adjacency = np.zeros((n_samples, n_samples), dtype=np.float64)
    rows = np.repeat(np.arange(n_samples), neighbors)
    adjacency[rows, nearest.ravel()] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    # Reuse component bridging with the fused similarity as the bridge score.
    bridge = np.asarray(fused, dtype=np.float64).copy()
    adjacency = _connect_components(adjacency, bridge)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def snf_hard_labels(fused, n_communities, seed=42):
    """Classic SNF hard cut used as the prognostic primary partition."""
    return SpectralClustering(
        n_clusters=int(n_communities),
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
    ).fit_predict(np.asarray(fused, dtype=np.float64))


def membership_entropy(F):
    """Row-wise Shannon entropy of normalized affiliations (overlap score)."""
    membership = np.asarray(F, dtype=np.float64)
    row = membership / np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    row = np.clip(row, 1e-12, 1.0)
    return -np.sum(row * np.log(row), axis=1)


def fit_snf_bigclam(
    X,
    n_communities,
    knn_k=15,
    snf_iterations=15,
    iterations=250,
    restarts=5,
    lr=0.05,
    seed=42,
    l1=0.01,
    n_neg=8,
    overlap_ratio=None,
):
    """Fit SNF and BIGCLAM-on-SNF for one cohort.

    Returns a dict with fused affinity, SNF hard labels, SNF-graph adjacency,
    BIGCLAM ``F``, hard/mixed labels, entropy, and fit quality.
    """
    fused = build_snf_affinity(X, knn_k=knn_k, snf_iterations=snf_iterations)
    snf_labels = snf_hard_labels(fused, n_communities, seed=seed)
    adjacency = snf_affinity_to_graph(fused, knn_k=knn_k)
    membership, likelihood, peak = fit_bigclam_c(
        adjacency,
        num_communities=n_communities,
        iterations=iterations,
        lr=lr,
        restarts=restarts,
        seed=seed,
        l1=l1,
        n_neg=n_neg,
    )
    entropy = membership_entropy(membership)
    hard, mixed, ratios = membership_summary(membership, ratio=0.75)
    locked_ratio = overlap_ratio
    mixed_fraction = float(np.mean(mixed))
    if locked_ratio is None:
        locked_ratio, mixed_fraction = choose_overlap_ratio(ratios)
        mixed = ratios >= locked_ratio
        mixed_fraction = float(np.mean(mixed))
    else:
        mixed = ratios >= float(locked_ratio)
        mixed_fraction = float(np.mean(mixed))
    return {
        "fused": fused,
        "A": adjacency,
        "snf_community": snf_labels.astype(int),
        "F": membership,
        "community": hard,
        "mixed": mixed,
        "membership_ratio": ratios,
        "entropy": entropy,
        "overlap_ratio": float(locked_ratio),
        "mixed_fraction": mixed_fraction,
        "peak_ratio": float(peak),
        "log_likelihood": float(likelihood),
    }
