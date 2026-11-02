"""Soft / overlapping patient stratification on SNF-fused affinity.

SNF literature almost always ends in a **hard** spectral cut (SNFtool,
CancerSubtypes, ab-SNF, WSNF, miss-SNF). This module keeps that hard SNF
partition as the prognostic backbone and compares several soft engines on the
same fused patient network:

  - snf_prototype : affinity to hard-SNF community prototypes (native soft SNF)
  - snf_gmm       : GMM on the spectral embedding of the fused affinity
  - snf_fcm       : fuzzy c-means on the same embedding
  - snf_nmf       : NMF factors of the fused affinity (soft loadings)
  - snf_bigclam   : overlapping affiliation on the sparsified SNF graph

None is assumed best a priori; residual Cox + enrichment decide.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from src.analysis.q1_clustering import choose_overlap_ratio, membership_summary
from src.analysis.snf_bigclam import (
    build_snf_affinity,
    membership_entropy,
    snf_affinity_to_graph,
    snf_hard_labels,
)
from src.bigclam.bigclam_model import fit_bigclam_c


def _peak_ratio(F: np.ndarray) -> float:
    membership = np.asarray(F, dtype=np.float64)
    if membership.ndim != 2 or membership.shape[1] < 2:
        return float("nan")
    order = np.argsort(membership, axis=1)
    rows = np.arange(membership.shape[0])
    top = membership[rows, order[:, -1]]
    second = membership[rows, order[:, -2]]
    return float(np.median(second / np.maximum(top, 1e-12)))


def _wrap_membership(F: np.ndarray, overlap_ratio=None):
    """Hard cut, mixed flag, entropy, peakiness from any non-negative F."""
    membership = np.asarray(F, dtype=np.float64)
    membership = np.maximum(membership, 0.0)
    entropy = membership_entropy(membership)
    hard, mixed, ratios = membership_summary(membership, ratio=0.75)
    locked = overlap_ratio
    if locked is None:
        locked, mixed_fraction = choose_overlap_ratio(ratios)
        mixed = ratios >= locked
    else:
        mixed = ratios >= float(locked)
        mixed_fraction = float(np.mean(mixed))
    return {
        "F": membership,
        "community": hard.astype(int),
        "mixed": mixed.astype(bool),
        "membership_ratio": ratios,
        "entropy": entropy,
        "overlap_ratio": float(locked),
        "mixed_fraction": float(np.mean(mixed)),
        "peak_ratio": _peak_ratio(membership),
    }


def spectral_embedding_from_affinity(fused: np.ndarray, n_communities: int, seed: int = 42):
    """Normalized Laplacian embedding used by spectral clustering / GMM / FCM."""
    affinity = np.asarray(fused, dtype=np.float64)
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    degree = np.maximum(degree, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(degree)
    laplacian = np.eye(affinity.shape[0]) - (inv_sqrt[:, None] * affinity * inv_sqrt[None, :])
    # smallest eigenvectors of normalized Laplacian
    vals, vecs = np.linalg.eigh(laplacian)
    k = int(n_communities)
    embedding = vecs[:, :k]
    embedding = normalize(embedding, norm="l2", axis=1)
    return embedding, vals[:k]


def soft_prototype(fused: np.ndarray, snf_labels: np.ndarray, n_communities: int):
    """Soft score = mean fused affinity to each hard SNF community."""
    affinity = np.asarray(fused, dtype=np.float64)
    labels = np.asarray(snf_labels, dtype=int)
    n = affinity.shape[0]
    F = np.zeros((n, int(n_communities)), dtype=np.float64)
    for community in range(int(n_communities)):
        members = np.where(labels == community)[0]
        if len(members) == 0:
            continue
        F[:, community] = affinity[:, members].mean(axis=1)
    F = np.maximum(F, 0.0)
    return F


def soft_nmf(fused: np.ndarray, n_communities: int, seed: int = 42):
    """NMF loadings of the fused affinity matrix (soft community factors)."""
    affinity = np.asarray(fused, dtype=np.float64)
    affinity = np.maximum(0.5 * (affinity + affinity.T), 0.0)
    np.fill_diagonal(affinity, affinity.max(axis=1) + 1e-6)  # stabilize
    model = NMF(
        n_components=int(n_communities),
        init="nndsvd",
        max_iter=800,
        random_state=int(seed),
    )
    return model.fit_transform(affinity)


def soft_gmm(embedding: np.ndarray, n_communities: int, seed: int = 42):
    """Gaussian mixture responsibilities on the spectral embedding."""
    model = GaussianMixture(
        n_components=int(n_communities),
        covariance_type="full",
        n_init=5,
        random_state=int(seed),
    )
    model.fit(embedding)
    return model.predict_proba(embedding)


def soft_fcm(embedding: np.ndarray, n_communities: int, m: float = 2.0, max_iter: int = 150, seed: int = 42):
    """Fuzzy c-means memberships on the spectral embedding."""
    rng = np.random.default_rng(int(seed))
    x = np.asarray(embedding, dtype=np.float64)
    n, d = x.shape
    k = int(n_communities)
    # init from hard spectral labels if possible, else random simplex
    try:
        hard = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=min(15, n - 1), random_state=int(seed)
        ).fit_predict(x)
        U = np.eye(k)[hard].astype(np.float64) * 0.9 + 0.1 / k
    except Exception:
        U = rng.random((n, k))
        U /= U.sum(axis=1, keepdims=True)
    power = 2.0 / max(m - 1.0, 1e-6)
    for _ in range(int(max_iter)):
        Um = U**m
        centers = (Um.T @ x) / np.maximum(Um.sum(axis=0)[:, None], 1e-12)
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        dist = np.maximum(dist, 1e-12)
        inv = dist ** (-power)
        U_new = inv / inv.sum(axis=1, keepdims=True)
        if np.max(np.abs(U_new - U)) < 1e-5:
            U = U_new
            break
        U = U_new
    return U


def fit_snf_soft_suite(
    X,
    n_communities,
    knn_k=15,
    snf_iterations=15,
    seed=42,
    bigclam_iterations=250,
    bigclam_restarts=5,
    bigclam_lr=0.05,
    bigclam_l1=0.01,
    n_neg=8,
    overlap_ratios=None,
    skip_bigclam=False,
    precomputed_bigclam_F=None,
):
    """Build SNF once; fit all soft engines; return hard SNF + soft dict.

    ``overlap_ratios`` maps method -> locked mixed threshold (discovery lock).
    """
    overlap_ratios = dict(overlap_ratios or {})
    fused = build_snf_affinity(X, knn_k=knn_k, snf_iterations=snf_iterations)
    snf_labels = snf_hard_labels(fused, n_communities, seed=seed)
    embedding, _ = spectral_embedding_from_affinity(fused, n_communities, seed=seed)
    adjacency = snf_affinity_to_graph(fused, knn_k=knn_k)

    raw = {
        "snf_prototype": soft_prototype(fused, snf_labels, n_communities),
        "snf_nmf": soft_nmf(fused, n_communities, seed=seed),
        "snf_gmm": soft_gmm(embedding, n_communities, seed=seed),
        "snf_fcm": soft_fcm(embedding, n_communities, seed=seed),
    }

    if precomputed_bigclam_F is not None:
        raw["snf_bigclam"] = np.asarray(precomputed_bigclam_F, dtype=np.float64)
    elif not skip_bigclam:
        membership, likelihood, _ = fit_bigclam_c(
            adjacency,
            num_communities=n_communities,
            iterations=bigclam_iterations,
            lr=bigclam_lr,
            restarts=bigclam_restarts,
            seed=seed,
            l1=bigclam_l1,
            n_neg=n_neg,
        )
        raw["snf_bigclam"] = membership
        bigclam_ll = float(likelihood)
    else:
        bigclam_ll = None

    soft = {}
    for name, F in raw.items():
        soft[name] = _wrap_membership(F, overlap_ratio=overlap_ratios.get(name))
        if name == "snf_bigclam" and precomputed_bigclam_F is None and not skip_bigclam:
            soft[name]["log_likelihood"] = bigclam_ll

    return {
        "fused": fused,
        "A": adjacency,
        "embedding": embedding,
        "snf_community": snf_labels.astype(int),
        "soft": soft,
    }
