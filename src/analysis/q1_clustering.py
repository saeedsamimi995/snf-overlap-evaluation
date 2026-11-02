"""Patient-graph BIGCLAM, overlap calls, stability, and hard-cluster baselines."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

from src.bigclam.bigclam_model import fit_bigclam_c, membership_peakiness


def symmetric_knn_graph(X, k, similarity=None):
    """Undirected k-nearest-neighbor graph from cosine similarity.

    An edge is kept when either sample lists the other among its ``k`` nearest
    neighbors. Components are then joined by the strongest cross edge so later
    spectral clustering sees one graph. The returned adjacency has a zero diagonal.
    """
    features = np.asarray(X, dtype=np.float64)
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to build a patient graph")
    if similarity is None:
        similarity = cosine_similarity(features)
    else:
        similarity = np.array(similarity, dtype=np.float64, copy=True)
    np.fill_diagonal(similarity, -np.inf)
    neighbors = int(min(k, n_samples - 1))
    nearest = np.argpartition(-similarity, kth=neighbors - 1, axis=1)[:, :neighbors]
    adjacency = np.zeros((n_samples, n_samples), dtype=np.float64)
    row_index = np.repeat(np.arange(n_samples), neighbors)
    adjacency[row_index, nearest.ravel()] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    adjacency = _connect_components(adjacency, similarity)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def _connect_components(adjacency, similarity):
    """Add the strongest bridge between components until the graph is connected."""
    working = adjacency.copy()
    bridge = similarity.copy()
    np.fill_diagonal(bridge, -np.inf)
    for _ in range(working.shape[0]):
        n_components, labels = connected_components(working, directed=False, connection="weak")
        if n_components <= 1:
            break
        left = np.where(labels == 0)[0]
        right = np.where(labels != 0)[0]
        block = bridge[np.ix_(left, right)]
        flat = int(np.argmax(block))
        i, j = np.unravel_index(flat, block.shape)
        a = int(left[i])
        b = int(right[j])
        working[a, b] = 1.0
        working[b, a] = 1.0
    return working


def membership_summary(F, ratio=0.75):
    """Hard label, mixed-membership flag, and second-to-first membership ratio.

    A patient is mixed when the second-strongest affiliation is at least
    ``ratio`` times the strongest. ``ratio`` is locked on the discovery cohort
    after the membership matrix is peaked.
    """
    membership = np.asarray(F, dtype=np.float64)
    if membership.ndim != 2 or membership.shape[1] < 2:
        hard = np.zeros(membership.shape[0], dtype=int)
        mixed = np.zeros(membership.shape[0], dtype=bool)
        strength_ratio = np.zeros(membership.shape[0], dtype=np.float64)
        return hard, mixed, strength_ratio
    order = np.argsort(membership, axis=1)
    top_index = order[:, -1]
    second_index = order[:, -2]
    rows = np.arange(membership.shape[0])
    top = membership[rows, top_index]
    second = membership[rows, second_index]
    strength_ratio = second / np.maximum(top, 1e-12)
    mixed = strength_ratio >= float(ratio)
    return top_index.astype(int), mixed, strength_ratio


def choose_overlap_ratio(strength_ratio, candidates=None, target=(0.10, 0.40)):
    """Pick the mixed threshold that yields a usable minority of mixed patients.

    Discovery only. Locked afterward and applied unchanged to validation cohorts.
    """
    ratios = np.asarray(strength_ratio, dtype=np.float64)
    if candidates is None:
        candidates = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    low, high = target
    scored = []
    for candidate in candidates:
        fraction = float(np.mean(ratios >= float(candidate)))
        if low <= fraction <= high:
            # Prefer the middle of the target window.
            scored.append((abs(fraction - 0.5 * (low + high)), candidate, fraction))
    if scored:
        scored.sort()
        return float(scored[0][1]), float(scored[0][2])
    # Fallback: closest fraction to the window, preferring fewer mixed patients.
    fallback = []
    for candidate in candidates:
        fraction = float(np.mean(ratios >= float(candidate)))
        if fraction >= 0.95:
            distance = 10.0 + fraction
        else:
            distance = abs(fraction - high) if fraction > high else abs(fraction - low)
        fallback.append((distance, candidate, fraction))
    fallback.sort()
    return float(fallback[0][1]), float(fallback[0][2])


def _pairwise_ari(labelings):
    scores = []
    for i in range(len(labelings)):
        idx_a, lab_a = labelings[i]
        map_a = {int(sample): int(label) for sample, label in zip(idx_a, lab_a)}
        for j in range(i + 1, len(labelings)):
            idx_b, lab_b = labelings[j]
            left = []
            right = []
            for sample, label in zip(idx_b, lab_b):
                sample = int(sample)
                if sample in map_a:
                    left.append(map_a[sample])
                    right.append(int(label))
            if len(left) < 20:
                continue
            scores.append(adjusted_rand_score(left, right))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def select_c_by_stability(
    X,
    communities,
    knn_k=15,
    n_boot=8,
    fraction=0.8,
    iterations=120,
    restarts=2,
    lr=0.05,
    seed=42,
    l1=0.01,
    min_ari=0.05,
):
    """Pick the community count with the highest mean bootstrap ARI.

    Ties go to the smaller count. Survival is not used here. Counts whose
    mean ARI stays below ``min_ari`` are discarded when any better option exists.
    """
    features = np.asarray(X, dtype=np.float64)
    n_samples = features.shape[0]
    subset_size = max(30, int(round(fraction * n_samples)))
    subset_size = min(subset_size, n_samples)
    table = {}
    peak_table = {}
    for n_communities in communities:
        labelings = []
        peaks = []
        for bootstrap in range(int(n_boot)):
            rng = np.random.default_rng(int(seed) + 1000 * int(n_communities) + bootstrap)
            chosen = np.sort(rng.choice(n_samples, size=subset_size, replace=False))
            adjacency = symmetric_knn_graph(features[chosen], knn_k)
            membership, _, peak = fit_bigclam_c(
                adjacency,
                num_communities=int(n_communities),
                iterations=iterations,
                lr=lr,
                restarts=restarts,
                seed=int(seed) + bootstrap,
                l1=l1,
            )
            hard, _, _ = membership_summary(membership, ratio=0.75)
            labelings.append((chosen, hard))
            peaks.append(peak)
            print(
                f"    stability C={n_communities} bootstrap {bootstrap + 1}/{n_boot} "
                f"n={subset_size} peak_ratio={peak:.3f}"
            )
        table[int(n_communities)] = _pairwise_ari(labelings)
        peak_table[int(n_communities)] = float(np.nanmean(peaks))
        print(
            f"  Mean pairwise ARI at C={n_communities}: {table[int(n_communities)]:.4f} "
            f"(mean peak ratio {peak_table[int(n_communities)]:.3f})"
        )
    valid = {k: v for k, v in table.items() if np.isfinite(v)}
    above = {k: v for k, v in valid.items() if v >= float(min_ari)}
    pool = above if above else valid
    if not pool:
        chosen = int(min(communities))
    else:
        best_score = max(pool.values())
        chosen = min(k for k, v in pool.items() if np.isclose(v, best_score))
    print(f"  Locked community count: {chosen}")
    if valid and valid[chosen] < float(min_ari):
        print(
            f"  [warn] locked ARI {valid[chosen]:.4f} is below the {min_ari} gate; "
            "communities may still be weak"
        )
    return chosen, table, peak_table


def fit_cohort(
    X,
    num_communities,
    knn_k,
    iterations,
    restarts,
    lr,
    seed,
    overlap_ratio,
    l1=0.01,
    n_neg=8,
    max_peak_ratio=0.85,
):
    """Fit the locked patient graph and BIGCLAM model. Returns labels and ``F``."""
    adjacency = symmetric_knn_graph(X, knn_k)
    membership, likelihood, peak = fit_bigclam_c(
        adjacency,
        num_communities=num_communities,
        iterations=iterations,
        lr=lr,
        restarts=restarts,
        seed=seed,
        l1=l1,
        n_neg=n_neg,
        max_peak_ratio=max_peak_ratio,
    )
    hard, mixed, strength_ratio = membership_summary(membership, ratio=overlap_ratio)
    return {
        "A": adjacency,
        "F": membership,
        "log_likelihood": float(likelihood),
        "peak_ratio": float(peak),
        "community": hard,
        "mixed": mixed,
        "membership_ratio": strength_ratio,
        "mixed_fraction": float(np.mean(mixed)),
    }


def _leiden_labels(adjacency, target_k, seed):
    import igraph as ig
    import leidenalg

    sources, targets = np.where(np.triu(adjacency, k=1) > 0)
    graph = ig.Graph(
        n=adjacency.shape[0],
        edges=list(zip(sources.tolist(), targets.tolist())),
        directed=False,
    )
    graph.simplify()
    low, high = 0.05, 20.0
    best_labels = None
    best_gap = 10**9
    for _ in range(16):
        resolution = 0.5 * (low + high)
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=int(seed),
        )
        labels = np.asarray(partition.membership, dtype=int)
        n_found = len(np.unique(labels))
        gap = abs(n_found - int(target_k))
        if gap < best_gap:
            best_gap = gap
            best_labels = labels
        if n_found > target_k:
            high = resolution
        elif n_found < target_k:
            low = resolution
        else:
            break
    return best_labels


def _knn_affinity_from_distance(distance, k):
    """Sparse KNN similarity used inside single-omics SNF."""
    distance = np.asarray(distance, dtype=np.float64)
    n_samples = distance.shape[0]
    neighbors = int(min(k, n_samples - 1))
    affinity = np.zeros_like(distance)
    for i in range(n_samples):
        order = np.argpartition(distance[i], neighbors)[:neighbors]
        mu = np.median(distance[i, order]) + 1e-8
        affinity[i, order] = np.exp(-(distance[i, order] ** 2) / (mu ** 2))
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _row_normalize(matrix):
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
    return matrix / row_sum


def snf_fuse(affinities, k=15, iterations=20):
    """Similarity Network Fusion (Wang et al., Nature Methods 2014).

    Used here with two expression-derived affinities (Euclidean and correlation),
    matching the single-omics SNF setup common when only RNA is available
    (e.g. MOVICS / metabolic-gene subtyping comparisons).
    """
    matrices = [np.asarray(a, dtype=np.float64) for a in affinities]
    n_samples = matrices[0].shape[0]
    neighbors = int(min(k, n_samples - 1))
    local = []
    for matrix in matrices:
        work = matrix.copy()
        np.fill_diagonal(work, 0.0)
        sparse = np.zeros_like(work)
        for i in range(n_samples):
            order = np.argpartition(-work[i], neighbors)[:neighbors]
            sparse[i, order] = work[i, order]
        sparse = 0.5 * (sparse + sparse.T)
        np.fill_diagonal(sparse, np.diag(matrix))
        local.append(_row_normalize(sparse))
    status = [_row_normalize(matrix.copy()) for matrix in matrices]
    for _ in range(int(iterations)):
        updated = []
        for view, loc in enumerate(local):
            others = [status[j] for j in range(len(status)) if j != view]
            average = np.mean(others, axis=0)
            fused = loc @ average @ loc.T
            np.fill_diagonal(fused, 1.0)
            updated.append(_row_normalize(fused))
        status = updated
    fused = np.mean(status, axis=0)
    fused = 0.5 * (fused + fused.T)
    np.fill_diagonal(fused, 1.0)
    return fused


def consensus_kmeans_labels(X, n_communities, n_iter=40, sample_fraction=0.8, seed=42):
    """Monti-style consensus clustering with k-means (TCGA/METABRIC papers).

    Subsamples patients repeatedly, accumulates a consensus co-assignment
    matrix, then cuts that matrix with spectral clustering at the locked ``C``.
    """
    features = np.asarray(X, dtype=np.float64)
    n_samples = features.shape[0]
    subset = max(20, int(round(sample_fraction * n_samples)))
    subset = min(subset, n_samples)
    consensus = np.zeros((n_samples, n_samples), dtype=np.float64)
    counts = np.zeros((n_samples, n_samples), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for step in range(int(n_iter)):
        chosen = np.sort(rng.choice(n_samples, size=subset, replace=False))
        labels = KMeans(
            n_clusters=int(n_communities),
            n_init=5,
            random_state=int(seed) + step,
        ).fit_predict(features[chosen])
        counts[np.ix_(chosen, chosen)] += 1.0
        for community in np.unique(labels):
            members = chosen[labels == community]
            consensus[np.ix_(members, members)] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.divide(consensus, counts, out=np.zeros_like(consensus), where=counts > 0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return SpectralClustering(
        n_clusters=int(n_communities),
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
    ).fit_predict(matrix)


def baseline_labels(X, adjacency, num_communities, seed=42, knn_k=15):
    """Hard partitions at the locked community count.

    Includes the methods most often used as fair expression-only comparisons in
    recent breast-cancer subtyping papers:
    k-means, NMF, spectral, Leiden, Monti consensus clustering, and SNF-style
    fusion of Euclidean + correlation affinities.
    """
    features = np.asarray(X, dtype=np.float64)
    n_communities = int(num_communities)
    labels = {}
    labels["kmeans"] = KMeans(
        n_clusters=n_communities, n_init=10, random_state=int(seed)
    ).fit_predict(features)

    shifted = features - np.nanmin(features, axis=0, keepdims=True)
    shifted = np.nan_to_num(shifted, nan=0.0) + 1e-6
    factors = NMF(
        n_components=n_communities,
        init="nndsvda",
        random_state=int(seed),
        max_iter=400,
    ).fit_transform(shifted)
    labels["nmf"] = factors.argmax(axis=1).astype(int)

    spectral = SpectralClustering(
        n_clusters=n_communities,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
    )
    labels["spectral"] = spectral.fit_predict(
        np.asarray(adjacency, dtype=np.float64) + np.eye(features.shape[0])
    )
    labels["leiden"] = _leiden_labels(adjacency, n_communities, seed)

    print("    consensus clustering (Monti / k-means)...")
    labels["consensus"] = consensus_kmeans_labels(
        features, n_communities, n_iter=40, sample_fraction=0.8, seed=int(seed)
    )

    print("    SNF-style Euclidean+correlation fusion...")
    # Pairwise distances for the two views used in single-omics SNF comparisons.
    from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

    euclidean = euclidean_distances(features)
    correlation = pairwise_distances(features, metric="correlation")
    w_euclid = _knn_affinity_from_distance(euclidean, knn_k)
    w_corr = _knn_affinity_from_distance(correlation, knn_k)
    fused = snf_fuse([w_euclid, w_corr], k=knn_k, iterations=15)
    labels["snf"] = SpectralClustering(
        n_clusters=n_communities,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
    ).fit_predict(fused)

    counts = {name: int(len(np.unique(lab))) for name, lab in labels.items()}
    return labels, counts
