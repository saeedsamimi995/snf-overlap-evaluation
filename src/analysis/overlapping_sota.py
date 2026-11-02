"""SOTA-leaning overlapping engines for SNF patient graphs.

BIGCLAM is treated as a classical baseline, not the paper lead. Primary
candidates from the overlapping-community literature:

  - NOCD  (Shchur & Günnemann, KDD DLG 2019): GNN + Bernoulli–Poisson
  - SLPA  (Xie et al., 2011): speaker–listener label propagation (soft counts)
  - CESNA-lite: attribute-aware affiliation (expression features + graph loss)

These are fitted on the sparsified SNF adjacency (and optional node features).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# NOCD: Neural Overlapping Community Detection
# ---------------------------------------------------------------------------


class _GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x, adj_norm):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return adj_norm @ self.linear(x)


class NOCD(nn.Module):
    """Two-layer GCN → non-negative community affiliations (ReLU)."""

    def __init__(self, in_dim: int, hidden: int, n_communities: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = _GraphConv(in_dim, hidden, dropout=dropout)
        self.conv2 = _GraphConv(hidden, int(n_communities), dropout=dropout)

    def forward(self, x, adj_norm):
        h = F.relu(self.conv1(x, adj_norm))
        return F.relu(self.conv2(h, adj_norm))


def _normalize_adj(adjacency: np.ndarray) -> torch.Tensor:
    a = np.asarray(adjacency, dtype=np.float64)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    a = a + np.eye(a.shape[0])
    deg = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    norm = inv_sqrt[:, None] * a * inv_sqrt[None, :]
    return torch.tensor(norm, dtype=torch.float32)


def _edge_index_from_adj(adjacency: np.ndarray) -> torch.Tensor:
    a = np.asarray(adjacency, dtype=np.float32)
    rows, cols = np.where(np.triu(a, k=1) > 0)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def _berpo_neg_loss(F: torch.Tensor, edge_index: torch.Tensor, n_neg: int, generator: torch.Generator):
    """Balanced Bernoulli–Poisson edge likelihood (NOCD / BIGCLAM family)."""
    sources = edge_index[0]
    targets = edge_index[1]
    pos = torch.clamp((F[sources] * F[targets]).sum(dim=1), max=40.0)
    pos_ll = torch.log(1.0 - torch.exp(-pos) + 1e-10).sum()
    n_edges = max(int(sources.shape[0]), 1)
    neg_s = sources.repeat_interleave(int(n_neg))
    neg_t = torch.randint(0, F.shape[0], (n_edges * int(n_neg),), generator=generator)
    mask = neg_t != neg_s
    if mask.any():
        neg = torch.clamp((F[neg_s[mask]] * F[neg_t[mask]]).sum(dim=1), max=40.0)
        neg_pen = neg.sum() * (float(n_edges) / float(int(mask.sum())))
    else:
        neg_pen = torch.zeros((), device=F.device)
    return -(pos_ll - neg_pen)


def _node_features(X: np.ndarray | None, adjacency: np.ndarray, n_pca: int = 32, seed: int = 42):
    """Expression PCA features, or degree+adjacency PCA fallback."""
    n = adjacency.shape[0]
    if X is not None:
        feats = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
        k = int(min(n_pca, feats.shape[1], n - 1))
        if k >= 2:
            return PCA(n_components=k, random_state=seed).fit_transform(feats)
    # Graph-only features: normalized degrees + random projection of A
    deg = adjacency.sum(axis=1, keepdims=True)
    deg = deg / (deg.max() + 1e-12)
    rng = np.random.default_rng(seed)
    proj = adjacency @ rng.normal(size=(n, min(n_pca, 16)))
    proj = StandardScaler().fit_transform(proj)
    return np.hstack([deg, proj]).astype(np.float64)


def fit_nocd(
    adjacency,
    X=None,
    n_communities=5,
    hidden=64,
    epochs=250,
    lr=1e-3,
    weight_decay=1e-2,
    n_neg=8,
    seed=42,
    feature_mode="attributes",
):
    """Fit NOCD on an SNF sparsified graph.

    ``feature_mode``: ``attributes`` uses expression (X); ``graph`` ignores X.
    """
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed))
    a = np.asarray(adjacency, dtype=np.float32)
    feats = _node_features(
        X if feature_mode == "attributes" else None,
        a,
        seed=seed,
    )
    x = torch.tensor(feats, dtype=torch.float32)
    adj_norm = _normalize_adj(a)
    edge_index = _edge_index_from_adj(a)
    model = NOCD(x.shape[1], hidden, n_communities, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    best_F, best_loss = None, float("inf")
    for epoch in range(int(epochs)):
        opt.zero_grad()
        F_hat = model(x, adj_norm)
        loss = _berpo_neg_loss(F_hat, edge_index, n_neg=n_neg, generator=generator)
        # Mild L1 on affiliations to encourage peakiness
        loss = loss + 1e-3 * F_hat.abs().mean()
        loss.backward()
        opt.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_F = F_hat.detach().cpu().numpy().copy()

    if best_F is None:
        model.eval()
        with torch.no_grad():
            best_F = model(x, adj_norm).cpu().numpy()
    return np.maximum(best_F, 0.0), float(best_loss)


# ---------------------------------------------------------------------------
# SLPA: Speaker–Listener Label Propagation (soft label histograms)
# ---------------------------------------------------------------------------


def fit_slpa(adjacency, n_communities=5, iterations=40, seed=42):
    """SLPA soft memberships restricted to the K most frequent communities.

    Each node keeps a memory of heard labels. Soft F[i,c] is the relative
    frequency of global community ``c`` in node i's memory (among top-K).
    """
    rng = np.random.default_rng(int(seed))
    a = np.asarray(adjacency, dtype=np.float32)
    n = a.shape[0]
    neighbors = [np.flatnonzero(a[i] > 0).astype(int) for i in range(n)]
    memory = [[i] for i in range(n)]  # init unique label = node id

    for _ in range(int(iterations)):
        order = rng.permutation(n)
        for listener in order:
            nbrs = neighbors[listener]
            if len(nbrs) == 0:
                continue
            spoken = []
            for speaker in nbrs:
                labels = memory[speaker]
                # speaker rule: sample proportional to frequency in memory
                values, counts = np.unique(labels, return_counts=True)
                spoken.append(int(rng.choice(values, p=counts / counts.sum())))
            if not spoken:
                continue
            values, counts = np.unique(spoken, return_counts=True)
            # listener rule: adopt most popular spoken label (ties broken randomly)
            top = counts.max()
            candidates = values[counts == top]
            memory[listener].append(int(rng.choice(candidates)))

    # Global top-K communities by total memory mass
    flat = np.concatenate([np.asarray(m, dtype=int) for m in memory])
    values, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    top_labels = values[order[: int(n_communities)]]
    label_to_col = {int(lab): i for i, lab in enumerate(top_labels)}

    F = np.zeros((n, int(n_communities)), dtype=np.float64)
    for i, mem in enumerate(memory):
        vals, cnts = np.unique(np.asarray(mem, dtype=int), return_counts=True)
        total = float(cnts.sum())
        for lab, cnt in zip(vals, cnts):
            col = label_to_col.get(int(lab))
            if col is not None:
                F[i, col] = float(cnt) / total
        if F[i].sum() <= 0:
            F[i] = 1.0 / float(n_communities)
        else:
            F[i] /= F[i].sum()
    return F


# ---------------------------------------------------------------------------
# CESNA-lite: graph Berpo + attribute reconstruction (expression → F)
# ---------------------------------------------------------------------------


def fit_cesna_lite(
    adjacency,
    X,
    n_communities=5,
    iterations=200,
    lr=0.05,
    n_neg=8,
    seed=42,
    attr_weight=0.5,
):
    """Attribute-aware affiliation: BIGCLAM graph loss + linear attribute fit.

    Approximates CESNA's joint community model without categorical attributes:
    continuous expression is predicted as F @ B (non-negative least via gradient).
    """
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed))
    a = np.asarray(adjacency, dtype=np.float32)
    n = a.shape[0]
    k = int(n_communities)
    edge_index = _edge_index_from_adj(a)

    # Reduce attributes for speed/stability
    feats = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    n_pca = int(min(32, feats.shape[1], n - 1))
    feats = PCA(n_components=n_pca, random_state=seed).fit_transform(feats)
    y = torch.tensor(feats, dtype=torch.float32)

    F_param = nn.Parameter(torch.rand(n, k) * 0.1)
    B = nn.Parameter(torch.randn(k, n_pca) * 0.01)
    opt = torch.optim.Adam([F_param, B], lr=lr)

    best_F, best_loss = None, float("inf")
    for _ in range(int(iterations)):
        opt.zero_grad()
        F_pos = F_param.clamp(min=0.0)
        graph_loss = _berpo_neg_loss(F_pos, edge_index, n_neg=n_neg, generator=generator)
        pred = F_pos @ B
        attr_loss = torch.nn.functional.mse_loss(pred, y)
        loss = graph_loss + float(attr_weight) * attr_loss * n + 1e-3 * F_pos.abs().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            F_param.clamp_(min=0.0)
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_F = F_param.detach().cpu().numpy().copy()

    return np.maximum(best_F if best_F is not None else F_param.detach().cpu().numpy(), 0.0), float(best_loss)
