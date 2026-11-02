"""BIGCLAM model implementation with GPU support."""

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import NMF


def sigm(x):
    """Sigmoid-like function for BIGCLAM."""
    return torch.div(torch.exp(-1. * x), 1. - torch.exp(-1. * x))


def log_likelihood_offdiag(F, A):
    """BIGCLAM log-likelihood with self-pairs removed (unbalanced non-edges)."""
    affinity = torch.clamp(F @ F.T, max=50)
    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    off_diagonal = 1.0 - eye
    edge = A * off_diagonal
    nonedge = (1.0 - A) * off_diagonal
    edge_term = edge * torch.log(1.0 - torch.exp(-affinity) + 1e-10)
    nonedge_term = nonedge * affinity
    return edge_term.sum() - nonedge_term.sum()


def balanced_log_likelihood(F, A):
    """Edge vs non-edge terms reweighted so non-edges do not dominate.

    On a sparse patient graph almost every pair is a non-edge. The classic
    unweighted likelihood then drives every affiliation toward zero and the
    membership matrix collapses to a flat plateau. Scaling the non-edge
    penalty by ``n_edges / n_nonedges`` restores a usable balance.
    """
    affinity = torch.clamp(F @ F.T, max=50.0)
    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    off = 1.0 - eye
    edge = A * off
    nonedge = (1.0 - A) * off
    n_edges = torch.clamp(edge.sum(), min=1.0)
    n_nonedges = torch.clamp(nonedge.sum(), min=1.0)
    edge_ll = (edge * torch.log(1.0 - torch.exp(-affinity) + 1e-10)).sum()
    nonedge_pen = (nonedge * affinity).sum()
    return edge_ll - (n_edges / n_nonedges) * nonedge_pen


def neg_sampling_log_likelihood(F, edge_index, n_nodes, n_neg, generator):
    """BIGCLAM-style edge likelihood with negative samples (Yang & Leskovec)."""
    sources = edge_index[0]
    targets = edge_index[1]
    pos_dots = torch.clamp((F[sources] * F[targets]).sum(dim=1), max=50.0)
    pos_ll = torch.log(1.0 - torch.exp(-pos_dots) + 1e-10).sum()

    n_edges = sources.shape[0]
    if n_edges == 0:
        return pos_ll
    neg_sources = sources.repeat_interleave(int(n_neg))
    neg_targets = torch.randint(
        0, n_nodes, (n_edges * int(n_neg),), device=F.device, generator=generator
    )
    # Drop accidental self-pairs; residual collisions are rare and harmless.
    mask = neg_targets != neg_sources
    if mask.any():
        neg_dots = torch.clamp(
            (F[neg_sources[mask]] * F[neg_targets[mask]]).sum(dim=1), max=50.0
        )
        neg_pen = neg_dots.sum()
    else:
        neg_pen = torch.zeros((), device=F.device)
    # Match the scale of the positive term.
    return pos_ll - neg_pen * (float(n_edges) / float(max(int(mask.sum()), 1)))


def log_likelihood(F, A):
    """
    Compute log-likelihood for BIGCLAM model.
    
    Args:
        F (torch.Tensor): Membership strength matrix (N x C).
        A (torch.Tensor): Adjacency matrix (N x N).
        
    Returns:
        torch.Tensor: Log-likelihood value.
    """
    A_soft = F @ F.T  # Matrix multiplication in PyTorch
    # Clip A_soft to prevent overflow in exp
    A_soft_clipped = torch.clamp(A_soft, max=50)
    FIRST_PART = A * torch.log(1. - torch.exp(-1. * A_soft_clipped) + 1e-10)
    sum_edges = torch.sum(FIRST_PART)
    SECOND_PART = (1 - A) * A_soft
    sum_nedges = torch.sum(SECOND_PART)
    log_likeli = sum_edges - sum_nedges
    return log_likeli


def membership_peakiness(F):
    """Median second/first affiliation ratio. Near 1 means a flat failed fit."""
    membership = np.asarray(F, dtype=np.float64)
    if membership.ndim != 2 or membership.shape[1] < 2:
        return float("nan")
    ordered = np.sort(membership, axis=1)
    top = ordered[:, -1]
    second = ordered[:, -2]
    ratio = second / np.maximum(top, 1e-12)
    return float(np.median(ratio))


def _adjacency_edges(adjacency):
    sources, targets = np.where(np.triu(np.asarray(adjacency) > 0, k=1))
    if len(sources) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    both = np.vstack([sources, targets])
    # Undirected: keep both directions for neighbor-style updates.
    return np.hstack([both, both[::-1]])


def _nmf_init(adjacency, num_communities, seed):
    matrix = np.asarray(adjacency, dtype=np.float64) + 1e-6
    model = NMF(
        n_components=int(num_communities),
        init="nndsvda",
        random_state=int(seed),
        max_iter=400,
    )
    factors = model.fit_transform(matrix)
    scale = np.sqrt(np.maximum(factors.max(), 1e-6))
    return np.maximum(factors / scale, 1e-6)


class BIGCLAM:
    """BIGCLAM model class."""
    
    def __init__(self, device=None):
        """
        Initialize BIGCLAM model.
        
        Args:
            device (torch.device, optional): Device to run computation on. 
                                            If None, uses GPU if available.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
    
    def train(self, A, max_communities=10, iterations=100, lr=0.08):
        """
        Train BIGCLAM model to find optimal number of communities.
        
        Args:
            A (np.ndarray or torch.Tensor): Adjacency matrix (N x N).
            max_communities (int): Maximum number of communities to try.
            iterations (int): Number of optimization iterations per community number.
            lr (float): Learning rate for optimizer.
            
        Returns:
            tuple: (best_F, best_num_communities) where best_F is the membership 
                   strength matrix and best_num_communities is the optimal number of communities.
        """
        # Convert to tensor if needed
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float32, device=self.device)
        else:
            A = A.to(self.device)
        
        N = A.shape[0]
        best_F = None
        best_num_communities = 1
        best_aic = float('inf')
        
        for num_communities in range(1, max_communities + 1):
            # Initialize membership strength matrix
            F = torch.rand((N, num_communities), device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([F], lr=lr)
            
            for n in range(iterations):
                optimizer.zero_grad()
                ll = log_likelihood(F, A)
                loss = -ll  # Minimize negative log-likelihood
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    # Ensure F is nonnegative
                    F.data = torch.clamp(F.data, min=1e-12)
            
            # Calculate AIC
            k = N * num_communities  # Number of parameters
            aic = -2 * ll.item() + 2 * k
            
            if aic < best_aic:
                best_aic = aic
                best_F = F.detach().cpu().numpy()
                best_num_communities = num_communities
            
            print(f'At step {num_communities}/{max_communities}, AIC: {aic:.4f}')
        
        print(f"Best number of communities: {best_num_communities} (AIC: {best_aic:.4f})")
        return best_F, best_num_communities

    def fit_c(
        self,
        A,
        num_communities,
        iterations=250,
        lr=0.05,
        restarts=5,
        seed=42,
        l1=0.01,
        n_neg=8,
        max_peak_ratio=0.85,
    ):
        """Fit a fixed community count with peaked memberships.

        Uses NMF warm-start, softplus parameters, L1 sparsity, and either
        balanced full likelihood (small graphs) or negative sampling (large).
        Restarts are ranked by peakiness first, then by likelihood, so a flat
        high-likelihood plateau cannot win.
        """
        if isinstance(A, np.ndarray):
            adjacency_np = np.asarray(A, dtype=np.float32)
        else:
            adjacency_np = A.detach().cpu().numpy().astype(np.float32)
        adjacency = torch.tensor(adjacency_np, dtype=torch.float32, device=self.device)
        n_nodes = adjacency.shape[0]
        edges = _adjacency_edges(adjacency_np)
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device)
        use_neg = n_nodes >= 400
        best = None

        for restart in range(int(restarts)):
            restart_seed = int(seed) + 17 * restart
            torch.manual_seed(restart_seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(restart_seed)

            init = _nmf_init(adjacency_np, num_communities, restart_seed)
            if restart > 0:
                noise = np.random.default_rng(restart_seed).random(init.shape) * 0.05
                init = np.maximum(init + noise, 1e-6)
            # softplus^{-1}(x) ≈ log(exp(x)-1); clamp for stability.
            raw = np.log(np.expm1(np.clip(init, 1e-4, 20.0)))
            weights = torch.tensor(raw, dtype=torch.float32, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([weights], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(int(iterations), 1)
            )

            for _ in range(int(iterations)):
                optimizer.zero_grad()
                membership = torch.nn.functional.softplus(weights) + 1e-8
                if use_neg:
                    likelihood = neg_sampling_log_likelihood(
                        membership, edge_index, n_nodes, n_neg, generator
                    )
                else:
                    likelihood = balanced_log_likelihood(membership, adjacency)
                loss = -likelihood + float(l1) * membership.sum()
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                membership = (torch.nn.functional.softplus(weights) + 1e-8).detach()
                if use_neg:
                    # Report the balanced full likelihood for comparability.
                    final_ll = float(balanced_log_likelihood(membership, adjacency).item())
                else:
                    final_ll = float(balanced_log_likelihood(membership, adjacency).item())
                final_F = membership.cpu().numpy()
            peak = membership_peakiness(final_F)
            candidate = {
                "F": final_F,
                "ll": final_ll,
                "peak_ratio": peak,
                "restart": restart,
            }
            if best is None:
                best = candidate
            else:
                # Prefer peaked solutions; only then maximize likelihood.
                better_peak = peak < best["peak_ratio"] - 0.02
                similar_peak = abs(peak - best["peak_ratio"]) <= 0.02
                better_ll = final_ll > best["ll"]
                if better_peak or (similar_peak and better_ll):
                    best = candidate

        if best is None:
            raise RuntimeError("BIGCLAM fit produced no membership matrix")
        if best["peak_ratio"] > float(max_peak_ratio):
            print(
                f"  [warn] best peak ratio {best['peak_ratio']:.3f} still exceeds "
                f"{max_peak_ratio}; memberships may be under-separated"
            )
        return best["F"], best["ll"], best["peak_ratio"]

    def visualize_bipartite(self, F, p2c=None, save_path=None):
        """
        Visualize bipartite graph of people and communities.
        
        Args:
            F (np.ndarray): Membership strength matrix (N x C).
            p2c (list, optional): Community assignments for visualization.
            save_path (str, optional): Path to save the figure.
        """
        B = nx.Graph()
        N, C = F.shape
        
        person_nodes = [f"Person {i}" for i in range(N)]
        community_nodes = [f"Community {i}" for i in range(C)]
        
        B.add_nodes_from(person_nodes, bipartite=0)
        B.add_nodes_from(community_nodes, bipartite=1)
        
        for i, comm_prefs in enumerate(F):
            for c, membership in enumerate(comm_prefs):
                if membership > 0.01:  # Threshold for visualization
                    B.add_edge(person_nodes[i], community_nodes[c])
        
        pos = nx.drawing.layout.bipartite_layout(B, person_nodes)
        plt.figure(figsize=(12, 8))
        nx.draw(
            B, pos, with_labels=True,
            node_color=['skyblue' if n in person_nodes else 'lightgreen' for n in B.nodes()],
            edge_color='gray'
        )
        plt.title('Bipartite Graph of People and Communities')
        
        if save_path:
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def train_bigclam(A, max_communities=10, iterations=100, lr=0.08, device=None):
    """
    Convenience function to train BIGCLAM model.
    
    Args:
        A (np.ndarray or torch.Tensor): Adjacency matrix.
        max_communities (int): Maximum number of communities to try.
        iterations (int): Number of optimization iterations.
        lr (float): Learning rate.
        device (torch.device, optional): Device to use.
        
    Returns:
        tuple: (best_F, best_num_communities) membership matrix and optimal communities.
    """
    model = BIGCLAM(device=device)
    return model.train(A, max_communities, iterations, lr)


def fit_bigclam_c(
    A,
    num_communities,
    iterations=250,
    lr=0.05,
    restarts=5,
    seed=42,
    device=None,
    l1=0.01,
    n_neg=8,
    max_peak_ratio=0.85,
):
    """Fit BIGCLAM at a caller-chosen community count.

    Returns ``(F, log_likelihood, peak_ratio)``.
    """
    model = BIGCLAM(device=device)
    return model.fit_c(
        A,
        num_communities=num_communities,
        iterations=iterations,
        lr=lr,
        restarts=restarts,
        seed=seed,
        l1=l1,
        n_neg=n_neg,
        max_peak_ratio=max_peak_ratio,
    )
