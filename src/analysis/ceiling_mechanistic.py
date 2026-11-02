"""Mechanistic checks on locked SNF graphs.

Density sweep, loss/regularization ablation, a small mixed-membership
simulation, and a soft single-view ablation. These fits use a reduced
training budget so the comparison is at fixed compute; they do not replace
the locked 250-epoch bake-off.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src.analysis.overlapping_sota import fit_nocd, fit_slpa
from src.analysis.q1_clustering import membership_summary
from src.analysis.snf_bigclam import (
    build_view_affinities,
    membership_entropy,
    snf_affinity_to_graph,
)
from src.bigclam.bigclam_model import fit_bigclam_c
from src.evaluation.q1_survival import cox_hard_plus_soft, subtype_enrichment
from src.preprocessing.cohorts import PROJECT_ROOT

K_GRID = (5, 10, 15, 30)
LOCKED_K = 15
LOCKED_RATIO = {"slpa": 0.80, "bigclam": 0.50, "nocd_x": 0.60, "nocd_g": 0.90}
SWEEP_BIGCLAM_ITERS = 80
SWEEP_NOCD_EPOCHS = 60
N_COMMUNITIES = 5


def bh_qvalues(p_values):
    """Benjamini–Hochberg q-values. Non-finite p-values stay missing."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    mono = np.minimum.accumulate(raw[::-1])[::-1]
    mono = np.clip(mono, 0.0, 1.0)
    placed = np.empty(m)
    placed[order] = mono
    q[valid] = placed
    return q


def edge_stats(adjacency):
    """Undirected edge count, density, and mean degree."""
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    n_edges = int(np.triu(adj > 0, k=1).sum())
    possible = n * (n - 1) / 2.0
    density = float(n_edges / possible) if possible else np.nan
    mean_degree = float(2.0 * n_edges / n) if n else np.nan
    return {"n": int(n), "n_edges": n_edges, "density": density, "mean_degree": mean_degree}


def soft_quality(F, ratio):
    """Mixed fraction, median second/first ratio, and median membership entropy."""
    _, mixed, strength = membership_summary(F, ratio=ratio)
    entropy = membership_entropy(F)
    return {
        "mixed_fraction": float(np.mean(mixed)),
        "median_peak_ratio": float(np.median(strength)),
        "median_entropy": float(np.median(entropy)),
        "n_mixed": int(np.sum(mixed)),
    }


def _fit_engine(engine, adjacency, X, seed=42, bigclam_l1=0.01, nocd_epochs=SWEEP_NOCD_EPOCHS, feature_mode="attributes"):
    if engine == "slpa":
        F = fit_slpa(adjacency, n_communities=N_COMMUNITIES, iterations=40, seed=seed)
        return F
    if engine == "bigclam":
        F, _, _ = fit_bigclam_c(
            adjacency,
            num_communities=N_COMMUNITIES,
            iterations=SWEEP_BIGCLAM_ITERS,
            restarts=1,
            seed=seed,
            l1=bigclam_l1,
        )
        return F
    if engine in {"nocd_x", "nocd_g"}:
        mode = "graph" if engine == "nocd_g" else feature_mode
        features = None if mode == "graph" else X
        F, _ = fit_nocd(
            adjacency,
            X=features,
            n_communities=N_COMMUNITIES,
            epochs=int(nocd_epochs),
            seed=seed,
            feature_mode="graph" if features is None else "attributes",
        )
        return F
    raise ValueError(engine)


def cache_locked_graphs(harmonized, out_dir, knn_k=LOCKED_K, snf_iterations=15):
    """Write fused and single-view affinities plus the locked kNN graphs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"knn_k": int(knn_k), "snf_iterations": int(snf_iterations), "cohorts": {}}
    for name, cohort in harmonized["cohorts"].items():
        fused_path = out / f"{name}_fused_affinity.npy"
        if fused_path.exists():
            print(f"  graphs cached: {name}")
            manifest["cohorts"][name] = json.loads((out / f"{name}_graph_meta.json").read_text(encoding="utf-8"))
            continue
        print(f"  building SNF views: {name} n={cohort['X'].shape[0]}")
        views = build_view_affinities(cohort["X"], knn_k=knn_k, snf_iterations=snf_iterations)
        meta = {"n": int(cohort["X"].shape[0]), "graphs": {}}
        for view, affinity in views.items():
            np.save(out / f"{name}_{view}_affinity.npy", np.asarray(affinity, dtype=np.float32))
            graph = snf_affinity_to_graph(affinity, knn_k=knn_k)
            np.save(out / f"{name}_{view}_k{knn_k}.npy", graph)
            meta["graphs"][view] = edge_stats(graph)
        (out / f"{name}_graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest["cohorts"][name] = meta
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_fused_affinity(graph_dir, cohort):
    return np.load(Path(graph_dir) / f"{cohort}_fused_affinity.npy")


def _luma_and_residual(F, clinical, hard_labels, ratio):
    _, mixed, _ = membership_summary(F, ratio=ratio)
    enrich = subtype_enrichment(clinical, mixed, focus="LumA")
    residual = cox_hard_plus_soft(clinical, hard_labels, mixed.astype(float), soft_name="mixed")
    return {
        "luma_or": enrich.get("odds_ratio", np.nan),
        "luma_p": enrich.get("p", np.nan),
        "residual_p": residual.get("p", np.nan),
        "residual_n": residual.get("n", 0),
    }


def density_sweep(fused_affinity, X, clinical, hard_labels, out_csv, ks=K_GRID):
    """Refit SLPA, BIGCLAM, and NOCD-X across kNN sparsifications of one fused affinity."""
    rows = []
    engines = ("slpa", "bigclam", "nocd_x")
    for k in ks:
        graph = snf_affinity_to_graph(fused_affinity, knn_k=int(k))
        stats = edge_stats(graph)
        print(f"  density k={k} edges={stats['n_edges']} degree={stats['mean_degree']:.2f}")
        for engine in engines:
            print(f"    fit {engine}")
            F = _fit_engine(engine, graph, X)
            quality = soft_quality(F, LOCKED_RATIO[engine])
            survival = _luma_and_residual(F, clinical, hard_labels, LOCKED_RATIO[engine])
            rows.append({
                "experiment": "density_sweep",
                "engine": engine,
                "k": int(k),
                "l1": 0.01 if engine == "bigclam" else np.nan,
                "epochs": SWEEP_NOCD_EPOCHS if engine == "nocd_x" else np.nan,
                "fit_budget": f"bigclam_iters={SWEEP_BIGCLAM_ITERS},nocd_epochs={SWEEP_NOCD_EPOCHS},restarts=1",
                **stats,
                **quality,
                **survival,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    return frame


def loss_ablation(graph, X, clinical, hard_labels, out_csv):
    """BIGCLAM L1 grid and NOCD feature-source / epoch grid on one adjacency."""
    rows = []
    print("  BIGCLAM L1 grid")
    for l1 in (0.0, 0.001, 0.01, 0.1):
        F = _fit_engine("bigclam", graph, X, bigclam_l1=l1)
        quality = soft_quality(F, LOCKED_RATIO["bigclam"])
        survival = _luma_and_residual(F, clinical, hard_labels, LOCKED_RATIO["bigclam"])
        rows.append({
            "experiment": "loss_ablation",
            "engine": "bigclam",
            "l1": l1,
            "feature_mode": "graph",
            "epochs": SWEEP_BIGCLAM_ITERS,
            **quality,
            **survival,
        })
    print("  NOCD feature and epoch grid")
    for mode, epochs in (("attributes", 40), ("attributes", SWEEP_NOCD_EPOCHS), ("graph", SWEEP_NOCD_EPOCHS)):
        engine = "nocd_x" if mode == "attributes" else "nocd_g"
        F = _fit_engine(engine, graph, X, nocd_epochs=epochs, feature_mode=mode)
        quality = soft_quality(F, LOCKED_RATIO[engine])
        survival = _luma_and_residual(F, clinical, hard_labels, LOCKED_RATIO[engine])
        rows.append({
            "experiment": "loss_ablation",
            "engine": engine,
            "l1": np.nan,
            "feature_mode": mode,
            "epochs": int(epochs),
            **quality,
            **survival,
        })
    frame = pd.DataFrame(rows)
    frame["residual_q"] = bh_qvalues(frame["residual_p"].to_numpy())
    frame.to_csv(out_csv, index=False)
    return frame


def _match_degree_graph(n, k, target_degree, seed):
    """Mixed-membership graph whose mean degree matches a kNN patient graph."""
    rng = np.random.default_rng(int(seed))
    membership = rng.gamma(0.35, 1.0, size=(n, k))
    membership = membership / membership.sum(axis=1, keepdims=True)
    dots = membership @ membership.T
    np.fill_diagonal(dots, 0.0)
    lo, hi = 1e-4, 80.0
    target_density = target_degree / max(n - 1, 1)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        prob = 1.0 - np.exp(-mid * dots)
        if float(prob.mean()) > target_density:
            hi = mid
        else:
            lo = mid
    scale = 0.5 * (lo + hi)
    prob = 1.0 - np.exp(-scale * dots)
    np.fill_diagonal(prob, 0.0)
    upper = rng.random((n, n))
    adjacency = (np.triu(upper, k=1) < np.triu(prob, k=1)).astype(np.float32)
    adjacency = adjacency + adjacency.T
    return adjacency, membership


def _aligned_correlation(true_F, est_F):
    left = np.asarray(true_F, dtype=np.float64)
    right = np.asarray(est_F, dtype=np.float64)
    k_left, k_right = left.shape[1], right.shape[1]
    corr = np.zeros((k_left, k_right))
    for i in range(k_left):
        for j in range(k_right):
            a = left[:, i]
            b = right[:, j]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(a, b)[0, 1])
    corr = np.nan_to_num(corr, nan=0.0)
    rows, cols = linear_sum_assignment(-corr)
    return float(np.mean(corr[rows, cols]))


def synthetic_recovery(target_degrees, out_csv, n=300, n_reps=4, seed=7):
    """Recover known soft memberships at mean degrees taken from the real graphs."""
    rows = []
    engines = ("slpa", "bigclam", "nocd_x")
    for degree in target_degrees:
        for rep in range(int(n_reps)):
            graph, truth = _match_degree_graph(n, N_COMMUNITIES, degree, seed=seed + rep + int(degree * 10))
            observed = edge_stats(graph)
            _, true_mixed, _ = membership_summary(truth, ratio=0.60)
            print(f"  synthetic degree={degree:.2f} rep={rep} edges={observed['n_edges']}")
            for engine in engines:
                # NOCD-X on a graph with no expression uses graph features.
                F = _fit_engine(
                    "nocd_g" if engine == "nocd_x" else engine,
                    graph,
                    X=None,
                    seed=seed + rep,
                    feature_mode="graph",
                )
                if engine == "nocd_x":
                    # Record the graph-feature NOCD fit under the sweep name.
                    pass
                quality = soft_quality(F, LOCKED_RATIO["nocd_g" if engine == "nocd_x" else engine])
                _, est_mixed, _ = membership_summary(F, ratio=0.60)
                rows.append({
                    "experiment": "mmsb",
                    "engine": "nocd_graph" if engine == "nocd_x" else engine,
                    "target_mean_degree": float(degree),
                    "rep": int(rep),
                    "n": int(n),
                    "mean_degree": observed["mean_degree"],
                    "density": observed["density"],
                    "F_correlation": _aligned_correlation(truth, F),
                    "mixed_ari": float(adjusted_rand_score(true_mixed.astype(int), est_mixed.astype(int))),
                    **quality,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    return frame


def view_ablation(graph_dir, cohort, X, clinical, hard_labels, out_csv):
    """Soft engines on Euclidean-only, correlation-only, and fused kNN graphs."""
    rows = []
    root = Path(graph_dir)
    for view in ("euclidean", "correlation", "fused"):
        graph = np.load(root / f"{cohort}_{view}_k{LOCKED_K}.npy")
        stats = edge_stats(graph)
        print(f"  view {view}")
        for engine in ("slpa", "bigclam", "nocd_x"):
            F = _fit_engine(engine, graph, X)
            quality = soft_quality(F, LOCKED_RATIO[engine])
            survival = _luma_and_residual(F, clinical, hard_labels, LOCKED_RATIO[engine])
            rows.append({
                "experiment": "view_ablation",
                "view": view,
                "engine": engine,
                **stats,
                **quality,
                **survival,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    return frame


def plot_density(frame, output_path):
    """Mean-degree versus peak ratio and mixed fraction for the three engines."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engines = ["slpa", "bigclam", "nocd_x"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for engine in engines:
        part = frame[frame["engine"] == engine].sort_values("mean_degree")
        axes[0].plot(part["mean_degree"], part["median_peak_ratio"], marker="o", label=engine)
        axes[1].plot(part["mean_degree"], part["mixed_fraction"], marker="o", label=engine)
    axes[0].axhline(0.85, color="0.5", ls="--", lw=0.8)
    axes[0].set_xlabel("Mean degree of sparsified SNF graph")
    axes[0].set_ylabel("Median second/first affiliation")
    axes[0].set_title("Peak ratio (1 = consensus)")
    axes[1].axhspan(0.10, 0.40, color="0.85", zorder=0)
    axes[1].set_xlabel("Mean degree of sparsified SNF graph")
    axes[1].set_ylabel("Mixed fraction at locked ratio")
    axes[1].set_title("Mixed fraction (band = 10–40%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def slpa_collapse_reversed(density_frame):
    """True when sparser graphs restore peaked SLPA overlap relative to k=15."""
    slpa = density_frame[density_frame["engine"] == "slpa"].copy()
    if slpa.empty or (slpa["k"] == LOCKED_K).sum() == 0:
        return False, "SLPA sweep missing"
    locked = slpa.loc[slpa["k"] == LOCKED_K].iloc[0]
    sparser = slpa[slpa["k"] < LOCKED_K]
    if sparser.empty:
        return False, "no sparser graphs"
    best = sparser.sort_values("median_peak_ratio").iloc[0]
    collapsed = float(locked["median_peak_ratio"]) >= 0.90
    restored = float(best["median_peak_ratio"]) < 0.85
    mixed_moves = float(best["mixed_fraction"]) + 0.05 < float(locked["mixed_fraction"]) or (
        0.10 <= float(best["mixed_fraction"]) <= 0.40 and float(locked["mixed_fraction"]) > 0.40
    )
    detail = (
        f"SLPA k=15 peak={float(locked['median_peak_ratio']):.3f} "
        f"mixed={float(locked['mixed_fraction']):.3f}; "
        f"best sparser k={int(best['k'])} peak={float(best['median_peak_ratio']):.3f} "
        f"mixed={float(best['mixed_fraction']):.3f}"
    )
    return bool(collapsed and restored and mixed_moves), detail


def run_mechanistic(harmonized, graph_dir, out_dir, cohort="metabric"):
    """Density sweep, ablation, simulation, and view ablation for one cohort."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort_data = harmonized["cohorts"][cohort]
    X = cohort_data["X"]
    clinical = cohort_data["clinical"]
    fused = load_fused_affinity(graph_dir, cohort)
    from src.analysis.snf_bigclam import snf_hard_labels

    hard = snf_hard_labels(fused, N_COMMUNITIES, seed=42)
    np.save(out / f"{cohort}_hard_labels.npy", hard)

    density = density_sweep(fused, X, clinical, hard, out / "density_sweep.csv")
    plot_density(density, out / "figures" / "density_sweep.png")
    graph15 = snf_affinity_to_graph(fused, knn_k=LOCKED_K)
    loss_ablation(graph15, X, clinical, hard, out / "loss_ablation.csv")
    degrees = sorted({round(float(v), 2) for v in density.loc[density["engine"] == "slpa", "mean_degree"]})
    # Two degrees are enough to see whether recovery depends on sparsity.
    chosen = [degrees[0], degrees[-1]] if len(degrees) >= 2 else degrees
    synthetic_recovery(chosen, out / "synthetic_recovery.csv")
    view_ablation(graph_dir, cohort, X, clinical, hard, out / "view_ablation.csv")
    reversed_flag, detail = slpa_collapse_reversed(density)
    summary = {"slpa_collapse_reversed": reversed_flag, "slpa_detail": detail, "cohort": cohort}
    (out / "mechanistic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def project_results_dir(relative):
    path = Path(relative)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
