"""Three-cohort BIGCLAM study.

Discovery is GSE96058. Gene list, neighbor count, overlap rule, and community
count are locked there. TCGA-BRCA and METABRIC are refit with those settings
and are not used to choose them.

The primary outputs are survival models and subtype agreement. MLP/SVM accuracy
is not part of this script.

Usage (from the project root):
    python run_q1_study.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.q1_clustering import (  # noqa: E402
    baseline_labels,
    choose_overlap_ratio,
    fit_cohort,
    select_c_by_stability,
)
from src.evaluation.q1_survival import (  # noqa: E402
    evaluate_partition,
    marker_table,
    plot_kaplan_meier,
)
from src.preprocessing.harmonize import (  # noqa: E402
    harmonize_loaded,
    load_harmonized,
    save_harmonized,
)
from src.preprocessing.cohorts import PROJECT_ROOT, load_q1_cohorts  # noqa: E402


def _load_q1(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if "q1" not in raw:
        raise KeyError(f"{config_path} has no q1 section")
    return raw["q1"]


def _apply_overrides(q1, args):
    if args.n_genes is not None:
        q1["n_genes"] = args.n_genes
    if args.bootstraps is not None:
        q1["bootstraps"] = args.bootstraps
    if args.stability_iterations is not None:
        q1["stability_iterations"] = args.stability_iterations
    if args.fit_iterations is not None:
        q1["fit_iterations"] = args.fit_iterations
    if args.restarts is not None:
        q1["restarts"] = args.restarts
    if args.communities:
        q1["communities"] = [int(part) for part in args.communities.split(",") if part.strip()]
    if args.output_dir:
        q1["results_dir"] = args.output_dir
    if args.harmonized_dir:
        q1["harmonized_dir"] = args.harmonized_dir
    q1["max_samples"] = args.max_samples
    return q1


def _results_dir(q1):
    path = PROJECT_ROOT / q1.get("results_dir", "results/q1")
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_prepare(q1):
    print("=" * 80)
    print("Preparing cohorts and locking the discovery gene list")
    print("=" * 80)
    loaded = load_q1_cohorts(q1, max_samples=q1.get("max_samples"))
    harmonized = harmonize_loaded(
        loaded,
        n_genes=q1.get("n_genes", 2000),
        discovery=q1.get("discovery_cohort", "gse96058"),
        min_shared_genes=q1.get("min_shared_genes", 200),
    )
    save_harmonized(
        harmonized,
        q1.get("harmonized_dir", "data/harmonized"),
        metabric_dir=q1.get("metabric", {}).get("output_dir", "data/metabric"),
    )
    return harmonized


def stage_discover(q1, harmonized):
    print("=" * 80)
    print("Selecting community count on the discovery cohort")
    print("=" * 80)
    discovery = q1.get("discovery_cohort", "gse96058")
    cohort = harmonized["cohorts"][discovery]
    fit_kwargs = dict(
        knn_k=q1.get("knn_k", 15),
        iterations=q1.get("fit_iterations", 250),
        restarts=q1.get("restarts", 5),
        lr=q1.get("learning_rate", 0.05),
        seed=q1.get("seed", 42),
        l1=q1.get("l1", 0.01),
        n_neg=q1.get("n_neg", 8),
        max_peak_ratio=q1.get("max_peak_ratio", 0.85),
    )
    chosen, table, peak_table = select_c_by_stability(
        cohort["X"],
        communities=q1.get("communities", [4, 5, 6, 7, 8]),
        knn_k=fit_kwargs["knn_k"],
        n_boot=q1.get("bootstraps", 8),
        fraction=q1.get("bootstrap_fraction", 0.8),
        iterations=q1.get("stability_iterations", 120),
        restarts=q1.get("stability_restarts", 2),
        lr=fit_kwargs["lr"],
        seed=fit_kwargs["seed"],
        l1=fit_kwargs["l1"],
        min_ari=q1.get("min_stability_ari", 0.05),
    )
    print(f"Fitting discovery BIGCLAM at C={chosen}")
    provisional = fit_cohort(
        cohort["X"],
        num_communities=chosen,
        overlap_ratio=0.75,
        **fit_kwargs,
    )
    overlap_ratio, _ = choose_overlap_ratio(provisional["membership_ratio"])
    provisional["mixed"] = provisional["membership_ratio"] >= overlap_ratio
    provisional["mixed_fraction"] = float(np.mean(provisional["mixed"]))
    print(
        f"  Discovery peak_ratio={provisional['peak_ratio']:.3f}; "
        f"locked overlap_ratio={overlap_ratio:.2f} "
        f"(mixed fraction {provisional['mixed_fraction']:.3f})"
    )
    if provisional["peak_ratio"] > fit_kwargs["max_peak_ratio"]:
        print(
            "  [warn] discovery memberships are still too flat for a Q1 overlap claim"
        )
    locked = {
        "discovery_cohort": discovery,
        "num_communities": int(chosen),
        "stability_ari": {str(k): v for k, v in table.items()},
        "stability_peak_ratio": {str(k): v for k, v in peak_table.items()},
        "n_genes": len(harmonized["genes"]),
        "knn_k": int(fit_kwargs["knn_k"]),
        "overlap_ratio": float(overlap_ratio),
        "discovery_mixed_fraction": float(provisional["mixed_fraction"]),
        "discovery_peak_ratio": float(provisional["peak_ratio"]),
        "fit_iterations": int(fit_kwargs["iterations"]),
        "restarts": int(fit_kwargs["restarts"]),
        "learning_rate": float(fit_kwargs["lr"]),
        "l1": float(fit_kwargs["l1"]),
        "n_neg": int(fit_kwargs["n_neg"]),
        "max_peak_ratio": float(fit_kwargs["max_peak_ratio"]),
        "seed": int(fit_kwargs["seed"]),
        "max_samples": q1.get("max_samples"),
        "graph": "symmetric cosine kNN, components joined by the strongest bridge",
        "gene_rule": "intersection, then top discovery variance, then within-cohort z-score",
        "likelihood": "balanced full LL (n<400) or neg-sampling + L1 + NMF init",
    }
    results = _results_dir(q1)
    (results / "locked_config.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "num_communities": k,
                "mean_pairwise_ari": table[k],
                "mean_peak_ratio": peak_table.get(k, np.nan),
            }
            for k in sorted(table)
        ]
    ).to_csv(results / "stability.csv", index=False)
    _save_fit(results, discovery, cohort, provisional, harmonized["genes"])
    print(f"Locked settings written to {results / 'locked_config.json'}")
    return locked, {discovery: provisional}


def stage_evaluate(q1, harmonized, locked, cached_fits=None):
    print("=" * 80)
    print("Evaluating locked settings on every cohort")
    print("=" * 80)
    results = _results_dir(q1)
    cached_fits = cached_fits or {}
    rows = []
    summary = {"locked": locked, "cohorts": {}}
    for name, cohort in harmonized["cohorts"].items():
        if name in cached_fits:
            fit = cached_fits[name]
        else:
            print(f"Fitting {name} at C={locked['num_communities']}")
            fit = fit_cohort(
                cohort["X"],
                num_communities=locked["num_communities"],
                knn_k=locked["knn_k"],
                iterations=locked["fit_iterations"],
                restarts=locked["restarts"],
                lr=locked["learning_rate"],
                seed=locked["seed"],
                overlap_ratio=locked["overlap_ratio"],
                l1=locked.get("l1", 0.01),
                n_neg=locked.get("n_neg", 8),
                max_peak_ratio=locked.get("max_peak_ratio", 0.85),
            )
            _save_fit(results, name, cohort, fit, harmonized["genes"])
        clinical = cohort["clinical"]
        rows.append(evaluate_partition(clinical, fit["community"], "bigclam", name))
        if 0.0 < float(np.mean(fit["mixed"])) < 1.0:
            rows.append(
                evaluate_partition(clinical, fit["mixed"].astype(int), "bigclam_mixed", name)
            )
        else:
            print(
                f"  [warn] {name}: mixed fraction {float(np.mean(fit['mixed'])):.3f} "
                "has no contrast; skipping mixed survival contrast"
            )
        print(f"  Baselines for {name}")
        baselines, baseline_k = baseline_labels(
            cohort["X"],
            fit["A"],
            locked["num_communities"],
            seed=locked["seed"],
            knn_k=locked.get("knn_k", 15),
        )
        for method, labels in baselines.items():
            row = evaluate_partition(clinical, labels, method, name)
            row["n_clusters_found"] = baseline_k[method]
            rows.append(row)
        plot_kaplan_meier(
            clinical,
            fit["community"],
            results / f"{name}_km_community.png",
            f"{name} BIGCLAM communities",
        )
        plot_kaplan_meier(
            clinical,
            np.where(fit["mixed"], "mixed", "single"),
            results / f"{name}_km_mixed.png",
            f"{name} mixed membership",
        )
        markers = marker_table(cohort["X"], harmonized["genes"], fit["community"])
        markers.to_csv(results / f"{name}_markers.csv", index=False)
        crosstab = pd.crosstab(
            pd.Series(fit["community"], name="community"),
            clinical["PAM50"].astype(str),
        )
        crosstab.to_csv(results / f"{name}_pam50_crosstab.csv")
        summary["cohorts"][name] = {
            "n_samples": int(len(cohort["samples"])),
            "mixed_fraction": float(np.mean(fit["mixed"])),
            "peak_ratio": float(fit.get("peak_ratio", np.nan)),
            "log_likelihood": fit["log_likelihood"],
            "baseline_cluster_counts": baseline_k,
        }
        print(
            f"  {name}: peak_ratio={summary['cohorts'][name]['peak_ratio']:.3f} "
            f"mixed={summary['cohorts'][name]['mixed_fraction']:.3f}"
        )
    table = pd.DataFrame(rows)
    table.to_csv(results / "survival_comparison.csv", index=False)
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results written to {results}")
    return table


def _save_fit(results, name, cohort, fit, genes):
    np.save(results / f"{name}_F.npy", fit["F"])
    frame = cohort["clinical"].copy()
    frame.insert(0, "sample_id", cohort["samples"])
    frame["community"] = fit["community"]
    frame["mixed"] = fit["mixed"].astype(int)
    frame["membership_ratio"] = fit["membership_ratio"]
    frame.to_csv(results / f"{name}_assignments.csv", index=False)
    pd.Series(genes, name="gene").to_csv(results / f"{name}_genes_used.csv", index=False)
    meta = {
        "peak_ratio": float(fit.get("peak_ratio", np.nan)),
        "mixed_fraction": float(np.mean(fit["mixed"])),
        "log_likelihood": float(fit["log_likelihood"]),
        "n_communities": int(fit["F"].shape[1]),
    }
    (results / f"{name}_fit_quality.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the three-cohort BIGCLAM study")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "discover", "evaluate"],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-genes", type=int, default=None)
    parser.add_argument("--bootstraps", type=int, default=None)
    parser.add_argument("--stability-iterations", type=int, default=None)
    parser.add_argument("--fit-iterations", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--communities", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--harmonized-dir", type=str, default=None)
    args = parser.parse_args()

    q1 = _apply_overrides(_load_q1(args.config), args)
    harmonized = None
    locked = None
    cached = None
    if args.stage in {"all", "prepare"}:
        harmonized = stage_prepare(q1)
    if args.stage in {"all", "discover", "evaluate"} and harmonized is None:
        harmonized = load_harmonized(q1.get("harmonized_dir", "data/harmonized"))
    if args.stage in {"all", "discover"}:
        locked, cached = stage_discover(q1, harmonized)
    if args.stage == "evaluate":
        lock_path = _results_dir(q1) / "locked_config.json"
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
    if args.stage in {"all", "evaluate"}:
        stage_evaluate(q1, harmonized, locked, cached_fits=cached)


if __name__ == "__main__":
    main()
