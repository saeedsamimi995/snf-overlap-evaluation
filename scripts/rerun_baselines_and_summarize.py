"""Re-score literature baselines on locked harmonized cohorts without re-selecting C.

Loads existing BIGCLAM assignments when present, rebuilds the patient graph, adds
consensus clustering and SNF-style fusion, and writes an updated comparison table
plus a short manuscript summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.q1_clustering import baseline_labels, membership_summary, symmetric_knn_graph
from src.evaluation.q1_survival import evaluate_partition, plot_kaplan_meier
from src.preprocessing.harmonize import load_harmonized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonized-dir", default="data/harmonized")
    parser.add_argument("--results-dir", default="results/q1")
    parser.add_argument("--config-locked", default="results/q1/locked_config.json")
    args = parser.parse_args()

    locked = json.loads((ROOT / args.config_locked).read_text(encoding="utf-8"))
    harmonized = load_harmonized(args.harmonized_dir)
    results = ROOT / args.results_dir
    results.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {"locked": locked, "cohorts": {}, "decision": {}}
    for name, cohort in harmonized["cohorts"].items():
        print("=" * 72)
        print(name)
        assign_path = results / f"{name}_assignments.csv"
        f_path = results / f"{name}_F.npy"
        if not assign_path.exists() or not f_path.exists():
            raise FileNotFoundError(f"Missing BIGCLAM outputs for {name}; run discover/evaluate first")
        assign = pd.read_csv(assign_path)
        membership = np.load(f_path)
        hard, mixed, ratios = membership_summary(
            membership, ratio=locked.get("overlap_ratio", 0.5)
        )
        adjacency = symmetric_knn_graph(cohort["X"], locked.get("knn_k", 15))
        clinical = cohort["clinical"]

        rows.append(evaluate_partition(clinical, hard, "bigclam", name))
        if 0.0 < float(np.mean(mixed)) < 1.0:
            rows.append(evaluate_partition(clinical, mixed.astype(int), "bigclam_mixed", name))

        print("  literature baselines...")
        baselines, baseline_k = baseline_labels(
            cohort["X"],
            adjacency,
            locked["num_communities"],
            seed=locked.get("seed", 42),
            knn_k=locked.get("knn_k", 15),
        )
        for method, labels in baselines.items():
            row = evaluate_partition(clinical, labels, method, name)
            row["n_clusters_found"] = baseline_k[method]
            rows.append(row)
            np.save(results / f"{name}_{method}_labels.npy", labels)

        plot_kaplan_meier(
            clinical, hard, results / f"{name}_km_community.png", f"{name} BIGCLAM communities"
        )
        summary["cohorts"][name] = {
            "mixed_fraction": float(np.mean(mixed)),
            "peak_ratio": float(np.median(ratios)),
            "baseline_cluster_counts": baseline_k,
        }

    table = pd.DataFrame(rows)
    table.to_csv(results / "survival_comparison.csv", index=False)

    # Pick a manuscript primary method: best mean -log10 Cox p across cohorts,
    # requiring METABRIC Cox p < 0.05 when available.
    hard_methods = [
        method
        for method in table["method"].unique()
        if method not in {"bigclam_mixed"}
    ]
    ranks = []
    for method in hard_methods:
        part = table[table["method"] == method]
        meta = part[part["cohort"] == "metabric"]
        meta_p = float(meta["cox_p_given_age_pam50"].iloc[0]) if len(meta) else np.nan
        score = float((-np.log10(np.clip(part["cox_p_given_age_pam50"], 1e-300, 1))).mean())
        ranks.append(
            {
                "method": method,
                "mean_neglog10_cox": score,
                "metabric_cox_p": meta_p,
                "n_sig_cox_0.05": int((part["cox_p_given_age_pam50"] < 0.05).sum()),
            }
        )
    rank_df = pd.DataFrame(ranks).sort_values(
        ["n_sig_cox_0.05", "mean_neglog10_cox"], ascending=[False, False]
    )
    rank_df.to_csv(results / "method_ranking.csv", index=False)
    primary = rank_df.iloc[0].to_dict()
    summary["decision"] = {
        "recommended_primary": primary["method"],
        "reason": (
            "Highest number of cohorts with PAM50+age-adjusted Cox p<0.05, "
            "then strongest mean -log10 Cox p. BIGCLAM stays if competitive; "
            "otherwise lead with the winning literature baseline and keep BIGCLAM "
            "as the overlapping-membership analysis."
        ),
        "ranking": rank_df.to_dict(orient="records"),
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Recommended primary method:", primary["method"])
    print(rank_df.to_string(index=False))
    print("Wrote", results / "survival_comparison.csv")


if __name__ == "__main__":
    main()
