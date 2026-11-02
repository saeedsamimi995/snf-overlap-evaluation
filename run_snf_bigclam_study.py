"""SNF-BIGCLAM hybrid study (manuscript primary method).

Uses the locked gene list and community count from results/q1/locked_config.json.
For each cohort:
  1. Build SNF fused affinity (Euclidean + correlation)
  2. Cut hard SNF subtypes
  3. Run BIGCLAM on the sparsified SNF graph (overlapping affiliation)
  4. Compare SNF, SNF-BIGCLAM, NMF, Leiden, consensus
  5. Test whether soft overlap adds residual survival beyond hard labels
  6. Test Claudin-low enrichment among mixed patients (METABRIC)

Usage:
    python run_snf_bigclam_study.py --harmonized-dir data/harmonized --results-dir results/q1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.q1_clustering import baseline_labels  # noqa: E402
from src.analysis.snf_bigclam import fit_snf_bigclam  # noqa: E402
from src.evaluation.q1_survival import (  # noqa: E402
    cox_hard_plus_soft,
    evaluate_partition,
    plot_kaplan_meier,
    subtype_enrichment,
)
from src.preprocessing.harmonize import load_harmonized  # noqa: E402


PRIMARY_BASELINES = ("nmf", "leiden", "consensus")  # drop redundant kmeans/spectral from lead table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonized-dir", default="data/harmonized")
    parser.add_argument("--locked-config", default="results/q1/locked_config.json")
    parser.add_argument("--results-dir", default="results/snf_bigclam")
    parser.add_argument("--fit-iterations", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    args = parser.parse_args()

    locked = json.loads((ROOT / args.locked_config).read_text(encoding="utf-8"))
    harmonized = load_harmonized(args.harmonized_dir)
    out = ROOT / args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    n_communities = int(locked["num_communities"])
    knn_k = int(locked.get("knn_k", 15))
    fit_iterations = int(args.fit_iterations or locked.get("fit_iterations", 250))
    restarts = int(args.restarts or locked.get("restarts", 5))
    lr = float(locked.get("learning_rate", 0.05))
    l1 = float(locked.get("l1", 0.01))
    seed = int(locked.get("seed", 42))

    overlap_ratio = None  # lock on discovery first
    discovery = locked.get("discovery_cohort", "gse96058")
    rows = []
    soft_rows = []
    enrich_rows = []
    summary = {"locked_from": locked, "method": "SNF-BIGCLAM", "cohorts": {}}

    # Ensure discovery first so overlap ratio can be locked.
    cohort_names = [discovery] + [n for n in harmonized["cohorts"] if n != discovery]

    for name in cohort_names:
        cohort = harmonized["cohorts"][name]
        print("=" * 80)
        print(f"SNF-BIGCLAM on {name}")
        fit = fit_snf_bigclam(
            cohort["X"],
            n_communities=n_communities,
            knn_k=knn_k,
            iterations=fit_iterations,
            restarts=restarts,
            lr=lr,
            seed=seed,
            l1=l1,
            overlap_ratio=overlap_ratio,
        )
        if name == discovery:
            overlap_ratio = fit["overlap_ratio"]
            print(f"  Locked overlap_ratio={overlap_ratio:.2f} (mixed={fit['mixed_fraction']:.3f})")

        clinical = cohort["clinical"]
        # Primary hard partitions
        rows.append(evaluate_partition(clinical, fit["snf_community"], "snf", name))
        rows.append(evaluate_partition(clinical, fit["community"], "snf_bigclam", name))
        if 0 < fit["mixed_fraction"] < 1:
            rows.append(evaluate_partition(clinical, fit["mixed"].astype(int), "snf_bigclam_mixed", name))

        # Compact literature baselines on the SNF graph adjacency
        print("  baselines (NMF / Leiden / consensus)...")
        baselines, counts = baseline_labels(
            cohort["X"], fit["A"], n_communities, seed=seed, knn_k=knn_k
        )
        for method in PRIMARY_BASELINES:
            row = evaluate_partition(clinical, baselines[method], method, name)
            row["n_clusters_found"] = counts[method]
            rows.append(row)

        # Residual overlap tests on top of SNF hard labels
        soft_rows.append(
            {
                "cohort": name,
                **cox_hard_plus_soft(clinical, fit["snf_community"], fit["mixed"].astype(int), "mixed"),
                "test": "snf_hard_plus_mixed",
            }
        )
        soft_rows.append(
            {
                "cohort": name,
                **cox_hard_plus_soft(clinical, fit["snf_community"], fit["entropy"], "entropy"),
                "test": "snf_hard_plus_entropy",
            }
        )
        soft_rows.append(
            {
                "cohort": name,
                **cox_hard_plus_soft(clinical, fit["community"], fit["mixed"].astype(int), "mixed"),
                "test": "snfbigclam_hard_plus_mixed",
            }
        )

        enrich = subtype_enrichment(clinical, fit["mixed"], focus="Claudin-low")
        enrich["cohort"] = name
        enrich_rows.append(enrich)
        print(
            f"  peak={fit['peak_ratio']:.3f} mixed={fit['mixed_fraction']:.3f} "
            f"Claudin-low OR={enrich.get('odds_ratio')} p={enrich.get('p')}"
        )

        # Save artifacts
        np.save(out / f"{name}_F.npy", fit["F"])
        np.save(out / f"{name}_snf_labels.npy", fit["snf_community"])
        np.save(out / f"{name}_snf_bigclam_labels.npy", fit["community"])
        frame = clinical.copy()
        frame.insert(0, "sample_id", cohort["samples"])
        frame["snf_community"] = fit["snf_community"]
        frame["snf_bigclam_community"] = fit["community"]
        frame["mixed"] = fit["mixed"].astype(int)
        frame["membership_ratio"] = fit["membership_ratio"]
        frame["entropy"] = fit["entropy"]
        frame.to_csv(out / f"{name}_assignments.csv", index=False)
        plot_kaplan_meier(
            clinical, fit["snf_community"], out / f"{name}_km_snf.png", f"{name} SNF"
        )
        plot_kaplan_meier(
            clinical,
            fit["community"],
            out / f"{name}_km_snf_bigclam.png",
            f"{name} SNF-BIGCLAM",
        )
        plot_kaplan_meier(
            clinical,
            np.where(fit["mixed"], "mixed", "single"),
            out / f"{name}_km_mixed.png",
            f"{name} SNF-BIGCLAM mixed",
        )
        summary["cohorts"][name] = {
            "peak_ratio": fit["peak_ratio"],
            "mixed_fraction": fit["mixed_fraction"],
            "overlap_ratio": fit["overlap_ratio"],
            "log_likelihood": fit["log_likelihood"],
            "claudin_low_odds_ratio": enrich.get("odds_ratio"),
            "claudin_low_p": enrich.get("p"),
        }

    table = pd.DataFrame(rows)
    table.to_csv(out / "survival_comparison.csv", index=False)
    pd.DataFrame(soft_rows).to_csv(out / "residual_overlap_cox.csv", index=False)
    pd.DataFrame(enrich_rows).to_csv(out / "mixed_enrichment.csv", index=False)

    # Rank hard methods by METABRIC-adjusted Cox then mean strength
    hard = table[~table["method"].astype(str).str.contains("mixed")].copy()
    ranks = []
    for method, part in hard.groupby("method"):
        meta = part[part["cohort"] == "metabric"]
        meta_p = float(meta["cox_p_given_age_pam50"].iloc[0]) if len(meta) else np.nan
        ranks.append(
            {
                "method": method,
                "metabric_cox_p": meta_p,
                "mean_neglog10_cox": float(
                    (-np.log10(np.clip(part["cox_p_given_age_pam50"], 1e-300, 1))).mean()
                ),
                "n_sig_cox_0.05": int((part["cox_p_given_age_pam50"] < 0.05).sum()),
            }
        )
    rank_df = pd.DataFrame(ranks).sort_values(
        ["metabric_cox_p", "mean_neglog10_cox"], ascending=[True, False]
    )
    rank_df.to_csv(out / "method_ranking.csv", index=False)

    summary["overlap_ratio_locked"] = overlap_ratio
    summary["recommended_story"] = {
        "primary_hard_partition": "snf",
        "overlapping_module": "snf_bigclam",
        "novelty": (
            "Overlapping community affiliation (BIGCLAM) inferred on SNF-fused "
            "patient similarity graphs, locked on SCAN-B and validated on TCGA "
            "and METABRIC, with residual soft-membership tests beyond hard labels."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 80)
    print(rank_df.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
