"""Bake-off: soft/overlapping engines on the same SNF fused patient network.

Compares snf_prototype, snf_nmf, snf_gmm, snf_fcm, snf_bigclam for:
  - hard-cut survival (age+PAM50 Cox)
  - residual soft signal on top of hard SNF labels
  - mixed×PAM50 enrichment (esp. LumA)

Usage:
    python run_snf_soft_bakeoff.py --harmonized-dir data/harmonized --results-dir results/snf_soft
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

from src.analysis.snf_soft import fit_snf_soft_suite  # noqa: E402
from src.evaluation.q1_survival import (  # noqa: E402
    cox_hard_plus_soft,
    evaluate_partition,
    subtype_enrichment,
)
from src.preprocessing.harmonize import load_harmonized  # noqa: E402

SOFT_METHODS = ("snf_prototype", "snf_nmf", "snf_gmm", "snf_fcm", "snf_bigclam")
FOCUS_SUBTYPES = ("LumA", "LumB", "Basal", "Her2", "Normal", "Claudin-low")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonized-dir", default="data/harmonized")
    parser.add_argument("--locked-config", default="results/q1/locked_config.json")
    parser.add_argument("--results-dir", default="results/snf_soft")
    parser.add_argument("--reuse-bigclam-dir", default="results/snf_bigclam",
                        help="Reuse saved F.npy if present to skip BIGCLAM refit")
    parser.add_argument("--fit-iterations", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--skip-bigclam", action="store_true")
    args = parser.parse_args()

    locked = json.loads((ROOT / args.locked_config).read_text(encoding="utf-8"))
    harmonized = load_harmonized(args.harmonized_dir)
    out = ROOT / args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    reuse_dir = ROOT / args.reuse_bigclam_dir

    n_communities = int(locked["num_communities"])
    knn_k = int(locked.get("knn_k", 15))
    fit_iterations = int(args.fit_iterations or locked.get("fit_iterations", 250))
    restarts = int(args.restarts or locked.get("restarts", 5))
    lr = float(locked.get("learning_rate", 0.05))
    l1 = float(locked.get("l1", 0.01))
    seed = int(locked.get("seed", 42))
    discovery = locked.get("discovery_cohort", "gse96058")

    overlap_ratios = {}
    hard_rows, residual_rows, enrich_rows, quality_rows = [], [], [], []
    summary = {"method": "SNF soft bake-off", "locked_from": locked, "cohorts": {}}

    cohort_names = [discovery] + [n for n in harmonized["cohorts"] if n != discovery]

    for name in cohort_names:
        cohort = harmonized["cohorts"][name]
        print("=" * 80)
        print(f"SNF soft suite on {name}")

        pre_F = None
        f_path = reuse_dir / f"{name}_F.npy"
        if f_path.exists() and not args.skip_bigclam:
            pre_F = np.load(f_path)
            print(f"  reusing BIGCLAM F from {f_path}")

        fit = fit_snf_soft_suite(
            cohort["X"],
            n_communities=n_communities,
            knn_k=knn_k,
            seed=seed,
            bigclam_iterations=fit_iterations,
            bigclam_restarts=restarts,
            bigclam_lr=lr,
            bigclam_l1=l1,
            overlap_ratios=overlap_ratios,
            skip_bigclam=args.skip_bigclam,
            precomputed_bigclam_F=pre_F,
        )

        if name == discovery:
            for method, soft in fit["soft"].items():
                overlap_ratios[method] = soft["overlap_ratio"]
                print(
                    f"  lock {method}: overlap_ratio={soft['overlap_ratio']:.2f} "
                    f"mixed={soft['mixed_fraction']:.3f} peak={soft['peak_ratio']:.3f}"
                )

        clinical = cohort["clinical"]
        hard_rows.append(evaluate_partition(clinical, fit["snf_community"], "snf", name))

        frame = clinical.copy()
        frame.insert(0, "sample_id", cohort["samples"])
        frame["snf_community"] = fit["snf_community"]

        for method, soft in fit["soft"].items():
            hard_rows.append(evaluate_partition(clinical, soft["community"], f"{method}_hard", name))
            if 0 < soft["mixed_fraction"] < 1:
                hard_rows.append(
                    evaluate_partition(clinical, soft["mixed"].astype(int), f"{method}_mixed", name)
                )

            residual_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    **cox_hard_plus_soft(clinical, fit["snf_community"], soft["mixed"].astype(int), "mixed"),
                    "test": "snf_hard_plus_mixed",
                }
            )
            residual_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    **cox_hard_plus_soft(clinical, fit["snf_community"], soft["entropy"], "entropy"),
                    "test": "snf_hard_plus_entropy",
                }
            )
            residual_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    **cox_hard_plus_soft(clinical, soft["community"], soft["mixed"].astype(int), "mixed"),
                    "test": "own_hard_plus_mixed",
                }
            )

            for focus in FOCUS_SUBTYPES:
                enrich = subtype_enrichment(clinical, soft["mixed"], focus=focus)
                enrich["cohort"] = name
                enrich["soft_method"] = method
                enrich_rows.append(enrich)

            quality_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    "peak_ratio": soft["peak_ratio"],
                    "mixed_fraction": soft["mixed_fraction"],
                    "overlap_ratio": soft["overlap_ratio"],
                    "mean_entropy": float(np.mean(soft["entropy"])),
                }
            )

            np.save(out / f"{name}_{method}_F.npy", soft["F"])
            frame[f"{method}_community"] = soft["community"]
            frame[f"{method}_mixed"] = soft["mixed"].astype(int)
            frame[f"{method}_entropy"] = soft["entropy"]
            frame[f"{method}_ratio"] = soft["membership_ratio"]

        frame.to_csv(out / f"{name}_assignments.csv", index=False)
        np.save(out / f"{name}_snf_labels.npy", fit["snf_community"])
        summary["cohorts"][name] = {
            method: {
                "peak_ratio": soft["peak_ratio"],
                "mixed_fraction": soft["mixed_fraction"],
                "overlap_ratio": soft["overlap_ratio"],
            }
            for method, soft in fit["soft"].items()
        }

    hard_df = pd.DataFrame(hard_rows)
    hard_df.to_csv(out / "survival_comparison.csv", index=False)
    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(out / "residual_overlap_cox.csv", index=False)
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(out / "mixed_enrichment.csv", index=False)
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(out / "soft_quality.csv", index=False)

    # Rank soft methods: METABRIC residual (own_hard+mixed) then LumA enrichment
    rank_rows = []
    for method in SOFT_METHODS:
        meta_res = residual_df[
            (residual_df["soft_method"] == method)
            & (residual_df["cohort"] == "metabric")
            & (residual_df["test"] == "own_hard_plus_mixed")
        ]
        snf_res = residual_df[
            (residual_df["soft_method"] == method)
            & (residual_df["cohort"] == "metabric")
            & (residual_df["test"] == "snf_hard_plus_mixed")
        ]
        luma = enrich_df[
            (enrich_df["soft_method"] == method)
            & (enrich_df["cohort"] == "metabric")
            & (enrich_df["subtype"] == "LumA")
        ]
        hard_meta = hard_df[
            (hard_df["method"] == f"{method}_hard") & (hard_df["cohort"] == "metabric")
        ]
        qual = quality_df[(quality_df["soft_method"] == method) & (quality_df["cohort"] == "metabric")]
        rank_rows.append(
            {
                "soft_method": method,
                "metabric_own_hard_plus_mixed_p": float(meta_res["p"].iloc[0]) if len(meta_res) else np.nan,
                "metabric_snf_hard_plus_mixed_p": float(snf_res["p"].iloc[0]) if len(snf_res) else np.nan,
                "metabric_hard_cox_p": float(hard_meta["cox_p_given_age_pam50"].iloc[0]) if len(hard_meta) else np.nan,
                "metabric_luma_OR": float(luma["odds_ratio"].iloc[0]) if len(luma) else np.nan,
                "metabric_luma_p": float(luma["p"].iloc[0]) if len(luma) else np.nan,
                "metabric_peak_ratio": float(qual["peak_ratio"].iloc[0]) if len(qual) else np.nan,
                "metabric_mixed_fraction": float(qual["mixed_fraction"].iloc[0]) if len(qual) else np.nan,
            }
        )
    rank_df = pd.DataFrame(rank_rows).sort_values("metabric_own_hard_plus_mixed_p")
    rank_df.to_csv(out / "soft_method_ranking.csv", index=False)

    summary["overlap_ratios_locked"] = overlap_ratios
    summary["recommended"] = {
        "primary_hard": "snf",
        "soft_ranking_file": "soft_method_ranking.csv",
        "note": (
            "Do not claim BIGCLAM is best a priori. Prefer the soft engine with "
            "best residual survival + coherent LumA enrichment after discovery lock."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 80)
    print(rank_df.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
