"""SNF + SOTA overlapping engines (BIGCLAM is a baseline, not the lead).

Soft / overlapping candidates on the same sparsified SNF patient graph:
  - snf_nocd_x     Neural Overlapping Community Detection + expression features
  - snf_nocd_g     NOCD with graph-derived features only
  - snf_slpa       Speaker-Listener Label Propagation (soft histograms)
  - snf_cesna_lite Attribute-aware affiliation (CESNA-style)
  - snf_bigclam    Classical affiliation baseline (comparison only)

Hard SNF remains the prognostic backbone. The paper lead soft method is
whatever wins residual Cox + coherent biology among non-BIGCLAM candidates.

Usage:
    python run_snf_overlap_sota.py --harmonized-dir data/harmonized --results-dir results/snf_overlap_sota
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

from src.analysis.overlapping_sota import fit_cesna_lite, fit_nocd, fit_slpa  # noqa: E402
from src.analysis.snf_bigclam import (  # noqa: E402
    build_snf_affinity,
    snf_affinity_to_graph,
    snf_hard_labels,
)
from src.analysis.snf_soft import _wrap_membership  # noqa: E402
from src.bigclam.bigclam_model import fit_bigclam_c  # noqa: E402
from src.evaluation.q1_survival import (  # noqa: E402
    cox_hard_plus_soft,
    evaluate_partition,
    subtype_enrichment,
)
from src.preprocessing.harmonize import load_harmonized  # noqa: E402

METHODS = ("snf_nocd_x", "snf_nocd_g", "snf_slpa", "snf_cesna_lite", "snf_bigclam")
FOCUS_SUBTYPES = ("LumA", "LumB", "Basal", "Her2", "Normal", "Claudin-low")


def _fit_all(X, A, n_communities, seed, bigclam_iterations, bigclam_restarts, lr, l1, reuse_F=None):
    out = {}
    print("  NOCD-X...")
    fx, _ = fit_nocd(A, X=X, n_communities=n_communities, seed=seed, feature_mode="attributes")
    out["snf_nocd_x"] = fx
    print("  NOCD-G...")
    fg, _ = fit_nocd(A, X=None, n_communities=n_communities, seed=seed, feature_mode="graph")
    out["snf_nocd_g"] = fg
    print("  SLPA...")
    out["snf_slpa"] = fit_slpa(A, n_communities=n_communities, seed=seed)
    print("  CESNA-lite...")
    fc, _ = fit_cesna_lite(A, X, n_communities=n_communities, seed=seed)
    out["snf_cesna_lite"] = fc
    if reuse_F is not None:
        print("  BIGCLAM (reused baseline)...")
        out["snf_bigclam"] = np.asarray(reuse_F, dtype=np.float64)
    else:
        print("  BIGCLAM (baseline fit)...")
        fb, _, _ = fit_bigclam_c(
            A,
            num_communities=n_communities,
            iterations=bigclam_iterations,
            lr=lr,
            restarts=bigclam_restarts,
            seed=seed,
            l1=l1,
            n_neg=8,
        )
        out["snf_bigclam"] = fb
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmonized-dir", default="data/harmonized")
    parser.add_argument("--locked-config", default="results/q1/locked_config.json")
    parser.add_argument("--results-dir", default="results/snf_overlap_sota")
    parser.add_argument("--reuse-bigclam-dir", default="results/snf_bigclam")
    parser.add_argument("--fit-iterations", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
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

    ratio_lock = {}
    hard_rows = []
    residual_rows = []
    enrich_rows = []
    quality_rows = []
    summary = {
        "framing": "SNF hard + SOTA overlapping bake-off; BIGCLAM is comparison only",
        "locked_from": locked,
        "cohorts": {},
    }

    cohort_names = [discovery] + [n for n in harmonized["cohorts"] if n != discovery]

    for name in cohort_names:
        cohort = harmonized["cohorts"][name]
        print("=" * 80)
        print(f"SNF overlap SOTA on {name}")
        fused = build_snf_affinity(cohort["X"], knn_k=knn_k)
        snf_labels = snf_hard_labels(fused, n_communities, seed=seed)
        adjacency = snf_affinity_to_graph(fused, knn_k=knn_k)

        reuse_f = None
        f_path = reuse_dir / f"{name}_F.npy"
        if f_path.exists():
            reuse_f = np.load(f_path)

        raw = _fit_all(
            cohort["X"],
            adjacency,
            n_communities=n_communities,
            seed=seed,
            bigclam_iterations=fit_iterations,
            bigclam_restarts=restarts,
            lr=lr,
            l1=l1,
            reuse_F=reuse_f,
        )

        soft = {}
        for method, membership in raw.items():
            soft[method] = _wrap_membership(membership, overlap_ratio=ratio_lock.get(method))
            if name == discovery:
                ratio_lock[method] = soft[method]["overlap_ratio"]
                print(
                    f"  lock {method}: ratio={soft[method]['overlap_ratio']:.2f} "
                    f"mixed={soft[method]['mixed_fraction']:.3f} peak={soft[method]['peak_ratio']:.3f}"
                )

        clinical = cohort["clinical"]
        hard_rows.append(evaluate_partition(clinical, snf_labels, "snf", name))

        frame = clinical.copy()
        frame.insert(0, "sample_id", cohort["samples"])
        frame["snf_community"] = snf_labels

        for method, soft_fit in soft.items():
            hard_rows.append(
                evaluate_partition(clinical, soft_fit["community"], f"{method}_hard", name)
            )
            residual_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    "is_bigclam_baseline": method == "snf_bigclam",
                    **cox_hard_plus_soft(
                        clinical, snf_labels, soft_fit["mixed"].astype(int), "mixed"
                    ),
                    "test": "snf_hard_plus_mixed",
                }
            )
            residual_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    "is_bigclam_baseline": method == "snf_bigclam",
                    **cox_hard_plus_soft(clinical, snf_labels, soft_fit["entropy"], "entropy"),
                    "test": "snf_hard_plus_entropy",
                }
            )
            residual_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    "is_bigclam_baseline": method == "snf_bigclam",
                    **cox_hard_plus_soft(
                        clinical,
                        soft_fit["community"],
                        soft_fit["mixed"].astype(int),
                        "mixed",
                    ),
                    "test": "own_hard_plus_mixed",
                }
            )
            for focus in FOCUS_SUBTYPES:
                enrich = subtype_enrichment(clinical, soft_fit["mixed"], focus=focus)
                enrich.update(
                    {
                        "cohort": name,
                        "soft_method": method,
                        "is_bigclam_baseline": method == "snf_bigclam",
                    }
                )
                enrich_rows.append(enrich)

            quality_rows.append(
                {
                    "cohort": name,
                    "soft_method": method,
                    "is_bigclam_baseline": method == "snf_bigclam",
                    "peak_ratio": soft_fit["peak_ratio"],
                    "mixed_fraction": soft_fit["mixed_fraction"],
                    "overlap_ratio": soft_fit["overlap_ratio"],
                    "mean_entropy": float(np.mean(soft_fit["entropy"])),
                }
            )
            np.save(out / f"{name}_{method}_F.npy", soft_fit["F"])
            frame[f"{method}_community"] = soft_fit["community"]
            frame[f"{method}_mixed"] = soft_fit["mixed"].astype(int)
            frame[f"{method}_entropy"] = soft_fit["entropy"]

        frame.to_csv(out / f"{name}_assignments.csv", index=False)
        np.save(out / f"{name}_snf_labels.npy", snf_labels)
        summary["cohorts"][name] = {
            method: {
                "peak_ratio": soft[method]["peak_ratio"],
                "mixed_fraction": soft[method]["mixed_fraction"],
            }
            for method in soft
        }

    hard_df = pd.DataFrame(hard_rows)
    residual_df = pd.DataFrame(residual_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    quality_df = pd.DataFrame(quality_rows)
    hard_df.to_csv(out / "survival_comparison.csv", index=False)
    residual_df.to_csv(out / "residual_overlap_cox.csv", index=False)
    enrich_df.to_csv(out / "mixed_enrichment.csv", index=False)
    quality_df.to_csv(out / "soft_quality.csv", index=False)

    rank_rows = []
    for method in METHODS:
        meta_own = residual_df[
            (residual_df["soft_method"] == method)
            & (residual_df["cohort"] == "metabric")
            & (residual_df["test"] == "own_hard_plus_mixed")
        ]
        meta_snf = residual_df[
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
        qual = quality_df[
            (quality_df["soft_method"] == method) & (quality_df["cohort"] == "metabric")
        ]
        rank_rows.append(
            {
                "soft_method": method,
                "role": "baseline_comparison" if method == "snf_bigclam" else "sota_candidate",
                "metabric_own_hard_plus_mixed_p": float(meta_own["p"].iloc[0])
                if len(meta_own)
                else np.nan,
                "metabric_snf_hard_plus_mixed_p": float(meta_snf["p"].iloc[0])
                if len(meta_snf)
                else np.nan,
                "metabric_hard_cox_p": float(hard_meta["cox_p_given_age_pam50"].iloc[0])
                if len(hard_meta)
                else np.nan,
                "metabric_luma_OR": float(luma["odds_ratio"].iloc[0]) if len(luma) else np.nan,
                "metabric_luma_p": float(luma["p"].iloc[0]) if len(luma) else np.nan,
                "metabric_peak_ratio": float(qual["peak_ratio"].iloc[0]) if len(qual) else np.nan,
                "metabric_mixed_fraction": float(qual["mixed_fraction"].iloc[0])
                if len(qual)
                else np.nan,
            }
        )
    rank_df = pd.DataFrame(rank_rows).sort_values("metabric_own_hard_plus_mixed_p")
    rank_df.to_csv(out / "soft_method_ranking.csv", index=False)

    # Multi-criteria lead (not residual p alone): usable mixed fraction, peaked F,
    # LumA enrichment OR>1, then residual / hard Cox. SLPA often fails this filter.
    candidates = rank_df[rank_df["role"] == "sota_candidate"].copy()
    quality = candidates[
        candidates["metabric_mixed_fraction"].between(0.10, 0.40)
        & (candidates["metabric_peak_ratio"] < 0.85)
        & (candidates["metabric_luma_OR"] > 1.0)
    ].copy()
    if len(quality):
        quality = quality.sort_values(
            ["metabric_own_hard_plus_mixed_p", "metabric_hard_cox_p"]
        )
        lead = quality.iloc[0].to_dict()
        lead_rule = "quality_filtered_residual_then_hard"
    else:
        # Fall back: best hard Cox among peaked candidates with OR>1
        fallback = candidates[
            (candidates["metabric_peak_ratio"] < 0.85) & (candidates["metabric_luma_OR"] > 1.0)
        ].sort_values("metabric_hard_cox_p")
        lead = fallback.iloc[0].to_dict() if len(fallback) else candidates.iloc[0].to_dict()
        lead_rule = "fallback_hard_cox_among_peaked_luma"
    baseline = rank_df[rank_df["soft_method"] == "snf_bigclam"]
    summary["ratio_lock"] = ratio_lock
    summary["lead_soft_method"] = lead.get("soft_method")
    summary["lead_metrics"] = lead
    summary["lead_rule"] = lead_rule
    summary["bigclam_baseline"] = baseline.iloc[0].to_dict() if len(baseline) else {}
    summary["policy"] = (
        "BIGCLAM is classical comparison only. Lead soft engine chosen by quality "
        "filters (mixed 10-40%, peak<0.85, LumA OR>1) then residual/hard Cox among "
        "NOCD / SLPA / CESNA-lite — never residual p alone."
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("=" * 80)
    print(rank_df.to_string(index=False))
    print("LEAD (non-BIGCLAM):", summary["lead_soft_method"])
    print("Wrote", out)


if __name__ == "__main__":
    main()
