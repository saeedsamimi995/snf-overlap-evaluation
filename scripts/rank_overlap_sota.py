"""Recompute quality-filtered lead from existing snf_overlap_sota tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "snf_overlap_sota"
rank = pd.read_csv(out / "soft_method_ranking.csv")
candidates = rank[rank["role"] == "sota_candidate"].copy()
quality = candidates[
    candidates["metabric_mixed_fraction"].between(0.10, 0.40)
    & (candidates["metabric_peak_ratio"] < 0.85)
    & (candidates["metabric_luma_OR"] > 1.0)
].sort_values(["metabric_own_hard_plus_mixed_p", "metabric_hard_cox_p"])
if len(quality):
    lead = quality.iloc[0]
    rule = "quality_filtered_residual_then_hard"
else:
    fallback = candidates[
        (candidates["metabric_peak_ratio"] < 0.85) & (candidates["metabric_luma_OR"] > 1.0)
    ].sort_values("metabric_hard_cox_p")
    lead = fallback.iloc[0]
    rule = "fallback_hard_cox_among_peaked_luma"

summary_path = out / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["lead_soft_method"] = lead["soft_method"]
summary["lead_metrics"] = lead.to_dict()
summary["lead_rule"] = rule
summary["policy"] = (
    "BIGCLAM is classical comparison only. Lead soft engine chosen by quality "
    "filters (mixed 10-40%, peak<0.85, LumA OR>1) then residual/hard Cox."
)
summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
print(rank.to_string(index=False))
print("LEAD:", lead["soft_method"], "rule:", rule)
