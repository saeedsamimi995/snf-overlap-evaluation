"""Run ceiling experiments on the locked SNF graphs and write the decision note.

Usage (from the project root):

    python -m scripts.run_ceiling_experiments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.ceiling_biology import run_biology
from src.analysis.ceiling_mechanistic import (
    cache_locked_graphs,
    run_mechanistic,
    slpa_collapse_reversed,
)
from src.preprocessing.cohorts import PROJECT_ROOT
from src.preprocessing.harmonize import load_harmonized


def _resolve(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _records(frame, mask, columns, limit=8):
    if frame.empty:
        return []
    part = frame.loc[mask, columns]
    return part.head(limit).to_dict(orient="records")


def write_decision_note(out_dir):
    """Choose an identity only when a pre-stated bar is cleared."""
    out = Path(out_dir)
    mech = json.loads((out / "mechanistic_summary.json").read_text(encoding="utf-8"))
    bio = json.loads((out / "biology_summary.json").read_text(encoding="utf-8"))
    density = pd.read_csv(out / "density_sweep.csv")
    # Recompute the SLPA bar from the table so the note matches the file.
    reversed_flag, detail = slpa_collapse_reversed(density)
    mech["slpa_collapse_reversed"] = bool(reversed_flag)
    mech["slpa_detail"] = detail

    enrich = pd.read_csv(out / "orthogonal_enrichment.csv")
    programs = pd.read_csv(out / "program_contrasts.csv")
    survival = pd.read_csv(out / "boundary_survival.csv")

    def hits(frame, cohort):
        if "q" not in frame.columns:
            return frame.iloc[0:0]
        part = frame[(frame["cohort"] == cohort) & frame["q"].notna() & (frame["q"] < 0.05)]
        return part

    survival_hits = hits(survival, "metabric")
    # LumA odds ratios are already in the main paper; they do not open a new identity.
    enrich_hits = hits(enrich, "metabric")
    enrich_hits = enrich_hits[enrich_hits["level"].astype(str) != "LumA"]
    program_hits = hits(programs[programs["kind"] == "program"], "metabric") if not programs.empty else programs
    scanb_hits = hits(programs[programs["kind"] == "program"], "gse96058") if not programs.empty else programs

    biological = (not survival_hits.empty) or (not enrich_hits.empty) or (not program_hits.empty)
    mechanistic = bool(reversed_flag)
    if biological and mechanistic:
        identity = "both"
        primary = "biological refinement" if not survival_hits.empty else "mechanistic boundary"
    elif biological:
        identity = "biological refinement"
        primary = identity
    elif mechanistic:
        identity = "mechanistic boundary"
        primary = identity
    else:
        identity = "protocol paper"
        primary = identity

    loss = pd.read_csv(out / "loss_ablation.csv")
    synthetic = pd.read_csv(out / "synthetic_recovery.csv")
    views = pd.read_csv(out / "view_ablation.csv")
    synth_mean = (
        synthetic.groupby("engine")[["F_correlation", "mixed_ari"]].mean().reset_index().to_dict(orient="records")
    )
    view_luma = views.groupby(["view", "engine"])["luma_or"].mean().reset_index().to_dict(orient="records")

    note = {
        "identity": identity,
        "primary": primary,
        "bars": {
            "slpa_density_reversal": mechanistic,
            "metabric_survival_q_lt_0.05": not survival_hits.empty,
            "metabric_orthogonal_or_program_q_lt_0.05": (not enrich_hits.empty) or (not program_hits.empty),
        },
        "slpa_detail": detail,
        "treatment": bio.get("treatment"),
        "survival_hits": survival_hits.to_dict(orient="records"),
        "enrichment_hits_excluding_luma": enrich_hits.to_dict(orient="records"),
        "program_hits": program_hits.to_dict(orient="records"),
        "scanb_program_hits": scanb_hits.to_dict(orient="records"),
        "synthetic_mean": synth_mean,
        "view_luma_or": view_luma,
        "loss_rows": loss.to_dict(orient="records"),
    }
    (out / "decision_summary.json").write_text(json.dumps(note, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Ceiling decision",
        "",
        f"Primary identity: **{primary}**.",
        "",
        "Pre-stated bars:",
        "",
        f"- SLPA density reversal (peak ratio falls below 0.85 on a sparser graph and the mixed fraction moves off the collapsed value): {mechanistic}. {detail}.",
        f"- METABRIC within-cluster or within-LumA age-adjusted survival at BH q < 0.05: {not survival_hits.empty}.",
        f"- METABRIC IntClust, Claudin-low, or within-cluster programme contrast at BH q < 0.05 (LumA odds ratios excluded; they are already reported): {(not enrich_hits.empty) or (not program_hits.empty)}.",
        "",
        "Treatment: the harmonized clinical tables do not contain hormone therapy or chemotherapy columns, so no treatment model was fit.",
        "",
        "Synthetic recovery (mean over replicates) and single-view LumA odds ratios are in `synthetic_recovery.csv` and `view_ablation.csv`. They inform the boundary section; they do not by themselves change the lead-engine policy.",
        "",
        "If the identity is `protocol paper`, the manuscript keeps the locked dual-output claim and reports these experiments as boundary checks that did not clear the bar.",
        "",
    ]
    (out / "decision_note.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"identity": identity, "primary": primary, "slpa": detail}, indent=2))
    return note


def main():
    parser = argparse.ArgumentParser(description="Ceiling experiments on locked SNF graphs")
    parser.add_argument("--harmonized-dir", default="data/harmonized")
    parser.add_argument("--results-dir", default="results/ceiling")
    parser.add_argument("--locked-dir", default="Results/snf_overlap_sota")
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "cache", "mechanistic", "biology", "gate"),
    )
    args = parser.parse_args()
    out = _resolve(args.results_dir)
    graphs = out / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    harmonized = load_harmonized(args.harmonized_dir)

    if args.stage in {"all", "cache"}:
        print("Caching locked graphs")
        cache_locked_graphs(harmonized, graphs)
    if args.stage in {"all", "mechanistic"}:
        print("Mechanistic track (METABRIC)")
        run_mechanistic(harmonized, graphs, out, cohort="metabric")
    if args.stage in {"all", "biology"}:
        print("Biology track")
        run_biology(harmonized, graphs, _resolve(args.locked_dir), out, harmonized["genes"])
    if args.stage in {"all", "gate"}:
        print("Decision gate")
        write_decision_note(out)


if __name__ == "__main__":
    main()
