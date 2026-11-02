"""Assemble manuscript figures into paper_figures/includegraphics/."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "paper_figures" / "includegraphics"
OUT.mkdir(parents=True, exist_ok=True)

SNF = ROOT / "results" / "snf_bigclam"
OVERLAP = ROOT / "results" / "snf_overlap_sota"
SOTA = SNF / "sota"


def _copy(src: Path, name: str) -> None:
    if not src.exists():
        # Windows may store Results vs results
        alt = Path(str(src).replace("\\results\\", "\\Results\\").replace("/results/", "/Results/"))
        if alt.exists():
            src = alt
        else:
            print(f"  missing {src}")
            return
    dest = OUT / f"{name}.png"
    shutil.copy2(src, dest)
    print(f"  {dest.name} <- {src.name}")


def write_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, fc="#F7F7F7"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=fc,
                edgecolor="#222222",
                linewidth=1.1,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, linespacing=1.25)

    boxes = [
        (0.2, 1.1, 2.0, 1.6, "Input\nGSE96058 / TCGA /\nMETABRIC expression\n+ clinical", "#EEF5FF"),
        (2.5, 1.1, 2.1, 1.6, "1. Harmonize\nIntersect genes\nLock top variance\nWithin-cohort z-score", "#F7F7F7"),
        (4.9, 1.1, 2.2, 1.6, "2. Multi-view SNF\nEuclidean + correlation\nFused patient affinity", "#FFF6E8"),
        (7.4, 1.1, 2.3, 1.6, "3. Dual output\nHard: spectral SNF\nSoft: NOCD / SLPA /\nCESNA / BIGCLAM", "#EAF8EF"),
        (10.0, 1.1, 2.2, 1.6, "4. Evaluate\nAge+PAM50 Cox\nResidual soft tests\nCross-cohort lock", "#F7F7F7"),
    ]
    for b in boxes:
        box(*b)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x1 - 0.05, 1.9),
            xytext=(x0 + 0.05, 1.9),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        )
    ax.set_title("Dual-output patient-network subtyping workflow", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_pipeline.png/pdf")


def write_cox_bars() -> None:
    path = SNF / "survival_comparison.csv"
    if not path.exists():
        path = ROOT / "Results" / "snf_bigclam" / "survival_comparison.csv"
    df = pd.read_csv(path)
    hard = df[~df["method"].astype(str).str.contains("mixed")].copy()
    methods = ["snf", "leiden", "nmf", "snf_bigclam", "consensus"]
    labels = ["SNF", "Leiden", "NMF", "BIGCLAM hard", "Consensus"]
    cohorts = ["gse96058", "tcga", "metabric"]
    cohort_labels = ["GSE96058", "TCGA", "METABRIC"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, cohort, clab in zip(axes, cohorts, cohort_labels):
        vals = []
        for m in methods:
            row = hard[(hard["cohort"] == cohort) & (hard["method"] == m)]
            p = float(row["cox_p_given_age_pam50"].iloc[0]) if len(row) else np.nan
            vals.append(-np.log10(max(p, 1e-300)) if np.isfinite(p) else np.nan)
        colors = ["#2E6F9E" if m == "snf" else "#888888" for m in methods]
        ax.bar(range(len(labels)), vals, color=colors, edgecolor="white")
        ax.axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1, label="p=0.05")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(clab)
        ax.set_ylabel(r"$-\log_{10}$ age+PAM50 Cox $p$" if cohort == "gse96058" else "")
        ax.set_ylim(0, max(4.5, np.nanmax(vals) * 1.1))
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Hard partitions: PAM50- and age-adjusted Cox strength", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cox_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_cox_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_cox_bars.png/pdf")


def write_soft_ranking() -> None:
    path = OVERLAP / "soft_method_ranking.csv"
    if not path.exists():
        path = ROOT / "Results" / "snf_overlap_sota" / "soft_method_ranking.csv"
    df = pd.read_csv(path)
    order = ["snf_nocd_x", "snf_cesna_lite", "snf_nocd_g", "snf_slpa", "snf_bigclam"]
    pretty = {
        "snf_nocd_x": "NOCD-X",
        "snf_nocd_g": "NOCD-G",
        "snf_slpa": "SLPA",
        "snf_cesna_lite": "CESNA-lite",
        "snf_bigclam": "BIGCLAM\n(baseline)",
    }
    df = df.set_index("soft_method").loc[[m for m in order if m in df["soft_method"].values]].reset_index()
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))

    axes[0].bar(x, -np.log10(np.clip(df["metabric_hard_cox_p"], 1e-300, 1)), color="#2E6F9E")
    axes[0].axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1)
    axes[0].set_title("Hard cut Cox (METABRIC)")
    axes[0].set_ylabel(r"$-\log_{10} p$")

    axes[1].bar(x, -np.log10(np.clip(df["metabric_own_hard_plus_mixed_p"], 1e-300, 1)), color="#4C8C4A")
    axes[1].axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1)
    axes[1].set_title("Own hard + mixed residual")

    axes[2].bar(x, df["metabric_luma_OR"], color="#C47A2C")
    axes[2].axhline(1.0, color="#333333", ls="--", lw=1)
    axes[2].set_title("Mixed × LumA odds ratio")
    axes[2].set_ylabel("Odds ratio")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([pretty[m] for m in df["soft_method"]], fontsize=8)

    fig.suptitle("Overlapping engines on the SNF graph (METABRIC)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_soft_bakeoff.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_soft_bakeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_soft_bakeoff.png/pdf")


def write_enrichment() -> None:
    path = OVERLAP / "mixed_enrichment.csv"
    if not path.exists():
        path = ROOT / "Results" / "snf_overlap_sota" / "mixed_enrichment.csv"
    df = pd.read_csv(path)
    focus = df[(df["cohort"] == "metabric") & (df["soft_method"].isin(["snf_nocd_x", "snf_bigclam"]))]
    focus = focus[focus["subtype"].isin(["LumA", "LumB", "Basal", "Her2", "Normal", "Claudin-low"])]
    subtypes = ["LumA", "LumB", "Basal", "Her2", "Normal", "Claudin-low"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.35
    x = np.arange(len(subtypes))
    for i, method, color, label in [
        (0, "snf_nocd_x", "#2E6F9E", "NOCD-X"),
        (1, "snf_bigclam", "#888888", "BIGCLAM (baseline)"),
    ]:
        part = focus[focus["soft_method"] == method].set_index("subtype")
        ors = [float(part.loc[s, "odds_ratio"]) if s in part.index else np.nan for s in subtypes]
        ax.bar(x + (i - 0.5) * width, ors, width=width, color=color, label=label)
    ax.axhline(1.0, color="#333", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(subtypes)
    ax.set_ylabel("Odds ratio (mixed vs not)")
    ax.set_title("METABRIC: mixed-membership PAM50 enrichment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mixed_enrichment.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_mixed_enrichment.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_mixed_enrichment.png/pdf")


def main() -> None:
    print("Exporting manuscript figures ->", OUT)
    write_pipeline()
    write_cox_bars()
    write_soft_ranking()
    write_enrichment()

    # KM panels with risk tables are produced by revise_figures_and_tables.py.
    # Do not overwrite them with the older snf_bigclam KM screenshots.
    optional = [
        (SNF / "metabric_km_snf_bigclam.png", "fig_km_bigclam_metabric"),
    ]
    for src, name in optional:
        if not (OUT / f"{name}.png").exists():
            _copy(src, name)

    used = sorted(p.name for p in OUT.glob("fig_*"))
    print("Wrote", len(used), "figure files")


if __name__ == "__main__":
    main()
