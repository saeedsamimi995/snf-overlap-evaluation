"""Revision analyses + publication figures for the manuscript.

Produces:
  - lead decision pass/fail table
  - KM plots with risk tables + log-rank p
  - single-view / PAM50 hard ablations (METABRIC primary)
  - mixed vs single gene-program enrichment (METABRIC, BIGCLAM + NOCD-X)
  - nested C-index with bootstrap SE note
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.q1_clustering import _knn_affinity_from_distance, snf_fuse  # noqa: E402
from src.analysis.snf_bigclam import snf_hard_labels  # noqa: E402
from src.evaluation.q1_survival import GENE_SETS, cox_added_value, evaluate_partition  # noqa: E402
from src.preprocessing.harmonize import load_harmonized  # noqa: E402

OUT = Path(__file__).resolve().parent / "paper_figures" / "includegraphics"
OUT.mkdir(parents=True, exist_ok=True)
TAB = Path(__file__).resolve().parent / "tables"
TAB.mkdir(parents=True, exist_ok=True)

OVERLAP = ROOT / "results" / "snf_overlap_sota"
if not OVERLAP.exists():
    OVERLAP = ROOT / "Results" / "snf_overlap_sota"
SNF_DIR = ROOT / "results" / "snf_bigclam"
if not SNF_DIR.exists():
    SNF_DIR = ROOT / "Results" / "snf_bigclam"


def write_lead_decision_table() -> pd.DataFrame:
    rank = pd.read_csv(OVERLAP / "soft_method_ranking.csv")
    summary = json.loads((OVERLAP / "summary.json").read_text(encoding="utf-8"))
    locks = summary.get("ratio_lock", {})
    rows = []
    for _, r in rank.iterrows():
        method = r["soft_method"]
        mixed = float(r["metabric_mixed_fraction"])
        peak = float(r["metabric_peak_ratio"])
        luma = float(r["metabric_luma_OR"])
        f_mixed = 0.10 <= mixed <= 0.40
        f_peak = peak < 0.85
        f_luma = luma > 1.0
        role = r["role"]
        eligible = role == "sota_candidate"
        quality_pass = f_mixed and f_peak and f_luma
        # Lead pool: quality pass AND eligible (BIGCLAM excluded a priori)
        lead_eligible = quality_pass and eligible
        rows.append(
            {
                "method": method,
                "role": role,
                "locked_mixed_ratio": locks.get(method, np.nan),
                "metabric_mixed_fraction": mixed,
                "pass_mixed_10_40": f_mixed,
                "metabric_peak_ratio": peak,
                "pass_peak_lt_0.85": f_peak,
                "metabric_luma_OR": luma,
                "pass_luma_OR_gt_1": f_luma,
                "quality_pass": quality_pass,
                "eligible_for_lead": eligible,
                "lead_pool": lead_eligible,
                "own_hard_plus_mixed_p": float(r["metabric_own_hard_plus_mixed_p"]),
                "hard_cox_p": float(r["metabric_hard_cox_p"]),
            }
        )
    df = pd.DataFrame(rows)
    # Among lead_pool, pick best hard Cox (only NOCD-X currently)
    pool = df[df["lead_pool"]].sort_values("hard_cox_p")
    lead = pool.iloc[0]["method"] if len(pool) else None
    df["selected_lead"] = df["method"] == lead
    df.to_csv(TAB / "lead_decision_table.csv", index=False)

    # Figure
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    cols = [
        "method",
        "role",
        "pass_mixed_10_40",
        "pass_peak_lt_0.85",
        "pass_luma_OR_gt_1",
        "quality_pass",
        "eligible_for_lead",
        "lead_pool",
        "selected_lead",
    ]
    display = df[cols].copy()
    display["method"] = display["method"].str.replace("snf_", "", regex=False)
    cell = display.astype(str).values.tolist()
    headers = [
        "Method",
        "Role",
        "Mixed\n10–40%",
        "Peak\n<0.85",
        "LumA\nOR>1",
        "Quality\npass",
        "Lead\neligible",
        "In lead\npool",
        "Selected\nlead",
    ]
    table = ax.table(cellText=cell, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 1.45)
    for (i, j), cell_obj in table.get_celld().items():
        if i == 0:
            cell_obj.set_facecolor("#D9E8F5")
            cell_obj.set_text_props(weight="bold")
        elif j >= 2 and i > 0:
            val = display.iloc[i - 1, j]
            if val == "True":
                cell_obj.set_facecolor("#D9F2D9")
            elif val == "False":
                cell_obj.set_facecolor("#F8D9D9")
    ax.set_title(
        "Soft-engine lead decision (METABRIC). BIGCLAM is a priori baseline → not lead-eligible.",
        fontsize=10,
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_lead_decision.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_lead_decision.pdf", bbox_inches="tight")
    plt.close(fig)
    print("lead decision ->", lead)
    return df


def km_with_risk(clinical, labels, title, out_name, max_groups=8):
    frame = clinical.copy()
    frame["group"] = pd.Series(labels, index=frame.index).astype(str)
    frame["OS_time"] = pd.to_numeric(frame["OS_time"], errors="coerce")
    frame["OS_event"] = pd.to_numeric(frame["OS_event"], errors="coerce")
    frame = frame.dropna(subset=["OS_time", "OS_event", "group"])
    frame = frame[(frame["OS_time"] > 0) & frame["OS_event"].isin([0, 1])]
    if frame["group"].nunique() < 2:
        return
    result = multivariate_logrank_test(frame["OS_time"], frame["group"], frame["OS_event"])
    p_value = float(result.p_value)

    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_axes([0.12, 0.40, 0.82, 0.50])
    fitter = KaplanMeierFitter()
    groups = sorted(frame["group"].unique(), key=str)[:max_groups]
    for group in groups:
        subset = frame[frame["group"] == group]
        if len(subset) < 5:
            continue
        fitter.fit(subset["OS_time"], subset["OS_event"], label=f"{group} (n={len(subset)})")
        fitter.plot_survival_function(ax=ax, ci_show=True, alpha=0.9)
    ax.set_xlabel("Days")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"{title}\nlog-rank p={p_value:.2e}")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="best")

    # Risk table sits below the axis label so "Days" and the counts do not collide.
    times = np.linspace(0, frame["OS_time"].max(), 6)
    ax_risk = fig.add_axes([0.12, 0.02, 0.82, 0.28])
    ax_risk.axis("off")
    rows = []
    row_labels = []
    for group in groups:
        subset = frame[frame["group"] == group]
        counts = [int((subset["OS_time"] >= t).sum()) for t in times]
        rows.append(counts)
        row_labels.append(str(group))
    table = ax_risk.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=[str(int(t)) for t in times],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    fig.text(0.12, 0.31, "Numbers at risk", fontsize=8, ha="left", va="bottom")
    fig.savefig(OUT / f"{out_name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("KM", out_name, "p=", p_value)


def rebuild_kms():
    mapping = {
        "gse96058": ("fig_km_snf_gse", "fig_km_mixed_gse", "fig_km_within_luma_gse"),
        "tcga": ("fig_km_snf_tcga", "fig_km_mixed_tcga", "fig_km_within_luma_tcga"),
        "metabric": ("fig_km_snf_metabric", "fig_km_mixed_metabric", "fig_km_within_luma_metabric"),
    }
    for cohort, (snf_name, mixed_name, luma_name) in mapping.items():
        path = SNF_DIR / f"{cohort}_assignments.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        km_with_risk(frame, frame["snf_community"], f"{cohort} SNF", snf_name)
        km_with_risk(
            frame,
            np.where(frame["mixed"].astype(int) == 1, "mixed", "single"),
            f"{cohort}: mixed vs single (BIGCLAM soft flag)",
            mixed_name,
        )
        pam = frame["PAM50"].astype(str).str.lower().str.replace(" ", "").str.replace("-", "", regex=False)
        sub = frame.loc[pam == "luma"].copy()
        if len(sub) >= 40:
            km_with_risk(
                sub,
                np.where(sub["mixed"].astype(int) == 1, "LumA mixed", "LumA single"),
                f"{cohort}: within-LumA mixed vs single",
                luma_name,
            )


def hard_ablations():
    """Single-view spectral, SNF+k-means; PAM50 reported by unadjusted log-rank only."""
    cfg = ROOT / "results" / "q1" / "locked_config.json"
    if not cfg.exists():
        cfg = ROOT / "Results" / "q1" / "locked_config.json"
    locked = json.loads(cfg.read_text(encoding="utf-8"))
    harmonized = load_harmonized("data/harmonized")
    k = int(locked["num_communities"])
    knn_k = int(locked.get("knn_k", 15))
    seed = int(locked.get("seed", 42))
    rows = []
    for name, cohort in harmonized["cohorts"].items():
        X = cohort["X"]
        clinical = cohort["clinical"]
        euclidean = euclidean_distances(X)
        correlation = pairwise_distances(X, metric="correlation")
        w_e = _knn_affinity_from_distance(euclidean, knn_k)
        w_c = _knn_affinity_from_distance(correlation, knn_k)
        fused = snf_fuse([w_e, w_c], k=knn_k, iterations=15)
        labels = {
            "snf_spectral": snf_hard_labels(fused, k, seed=seed),
            "euclidean_spectral": SpectralClustering(
                n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=seed
            ).fit_predict(w_e),
            "correlation_spectral": SpectralClustering(
                n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=seed
            ).fit_predict(w_c),
            "snf_kmeans": KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(fused),
        }
        for method, lab in labels.items():
            rows.append(evaluate_partition(clinical, lab, method, name))
        if "PAM50" in clinical.columns and clinical["PAM50"].notna().sum() > 50:
            # Do not put PAM50 labels into an age+PAM50 Cox; report log-rank only.
            from src.evaluation.q1_survival import logrank_pvalue

            lr = logrank_pvalue(clinical, clinical["PAM50"].astype(str))
            rows.append(
                {
                    "cohort": name,
                    "method": "pam50_hard",
                    "n": lr["n"],
                    "n_events": lr["n_events"],
                    "logrank_p": lr["p"],
                    "cox_lr_stat": np.nan,
                    "cox_df": 0,
                    "cox_p_given_age_pam50": np.nan,
                    "cox_note": "reference taxonomy; adjusted Cox undefined vs itself",
                    "pam50_n": lr["n"],
                    "pam50_ari": 1.0,
                    "pam50_nmi": 1.0,
                    "intclust_n": 0,
                    "intclust_ari": np.nan,
                    "intclust_nmi": np.nan,
                }
            )
        print("ablations", name)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "hard_ablations.csv", index=False)

    meta = df[df["cohort"] == "metabric"].copy()
    order = ["snf_spectral", "snf_kmeans", "euclidean_spectral", "correlation_spectral"]
    pretty = {
        "snf_spectral": "SNF+\nspectral",
        "snf_kmeans": "SNF+\nk-means",
        "euclidean_spectral": "Euclidean\nonly",
        "correlation_spectral": "Correlation\nonly",
    }
    meta = meta.set_index("method").reindex([m for m in order if m in set(meta["method"])]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    vals = -np.log10(np.clip(meta["cox_p_given_age_pam50"].astype(float), 1e-300, 1))
    colors = ["#2E6F9E" if m == "snf_spectral" else "#888888" for m in meta["method"]]
    ax.bar(range(len(meta)), vals, color=colors)
    ax.axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1)
    ax.set_xticks(range(len(meta)))
    ax.set_xticklabels([pretty.get(m, m) for m in meta["method"]], fontsize=8)
    ax.set_ylabel(r"$-\log_{10}$ age+PAM50 Cox $p$")
    ax.set_title("Hard ablations on METABRIC (single-view vs fused SNF)")
    for i, row in meta.iterrows():
        ax.text(i, float(vals.iloc[i]) + 0.05, f"n={int(row['n'])}", ha="center", fontsize=7)
    # PAM50 log-rank footnote
    pam = df[(df["cohort"] == "metabric") & (df["method"] == "pam50_hard")]
    if len(pam):
        ax.text(
            0.02,
            0.98,
            f"PAM50 reference log-rank p={float(pam['logrank_p'].iloc[0]):.2e} (not age+PAM50 Cox)",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(OUT / "fig_hard_ablations.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_hard_ablations.pdf", bbox_inches="tight")
    plt.close(fig)
    return df


def mixed_biology():
    """Program scores + top differential genes for mixed vs single on METABRIC."""
    from scipy.stats import mannwhitneyu

    harmonized = load_harmonized("data/harmonized")
    cohort = harmonized["cohorts"]["metabric"]
    X = cohort["X"]
    genes = list(harmonized["genes"])
    assign = pd.read_csv(SNF_DIR / "metabric_assignments.csv")
    # BIGCLAM mixed from assignments; NOCD mixed from overlap file if present
    overlap_assign = OVERLAP / "metabric_assignments.csv"
    frames = [("bigclam", assign["mixed"].astype(int).to_numpy())]
    if overlap_assign.exists():
        oa = pd.read_csv(overlap_assign)
        if "snf_nocd_x_mixed" in oa.columns:
            frames.append(("nocd_x", oa["snf_nocd_x_mixed"].astype(int).to_numpy()))

    program_rows = []
    gene_rows = []
    for tag, mixed in frames:
        for name, members in GENE_SETS.items():
            idx = [genes.index(g) for g in members if g in genes]
            if len(idx) < 2:
                continue
            scores = X[:, idx].mean(axis=1)
            a = scores[mixed == 1]
            b = scores[mixed == 0]
            if len(a) < 10 or len(b) < 10:
                continue
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            program_rows.append(
                {
                    "engine": tag,
                    "program": name,
                    "n_mixed": int(mixed.sum()),
                    "mean_mixed": float(a.mean()),
                    "mean_single": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "p": float(p),
                }
            )
        # top |delta| genes
        deltas = []
        for j, g in enumerate(genes):
            a = X[mixed == 1, j]
            b = X[mixed == 0, j]
            deltas.append((g, float(a.mean() - b.mean()), float(mannwhitneyu(a, b).pvalue)))
        deltas.sort(key=lambda t: abs(t[1]), reverse=True)
        for g, d, p in deltas[:30]:
            gene_rows.append({"engine": tag, "gene": g, "delta_mixed_minus_single": d, "p": p})

    prog = pd.DataFrame(program_rows)
    prog.to_csv(TAB / "mixed_program_enrichment.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(TAB / "mixed_top_genes.csv", index=False)

    if len(prog):
        # BH within engine
        for eng, part in prog.groupby("engine"):
            pvals = part["p"].to_numpy()
            order = np.argsort(pvals)
            ranked = np.empty_like(order, dtype=float)
            n = len(pvals)
            prev = 1.0
            for rank, idx in enumerate(order[::-1], start=1):
                # reverse for BH from largest p
                pass
            # standard BH
            order = np.argsort(pvals)
            q = np.empty(n)
            prev = 1.0
            for i in range(n - 1, -1, -1):
                idx = order[i]
                rank = i + 1
                val = min(prev, pvals[idx] * n / rank)
                q[idx] = val
                prev = val
            prog.loc[part.index, "q_bh"] = q

        pivot = prog.pivot_table(index="program", columns="engine", values="delta")
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        x = np.arange(len(pivot))
        width = 0.35
        engines = list(pivot.columns)
        for i, eng in enumerate(engines):
            ax.bar(x + (i - 0.5) * width, pivot[eng], width=width, label=eng)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.set_ylabel("Mean z-score (mixed − single)")
        ax.set_title("METABRIC program scores: mixed vs single")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_mixed_programs.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / "fig_mixed_programs.pdf", bbox_inches="tight")
        plt.close(fig)
    print("biology tables written")


def annotate_cox_bars():
    path = SNF_DIR / "survival_comparison.csv"
    df = pd.read_csv(path)
    hard = df[~df["method"].astype(str).str.contains("mixed")].copy()
    methods = ["snf", "leiden", "nmf", "snf_bigclam", "consensus"]
    labels = ["SNF", "Leiden", "NMF", "BIGCLAM hard", "Consensus"]
    cohorts = ["gse96058", "tcga", "metabric"]
    cohort_labels = ["GSE96058", "TCGA", "METABRIC"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, cohort, clab in zip(axes, cohorts, cohort_labels):
        vals, ns = [], []
        for m in methods:
            row = hard[(hard["cohort"] == cohort) & (hard["method"] == m)]
            p = float(row["cox_p_given_age_pam50"].iloc[0]) if len(row) else np.nan
            vals.append(-np.log10(max(p, 1e-300)) if np.isfinite(p) else np.nan)
            ns.append(int(row["n"].iloc[0]) if len(row) else 0)
        colors = ["#2E6F9E" if m == "snf" else "#888888" for m in methods]
        ax.bar(range(len(labels)), vals, color=colors, edgecolor="white")
        ax.axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1, label="uncorrected p=0.05")
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + 0.08, f"n={n}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(clab)
        ax.set_ylabel(r"$-\log_{10}$ age+PAM50 Cox $p$" if cohort == "gse96058" else "")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Hard partitions (uncorrected p; BH discussed in text). Bars annotated with analysis n.",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_cox_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_cox_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== lead decision ===")
    write_lead_decision_table()
    print("=== KM rebuild ===")
    rebuild_kms()
    print("=== cox bars ===")
    annotate_cox_bars()
    print("=== biology ===")
    mixed_biology()
    print("=== hard ablations (may take a few minutes) ===")
    hard_ablations()
    print("Done")


if __name__ == "__main__":
    main()
