"""Bootstrap ΔC, BH q-values, soft cross-cohort summary, figure polish."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "paper_figures" / "includegraphics"
TAB = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SNF_DIR = ROOT / "results" / "snf_bigclam"
if not SNF_DIR.exists():
    SNF_DIR = ROOT / "Results" / "snf_bigclam"
OVERLAP = ROOT / "results" / "snf_overlap_sota"
if not OVERLAP.exists():
    OVERLAP = ROOT / "Results" / "snf_overlap_sota"


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = min(prev, pvals[idx] * n / rank)
        q[idx] = val
        prev = val
    return q


def write_bh_tables():
    hard = pd.read_csv(SNF_DIR / "survival_comparison.csv")
    hard = hard[~hard["method"].astype(str).str.contains("mixed")].copy()
    rows = []
    for cohort, part in hard.groupby("cohort"):
        part = part.copy()
        part["q_bh_within_cohort"] = bh_adjust(part["cox_p_given_age_pam50"].astype(float).to_numpy())
        rows.append(part)
    hard_bh = pd.concat(rows, ignore_index=True)
    # also family-wise on METABRIC only (primary hard claim)
    meta = hard_bh[hard_bh["cohort"] == "metabric"].copy()
    meta["q_bh_metabric_family"] = bh_adjust(meta["cox_p_given_age_pam50"].astype(float).to_numpy())
    hard_bh = hard_bh.merge(
        meta[["method", "q_bh_metabric_family"]], on="method", how="left"
    )
    hard_bh.to_csv(TAB / "hard_cox_bh.csv", index=False)

    soft = pd.read_csv(OVERLAP / "residual_overlap_cox.csv")
    soft = soft[soft["test"] == "own_hard_plus_mixed"].copy()
    soft_rows = []
    for cohort, part in soft.groupby("cohort"):
        part = part.copy()
        part["q_bh_within_cohort"] = bh_adjust(part["p"].astype(float).to_numpy())
        soft_rows.append(part)
    soft_bh = pd.concat(soft_rows, ignore_index=True)
    meta_s = soft_bh[soft_bh["cohort"] == "metabric"].copy()
    meta_s["q_bh_metabric_family"] = bh_adjust(meta_s["p"].astype(float).to_numpy())
    soft_bh = soft_bh.drop(columns=["q_bh_metabric_family"], errors="ignore")
    soft_bh = soft_bh.merge(
        meta_s[["soft_method", "q_bh_metabric_family"]], on="soft_method", how="left"
    )
    soft_bh.to_csv(TAB / "soft_residual_bh.csv", index=False)

    # Cross-cohort soft summary
    summary_rows = []
    for method, part in soft_bh.groupby("soft_method"):
        row = {"soft_method": method}
        for cohort in ("gse96058", "tcga", "metabric"):
            sub = part[part["cohort"] == cohort]
            if len(sub):
                row[f"{cohort}_p"] = float(sub["p"].iloc[0])
                row[f"{cohort}_q"] = float(sub["q_bh_within_cohort"].iloc[0])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(TAB / "soft_cross_cohort_summary.csv", index=False)
    print("BH tables written")
    return hard_bh, soft_bh


def _design_base(frame: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    design["OS_time"] = pd.to_numeric(frame["OS_time"], errors="coerce")
    design["OS_event"] = pd.to_numeric(frame["OS_event"], errors="coerce")
    age = pd.to_numeric(frame["age"], errors="coerce")
    design["age"] = age.fillna(age.median())
    pam = frame["PAM50"].astype(str)
    pam = pam.where(~pam.str.lower().isin(["nan", "none", ""]), np.nan)
    counts = pam.value_counts()
    frequent = set(counts[counts >= 5].index)
    pam = pam.where(pam.isin(frequent), "Other")
    ref = counts.idxmax()
    for level in sorted(set(pam.dropna().unique()) - {ref}):
        design[f"pam50_{level}"] = (pam == level).astype(float)
    return design


def bootstrap_delta_c(n_boot: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for cohort in ("gse96058", "tcga", "metabric"):
        path = SNF_DIR / f"{cohort}_assignments.csv"
        frame = pd.read_csv(path)
        frame = frame.dropna(subset=["OS_time", "OS_event"])
        frame = frame[(frame["OS_time"] > 0) & frame["OS_event"].isin([0, 1])].reset_index(drop=True)
        base = _design_base(frame)
        snf = base.copy()
        groups = frame["snf_community"].astype(str)
        ref = groups.value_counts().idxmax()
        for level in sorted(set(groups.unique()) - {ref}):
            snf[f"snf_{level}"] = (groups == level).astype(float)

        def fit_c(design, idx):
            d = design.iloc[idx].copy()
            try:
                m = CoxPHFitter(penalizer=0.0)
                m.fit(d, duration_col="OS_time", event_col="OS_event")
                return float(m.concordance_index_)
            except Exception:
                try:
                    m = CoxPHFitter(penalizer=0.1)
                    m.fit(d, duration_col="OS_time", event_col="OS_event")
                    return float(m.concordance_index_)
                except Exception:
                    return np.nan

        c0 = fit_c(base, np.arange(len(frame)))
        c1 = fit_c(snf, np.arange(len(frame)))
        deltas = []
        n = len(frame)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            # need events in bootstrap
            if frame.iloc[idx]["OS_event"].sum() < 10:
                continue
            d0 = fit_c(base, idx)
            d1 = fit_c(snf, idx)
            if np.isfinite(d0) and np.isfinite(d1):
                deltas.append(d1 - d0)
        deltas = np.asarray(deltas, dtype=float)
        lo, hi = np.nanpercentile(deltas, [2.5, 97.5]) if len(deltas) else (np.nan, np.nan)
        rows.append(
            {
                "cohort": cohort,
                "c_age_pam50": c0,
                "c_age_pam50_snf": c1,
                "delta_c": c1 - c0,
                "delta_c_boot_mean": float(np.nanmean(deltas)) if len(deltas) else np.nan,
                "delta_c_ci95_lo": float(lo),
                "delta_c_ci95_hi": float(hi),
                "n_boot_used": int(len(deltas)),
                "ci_excludes_zero": bool(lo > 0 or hi < 0) if np.isfinite(lo) else False,
            }
        )
        print(cohort, "ΔC", c1 - c0, "CI", lo, hi)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "delta_c_bootstrap.csv", index=False)
    return out


def rewrite_lead_decision_figure():
    df = pd.read_csv(TAB / "lead_decision_table.csv")
    pretty = {
        "snf_slpa": "SLPA",
        "snf_bigclam": "BIGCLAM",
        "snf_cesna_lite": "CESNA-lite",
        "snf_nocd_g": "NOCD-G",
        "snf_nocd_x": "NOCD-X",
    }
    role_pretty = {"sota_candidate": "candidate", "baseline_comparison": "baseline"}
    display = pd.DataFrame(
        {
            "Method": [pretty.get(m, m) for m in df["method"]],
            "Role": [role_pretty.get(r, r) for r in df["role"]],
            "Mixed\n10–40%": df["pass_mixed_10_40"].map({True: "Pass", False: "Fail"}),
            "Peak\n<0.85": df["pass_peak_lt_0.85"].map({True: "Pass", False: "Fail"}),
            "LumA\nOR>1": df["pass_luma_OR_gt_1"].map({True: "Pass", False: "Fail"}),
            "Quality": df["quality_pass"].map({True: "Pass", False: "Fail"}),
            "Lead\neligible": df["eligible_for_lead"].map({True: "Yes", False: "No"}),
            "Lead\npool": df["lead_pool"].map({True: "Yes", False: "No"}),
            "Selected": df["selected_lead"].map({True: "Yes", False: "No"}),
        }
    )
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    ax.axis("off")
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=list(display.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.2, 1.55)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#D9E8F5")
            cell.set_text_props(weight="bold")
        elif i > 0 and j >= 2:
            val = display.iloc[i - 1, j]
            if val in {"Pass", "Yes"}:
                cell.set_facecolor("#D9F2D9")
            elif val in {"Fail", "No"}:
                cell.set_facecolor("#F8D9D9")
    ax.set_title(
        "Soft-engine lead decision (METABRIC). BIGCLAM is a priori baseline → not lead-eligible.",
        fontsize=10,
        pad=6,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_lead_decision.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_lead_decision.pdf", bbox_inches="tight")
    plt.close(fig)
    print("lead decision figure rewritten")


def rewrite_soft_bakeoff():
    path = OVERLAP / "soft_method_ranking.csv"
    df = pd.read_csv(path)
    order = ["snf_nocd_x", "snf_cesna_lite", "snf_nocd_g", "snf_slpa", "snf_bigclam"]
    pretty = {
        "snf_nocd_x": "NOCD-X",
        "snf_nocd_g": "NOCD-G",
        "snf_slpa": "SLPA",
        "snf_cesna_lite": "CESNA-lite",
        "snf_bigclam": "BIGCLAM\n(baseline)",
    }
    df = df.set_index("soft_method").loc[[m for m in order if m in set(df["soft_method"])]].reset_index()
    # attach BH q for own_hard+mixed METABRIC
    soft_bh = pd.read_csv(TAB / "soft_residual_bh.csv")
    soft_bh = soft_bh[(soft_bh["cohort"] == "metabric") & (soft_bh["test"] == "own_hard_plus_mixed")]
    qmap = dict(zip(soft_bh["soft_method"], soft_bh["q_bh_metabric_family"]))
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    axes[0].bar(x, -np.log10(np.clip(df["metabric_hard_cox_p"], 1e-300, 1)), color="#2E6F9E")
    axes[0].axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1)
    axes[0].set_ylabel(r"$-\log_{10}$ uncorrected $p$")
    axes[0].set_title("Hard-cut Cox (METABRIC)")

    axes[1].bar(x, -np.log10(np.clip(df["metabric_own_hard_plus_mixed_p"], 1e-300, 1)), color="#4C8C4A")
    axes[1].axhline(-np.log10(0.05), color="#B33A3A", ls="--", lw=1)
    axes[1].set_ylabel(r"$-\log_{10}$ uncorrected $p$")
    axes[1].set_title("Own hard + mixed residual")
    for i, m in enumerate(df["soft_method"]):
        q = qmap.get(m, np.nan)
        if np.isfinite(q):
            axes[1].text(i, -np.log10(max(float(df.loc[i, "metabric_own_hard_plus_mixed_p"]), 1e-300)) + 0.08, f"q={q:.2f}", ha="center", fontsize=6.5)

    axes[2].bar(x, df["metabric_luma_OR"], color="#C47A2C")
    axes[2].axhline(1.0, color="#333", ls="--", lw=1)
    axes[2].set_ylabel("Odds ratio")
    axes[2].set_title("Mixed × LumA OR")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([pretty[m] for m in df["soft_method"]], fontsize=8)
    fig.suptitle("Overlapping engines on the SNF graph (METABRIC; q = BH within soft residual family)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_soft_bakeoff.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_soft_bakeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def rewrite_program_figure():
    prog = pd.read_csv(TAB / "mixed_program_enrichment.csv")
    # BH within engine
    prog = prog.copy()
    prog["q_bh"] = np.nan
    for eng, part in prog.groupby("engine"):
        prog.loc[part.index, "q_bh"] = bh_adjust(part["p"].to_numpy())
    prog.to_csv(TAB / "mixed_program_enrichment.csv", index=False)
    pivot = prog.pivot_table(index="program", columns="engine", values="delta")
    nmap = prog.groupby("engine")["n_mixed"].first().to_dict()
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(pivot))
    width = 0.35
    engines = list(pivot.columns)
    colors = {"bigclam": "#888888", "nocd_x": "#2E6F9E"}
    for i, eng in enumerate(engines):
        ax.bar(
            x + (i - 0.5) * width,
            pivot[eng],
            width=width,
            label=f"{eng} (n_mixed={int(nmap.get(eng, 0))})",
            color=colors.get(eng, None),
        )
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Mean z-score (mixed − single)\n(positive = higher in mixed)")
    ax.set_title("METABRIC programme scores: mixed vs single")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mixed_programs.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_mixed_programs.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    write_bh_tables()
    bootstrap_delta_c(n_boot=200)
    rewrite_lead_decision_figure()
    rewrite_soft_bakeoff()
    rewrite_program_figure()
    print("done")


if __name__ == "__main__":
    main()
