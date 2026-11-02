"""SOTA manuscript analyses on existing SNF + overlap outputs.

Does not re-fit clustering. Reads results/snf_bigclam/*_assignments.csv and
produces the tables a framework paper needs:

  - concordance (C-index) for age+PAM50 vs +SNF vs +overlap
  - within-LumA / within-LumB residual survival for mixed and entropy
  - PAM50 and IntClust composition of SNF communities
  - cross-cohort SNF community matching via PAM50 composition

Usage:
    python scripts/run_sota_analyses.py --results-dir results/snf_bigclam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, chi2_contingency, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clean_survival(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["OS_time"] = pd.to_numeric(out["OS_time"], errors="coerce")
    out["OS_event"] = pd.to_numeric(out["OS_event"], errors="coerce")
    out = out.dropna(subset=["OS_time", "OS_event"])
    out = out[(out["OS_time"] > 0) & out["OS_event"].isin([0, 1])]
    return out


def _add_age_pam50(design: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if "age" in frame.columns and frame["age"].notna().sum() >= 20:
        age = pd.to_numeric(frame["age"], errors="coerce")
        design["age"] = age.fillna(age.median()).astype(float)
    if "PAM50" not in frame.columns:
        return design
    pam = frame["PAM50"].astype(str)
    pam = pam.where(~pam.str.lower().isin(["nan", "none", ""]), np.nan)
    if pam.notna().sum() < 20 or pam.dropna().nunique() < 2:
        return design
    counts = pam.value_counts()
    frequent = set(counts[counts >= 5].index)
    pam = pam.where(pam.isin(frequent), "Other")
    reference = counts.idxmax()
    for level in sorted(set(pam.dropna().unique()) - {reference}):
        design[f"pam50_{level}"] = (pam == level).astype(float)
    return design


def _fit_cox(design: pd.DataFrame, penalizer: float = 0.0):
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(design, duration_col="OS_time", event_col="OS_event")
    return model


def concordance_table(frame: pd.DataFrame, cohort: str) -> list[dict]:
    """C-index for nested models: clinical → +SNF → +soft."""
    frame = _clean_survival(frame)
    rows = []
    base = pd.DataFrame(
        {
            "OS_time": frame["OS_time"].astype(float),
            "OS_event": frame["OS_event"].astype(float),
        },
        index=frame.index,
    )
    base = _add_age_pam50(base, frame)

    models = [("age_pam50", base)]
    if "snf_community" in frame.columns:
        snf = base.copy()
        groups = frame["snf_community"].astype(str)
        ref = groups.value_counts().idxmax()
        for level in sorted(set(groups.unique()) - {ref}):
            snf[f"snf_{level}"] = (groups == level).astype(float)
        models.append(("age_pam50_snf", snf))
    if "snf_community" in frame.columns and "mixed" in frame.columns:
        soft = models[-1][1].copy()
        soft["mixed"] = pd.to_numeric(frame["mixed"], errors="coerce").fillna(0).astype(float)
        models.append(("age_pam50_snf_mixed", soft))
    if "snf_community" in frame.columns and "entropy" in frame.columns:
        ent = models[1][1].copy() if len(models) > 1 else base.copy()
        e = pd.to_numeric(frame["entropy"], errors="coerce")
        ent["entropy_z"] = ((e - e.mean()) / (e.std() + 1e-8)).astype(float)
        models.append(("age_pam50_snf_entropy", ent))

    for name, design in models:
        try:
            fitted = _fit_cox(design, 0.0)
        except Exception:
            try:
                fitted = _fit_cox(design, 0.1)
            except Exception as exc:
                rows.append(
                    {
                        "cohort": cohort,
                        "model": name,
                        "n": len(design),
                        "n_events": int(design["OS_event"].sum()),
                        "c_index": np.nan,
                        "note": str(exc),
                    }
                )
                continue
        rows.append(
            {
                "cohort": cohort,
                "model": name,
                "n": len(design),
                "n_events": int(design["OS_event"].sum()),
                "c_index": float(fitted.concordance_index_),
                "loglik": float(fitted.log_likelihood_),
                "note": "",
            }
        )
    return rows


def within_subtype_residual(frame: pd.DataFrame, cohort: str, subtype: str) -> list[dict]:
    """Within one PAM50 class: does mixed / entropy separate survival?"""
    frame = _clean_survival(frame)
    pam = frame["PAM50"].astype(str)
    key = subtype.lower().replace(" ", "").replace("-", "")
    mask = pam.str.lower().str.replace(" ", "").str.replace("-", "", regex=False) == key
    sub = frame.loc[mask].copy()
    rows = []
    if len(sub) < 40 or sub["OS_event"].sum() < 10:
        return [
            {
                "cohort": cohort,
                "subtype": subtype,
                "n": int(len(sub)),
                "n_events": int(sub["OS_event"].sum()) if len(sub) else 0,
                "predictor": "mixed",
                "logrank_p": np.nan,
                "cox_p_age": np.nan,
                "hr": np.nan,
                "note": "too few",
            }
        ]

    for predictor in ("mixed", "entropy"):
        if predictor not in sub.columns:
            continue
        values = pd.to_numeric(sub[predictor], errors="coerce")
        design = pd.DataFrame(
            {
                "OS_time": sub["OS_time"].astype(float),
                "OS_event": sub["OS_event"].astype(float),
                "x": values.astype(float),
            },
            index=sub.index,
        )
        if "age" in sub.columns and sub["age"].notna().sum() >= 20:
            age = pd.to_numeric(sub["age"], errors="coerce")
            design["age"] = age.fillna(age.median()).astype(float)
        design = design.dropna()
        if predictor == "mixed":
            a = sub.loc[values == 1]
            b = sub.loc[values == 0]
            if len(a) >= 10 and len(b) >= 10:
                lr = logrank_test(a["OS_time"], b["OS_time"], a["OS_event"], b["OS_event"])
                logrank_p = float(lr.p_value)
            else:
                logrank_p = np.nan
        else:
            # median split for KM/log-rank
            med = values.median()
            a = sub.loc[values > med]
            b = sub.loc[values <= med]
            if len(a) >= 10 and len(b) >= 10:
                lr = logrank_test(a["OS_time"], b["OS_time"], a["OS_event"], b["OS_event"])
                logrank_p = float(lr.p_value)
            else:
                logrank_p = np.nan
            design["x"] = ((design["x"] - design["x"].mean()) / (design["x"].std() + 1e-8)).astype(float)

        try:
            model = _fit_cox(design, 0.0)
            hr = float(np.exp(model.params_["x"]))
            cox_p = float(model.summary.loc["x", "p"])
            note = ""
        except Exception as exc:
            hr, cox_p, note = np.nan, np.nan, str(exc)

        rows.append(
            {
                "cohort": cohort,
                "subtype": subtype,
                "n": int(len(design)),
                "n_events": int(design["OS_event"].sum()),
                "n_mixed": int((values == 1).sum()) if predictor == "mixed" else np.nan,
                "predictor": predictor,
                "logrank_p": logrank_p,
                "cox_p_age": cox_p,
                "hr": hr,
                "note": note,
            }
        )
    return rows


def plot_within_luma_km(frame: pd.DataFrame, out_path: Path, cohort: str) -> None:
    frame = _clean_survival(frame)
    pam = frame["PAM50"].astype(str)
    mask = pam.str.lower().str.replace(" ", "").str.replace("-", "", regex=False) == "luma"
    sub = frame.loc[mask]
    if "mixed" not in sub.columns or sub["mixed"].nunique() < 2 or len(sub) < 40:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fitter = KaplanMeierFitter()
    for flag, label in ((0, "single"), (1, "mixed")):
        chunk = sub.loc[pd.to_numeric(sub["mixed"], errors="coerce") == flag]
        if len(chunk) < 5:
            continue
        fitter.fit(chunk["OS_time"], chunk["OS_event"], label=f"LumA {label} (n={len(chunk)})")
        fitter.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title(f"{cohort}: within-LumA mixed vs single")
    ax.set_xlabel("Days")
    ax.set_ylabel("Overall survival")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def composition_tables(frame: pd.DataFrame, cohort: str, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-normalized PAM50 and IntClust composition of network communities."""
    labels = frame[label_col].astype(str)
    pam = frame["PAM50"].astype(str)
    valid = ~pam.str.lower().isin(["nan", "none", "", "unknown"])
    ct = pd.crosstab(labels[valid], pam[valid])
    frac = ct.div(ct.sum(axis=1), axis=0)
    frac.insert(0, "cohort", cohort)
    frac.insert(1, "partition", label_col)
    frac = frac.reset_index().rename(columns={label_col: "community"})

    intclust = pd.DataFrame()
    if "intclust" in frame.columns and frame["intclust"].notna().any():
        ic = frame["intclust"].astype(str)
        ok = ~ic.str.lower().isin(["nan", "none", ""])
        ict = pd.crosstab(labels[ok], ic[ok])
        ifrac = ict.div(ict.sum(axis=1), axis=0)
        ifrac.insert(0, "cohort", cohort)
        ifrac.insert(1, "partition", label_col)
        intclust = ifrac.reset_index().rename(columns={label_col: "community"})
    return frac, intclust


def match_communities(disc_comp: pd.DataFrame, val_comp: pd.DataFrame, pam_cols: list[str]) -> pd.DataFrame:
    """Hungarian match of validation SNF communities to discovery by PAM50 profile."""
    d = disc_comp.set_index("community")[pam_cols].fillna(0.0)
    v = val_comp.set_index("community")[pam_cols].fillna(0.0)
    # correlation cost: maximize corr → minimize 1-corr
    cost = np.zeros((len(v), len(d)))
    v_ids = list(v.index.astype(str))
    d_ids = list(d.index.astype(str))
    for i, vid in enumerate(v_ids):
        for j, did in enumerate(d_ids):
            a = v.loc[vid].astype(float).to_numpy()
            b = d.loc[did].astype(float).to_numpy()
            if a.std() < 1e-12 or b.std() < 1e-12:
                cost[i, j] = 1.0
            else:
                cost[i, j] = 1.0 - float(np.corrcoef(a, b)[0, 1])
    row_ind, col_ind = linear_sum_assignment(cost)
    rows = []
    for r, c in zip(row_ind, col_ind):
        rows.append(
            {
                "validation_community": v_ids[r],
                "discovery_community": d_ids[c],
                "pam50_profile_corr": 1.0 - float(cost[r, c]),
            }
        )
    return pd.DataFrame(rows)


def enrichment_all_subtypes(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Fisher OR for mixed flag across every PAM50 label present."""
    frame = frame.copy()
    frame["_flag"] = pd.to_numeric(frame["mixed"], errors="coerce").fillna(0).astype(int)
    pam = frame["PAM50"].astype(str)
    valid = ~pam.str.lower().isin(["nan", "none", "", "unknown"])
    frame = frame.loc[valid]
    pam = frame["PAM50"].astype(str)
    rows = []
    for subtype in sorted(pam.unique()):
        focus = pam == subtype
        table = np.array(
            [
                [int(((frame["_flag"] == 1) & focus).sum()), int(((frame["_flag"] == 0) & focus).sum())],
                [int(((frame["_flag"] == 1) & ~focus).sum()), int(((frame["_flag"] == 0) & ~focus).sum())],
            ]
        )
        if table.sum() < 20 or focus.sum() < 5:
            continue
        odds, p_value = fisher_exact(table)
        rows.append(
            {
                "cohort": cohort,
                "subtype": subtype,
                "n_subtype": int(focus.sum()),
                "n_mixed_in_subtype": int(table[0, 0]),
                "odds_ratio": float(odds),
                "p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/snf_bigclam")
    args = parser.parse_args()
    results = ROOT / args.results_dir
    out = results / "sota"
    out.mkdir(parents=True, exist_ok=True)

    cohorts = ["gse96058", "tcga", "metabric"]
    c_rows, within_rows, enrich_rows = [], [], []
    pam_comps, ic_comps = [], []
    pam_by_cohort = {}

    for name in cohorts:
        path = results / f"{name}_assignments.csv"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        frame = pd.read_csv(path)
        print(f"SOTA analyses: {name} n={len(frame)}")
        c_rows.extend(concordance_table(frame, name))
        for subtype in ("LumA", "LumB", "Basal", "Her2", "Normal", "Claudin-low"):
            within_rows.extend(within_subtype_residual(frame, name, subtype))
        enrich_rows.append(enrichment_all_subtypes(frame, name))
        pam_frac, ic_frac = composition_tables(frame, name, "snf_community")
        pam_comps.append(pam_frac)
        if len(ic_frac):
            ic_comps.append(ic_frac)
        pam_by_cohort[name] = pam_frac
        plot_within_luma_km(frame, out / f"{name}_km_within_luma_mixed.png", name)

    pd.DataFrame(c_rows).to_csv(out / "concordance.csv", index=False)
    pd.DataFrame(within_rows).to_csv(out / "within_subtype_residual.csv", index=False)
    if enrich_rows:
        pd.concat(enrich_rows, ignore_index=True).to_csv(out / "mixed_pam50_enrichment_all.csv", index=False)
    if pam_comps:
        pam_all = pd.concat(pam_comps, ignore_index=True)
        pam_all.to_csv(out / "snf_pam50_composition.csv", index=False)
    if ic_comps:
        pd.concat(ic_comps, ignore_index=True).to_csv(out / "snf_intclust_composition.csv", index=False)

    # Cross-cohort match discovery → validation
    match_rows = []
    if "gse96058" in pam_by_cohort:
        disc = pam_by_cohort["gse96058"]
        pam_cols = [c for c in disc.columns if c not in {"cohort", "partition", "community"}]
        for val_name in ("tcga", "metabric"):
            if val_name not in pam_by_cohort:
                continue
            matched = match_communities(disc, pam_by_cohort[val_name], pam_cols)
            matched.insert(0, "validation_cohort", val_name)
            matched.insert(0, "discovery_cohort", "gse96058")
            match_rows.append(matched)
    if match_rows:
        pd.concat(match_rows, ignore_index=True).to_csv(out / "cross_cohort_snf_match.csv", index=False)

    # Primary manuscript numbers (honest ranking)
    survival = results / "survival_comparison.csv"
    residual = results / "residual_overlap_cox.csv"
    primary = {
        "paper_identity": "Hard SNF subtypes + soft overlap quantification (dual-output framework)",
        "not_claiming": "BIGCLAM is not sold as the best hard clusterer",
        "primary_hard_method": "SNF (Euclidean + correlation fusion)",
        "overlap_module": "community affiliation on sparsified SNF graph (BIGCLAM engine)",
        "cohorts": cohorts,
        "concordance_file": str(out / "concordance.csv"),
        "within_subtype_file": str(out / "within_subtype_residual.csv"),
    }
    if survival.exists():
        sc = pd.read_csv(survival)
        meta = sc[sc["cohort"] == "metabric"].sort_values("cox_p_given_age_pam50")
        primary["metabric_hard_ranking"] = meta[["method", "cox_p_given_age_pam50", "pam50_ari", "intclust_ari"]].to_dict(
            orient="records"
        )
    if residual.exists():
        primary["residual_overlap"] = pd.read_csv(residual).to_dict(orient="records")
    (out / "primary_claims.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")
    print(f"Wrote SOTA tables to {out}")


if __name__ == "__main__":
    main()
