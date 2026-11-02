"""Survival, subtype agreement, and marker-program scores for the Q1 study.

Classifier accuracy is not computed here. The primary contrast is a Cox model
for BIGCLAM communities, or the mixed-membership flag, on top of age and PAM50.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2, kruskal
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MARKER_GENES = ("ESR1", "PGR", "ERBB2", "MKI67")

GENE_SETS = {
    "estrogen": ["ESR1", "PGR", "FOXA1", "GATA3", "XBP1", "TFF1", "TFF3"],
    "her2": ["ERBB2", "GRB7", "STARD3"],
    "proliferation": ["MKI67", "CCNB1", "CDK1", "MYBL2", "BIRC5", "AURKA"],
    "basal": ["KRT5", "KRT14", "KRT17", "EGFR", "FOXC1"],
    "claudin": ["CLDN3", "CLDN4", "CLDN7", "CDH1"],
}


def _finite_pair(labels, reference):
    labels = pd.Series(labels).reset_index(drop=True)
    reference = pd.Series(reference).reset_index(drop=True)
    mask = labels.notna() & reference.notna()
    mask &= reference.astype(str).str.len().gt(0)
    mask &= ~reference.astype(str).str.lower().isin(["nan", "none", "unknown"])
    return labels[mask].astype(str), reference[mask].astype(str)


def agreement_scores(labels, reference):
    """ARI and NMI against an external label. Missing reference labels are dropped."""
    left, right = _finite_pair(labels, reference)
    if len(left) < 10 or right.nunique() < 2 or left.nunique() < 2:
        return {"n": int(len(left)), "ari": np.nan, "nmi": np.nan}
    return {
        "n": int(len(left)),
        "ari": float(adjusted_rand_score(right, left)),
        "nmi": float(normalized_mutual_info_score(right, left)),
    }


def _survival_frame(clinical, labels):
    frame = clinical.copy()
    frame["group"] = pd.Series(labels, index=frame.index).astype(str)
    frame["OS_time"] = pd.to_numeric(frame["OS_time"], errors="coerce")
    frame["OS_event"] = pd.to_numeric(frame["OS_event"], errors="coerce")
    frame = frame.dropna(subset=["OS_time", "OS_event", "group"])
    frame = frame[(frame["OS_time"] > 0) & frame["OS_event"].isin([0, 1])]
    return frame


def logrank_pvalue(clinical, labels):
    frame = _survival_frame(clinical, labels)
    if frame["group"].nunique() < 2 or len(frame) < 20:
        return {"n": int(len(frame)), "n_events": int(frame["OS_event"].sum()) if len(frame) else 0, "p": np.nan}
    result = multivariate_logrank_test(frame["OS_time"], frame["group"], frame["OS_event"])
    return {
        "n": int(len(frame)),
        "n_events": int(frame["OS_event"].sum()),
        "p": float(result.p_value),
    }


def _design_matrix(frame, group_column, include_group=True):
    design = pd.DataFrame(index=frame.index)
    design["OS_time"] = frame["OS_time"].astype(float)
    design["OS_event"] = frame["OS_event"].astype(float)
    if frame["age"].notna().sum() >= 20:
        age = pd.to_numeric(frame["age"], errors="coerce")
        design["age"] = age.fillna(age.median()).astype(float)
    pam = frame["PAM50"] if "PAM50" in frame.columns else pd.Series(np.nan, index=frame.index)
    pam = pam.where(pam.notna(), np.nan).astype(str)
    pam = pam.where(~pam.isin(["nan", "None", ""]), np.nan)
    if pam.notna().sum() >= 20 and pam.dropna().nunique() >= 2:
        counts = pam.value_counts()
        frequent = set(counts[counts >= 5].index)
        pam = pam.where(pam.isin(frequent), "Other")
        reference = counts.idxmax()
        for level in sorted(set(pam.dropna().unique()) - {reference}):
            design[f"pam50_{level}"] = (pam == level).astype(float)
    if include_group:
        groups = frame[group_column].astype(str)
        counts = groups.value_counts()
        reference = counts.idxmax()
        for level in sorted(set(groups.unique()) - {reference}):
            design[f"grp_{level}"] = (groups == level).astype(float)
    covariates = [column for column in design.columns if column not in {"OS_time", "OS_event"}]
    return design, covariates


def _fit_cox(design, penalizer):
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(design, duration_col="OS_time", event_col="OS_event")
    return model


def cox_added_value(clinical, labels, group_name="community"):
    """Likelihood-ratio test of labels on top of age and PAM50.

    The returned p-value is not the unadjusted log-rank test. It asks whether
    the grouping still separates survival after PAM50 and age are in the model.
    """
    frame = _survival_frame(clinical, labels)
    frame = frame.rename(columns={"group": group_name})
    empty = {
        "n": int(len(frame)),
        "n_events": int(frame["OS_event"].sum()) if len(frame) else 0,
        "lr_stat": np.nan,
        "df": 0,
        "p": np.nan,
        "base_covariates": "",
        "note": "",
    }
    if frame[group_name].nunique() < 2 or len(frame) < 30:
        empty["note"] = "too few samples or groups"
        return empty
    full, _ = _design_matrix(frame, group_name, include_group=True)
    base, base_cols = _design_matrix(frame, group_name, include_group=False)
    group_cols = [column for column in full.columns if column.startswith("grp_")]
    if not group_cols:
        empty["note"] = "no group indicators"
        return empty
    penalizer = 0.0
    try:
        full_model = _fit_cox(full, penalizer)
        base_model = _fit_cox(base, penalizer) if base_cols else None
    except Exception:
        penalizer = 0.1
        try:
            full_model = _fit_cox(full, penalizer)
            base_model = _fit_cox(base, penalizer) if base_cols else None
        except Exception as exc:
            empty["note"] = str(exc)
            return empty
    if base_model is not None:
        lr_stat = 2.0 * (float(full_model.log_likelihood_) - float(base_model.log_likelihood_))
        degrees = max(len(full_model.params_) - len(base_model.params_), 1)
    else:
        lr_stat = float(full_model.log_likelihood_ratio_test().test_statistic)
        degrees = max(len(group_cols), 1)
    p_value = float(chi2.sf(max(lr_stat, 0.0), degrees))
    note = "adjusted for " + (",".join(base_cols) if base_cols else "intercept only")
    if penalizer:
        note += f"; penalizer={penalizer}"
    empty.update(
        {
            "lr_stat": lr_stat,
            "df": int(degrees),
            "p": p_value,
            "base_covariates": ",".join(base_cols),
            "note": note,
        }
    )
    return empty


def plot_kaplan_meier(clinical, labels, output_path, title):
    """Save one Kaplan–Meier curve per group. Returns the path or None."""
    frame = _survival_frame(clinical, labels)
    if frame["group"].nunique() < 2:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fitter = KaplanMeierFitter()
    for group in sorted(frame["group"].unique(), key=str):
        subset = frame[frame["group"] == group]
        if len(subset) < 5:
            continue
        fitter.fit(subset["OS_time"], subset["OS_event"], label=f"{group} (n={len(subset)})")
        fitter.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("Days")
    ax.set_ylabel("Overall survival")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def marker_table(X, genes, communities):
    """Per-community means for canonical genes and curated programs.

    ``X`` is samples by genes, already z-scored inside the cohort. Program
    scores are the mean of the member genes that are on the locked list.
    """
    gene_index = {gene: i for i, gene in enumerate(genes)}
    communities = np.asarray(communities)
    rows = []
    for gene in MARKER_GENES:
        if gene not in gene_index:
            rows.append({"feature": gene, "kind": "gene", "n_genes_in_set": 0, "kruskal_p": np.nan})
            continue
        values = X[:, gene_index[gene]]
        grouped = [values[communities == community] for community in sorted(np.unique(communities))]
        grouped = [chunk for chunk in grouped if len(chunk) >= 3]
        p_value = float(kruskal(*grouped).pvalue) if len(grouped) >= 2 else np.nan
        row = {"feature": gene, "kind": "gene", "n_genes_in_set": 1, "kruskal_p": p_value}
        for community in sorted(np.unique(communities)):
            row[f"mean_c{community}"] = float(np.mean(values[communities == community]))
        rows.append(row)
    for name, members in GENE_SETS.items():
        present = [gene_index[gene] for gene in members if gene in gene_index]
        if not present:
            rows.append({"feature": name, "kind": "program", "n_genes_in_set": 0, "kruskal_p": np.nan})
            continue
        scores = X[:, present].mean(axis=1)
        grouped = [scores[communities == community] for community in sorted(np.unique(communities))]
        grouped = [chunk for chunk in grouped if len(chunk) >= 3]
        p_value = float(kruskal(*grouped).pvalue) if len(grouped) >= 2 else np.nan
        row = {
            "feature": name,
            "kind": "program",
            "n_genes_in_set": len(present),
            "kruskal_p": p_value,
        }
        for community in sorted(np.unique(communities)):
            row[f"mean_c{community}"] = float(np.mean(scores[communities == community]))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_partition(clinical, labels, method, cohort):
    """One results row: external agreement plus survival with and without PAM50."""
    logrank = logrank_pvalue(clinical, labels)
    added = cox_added_value(clinical, labels)
    pam = agreement_scores(labels, clinical["PAM50"] if "PAM50" in clinical.columns else None)
    intclust = {"n": 0, "ari": np.nan, "nmi": np.nan}
    if "intclust" in clinical.columns and clinical["intclust"].notna().any():
        intclust = agreement_scores(labels, clinical["intclust"])
    return {
        "cohort": cohort,
        "method": method,
        "n": logrank["n"],
        "n_events": logrank["n_events"],
        "logrank_p": logrank["p"],
        "cox_lr_stat": added["lr_stat"],
        "cox_df": added["df"],
        "cox_p_given_age_pam50": added["p"],
        "cox_note": added["note"],
        "pam50_n": pam["n"],
        "pam50_ari": pam["ari"],
        "pam50_nmi": pam["nmi"],
        "intclust_n": intclust["n"],
        "intclust_ari": intclust["ari"],
        "intclust_nmi": intclust["nmi"],
    }


def subtype_enrichment(clinical, binary_flag, subtype_col="PAM50", focus="Claudin-low"):
    """Odds ratio of a binary flag inside one subtype vs all other labeled tumors."""
    from scipy.stats import fisher_exact

    frame = clinical.copy()
    frame["_flag"] = np.asarray(binary_flag).astype(int)
    if subtype_col not in frame.columns:
        return {"subtype": focus, "n_focus": 0, "odds_ratio": np.nan, "p": np.nan}
    labels = frame[subtype_col].astype(str)
    valid = labels.notna() & ~labels.str.lower().isin(["nan", "none", "unknown", ""])
    frame = frame.loc[valid]
    labels = frame[subtype_col].astype(str)
    if "claudin" in str(focus).lower():
        focus_mask = labels.str.contains("claudin", case=False, na=False)
    else:
        focus_key = str(focus).lower().replace(" ", "").replace("-", "")
        focus_mask = labels.str.lower().str.replace(" ", "").str.replace("-", "", regex=False) == focus_key
    table = np.array(
        [
            [int(((frame["_flag"] == 1) & focus_mask).sum()), int(((frame["_flag"] == 0) & focus_mask).sum())],
            [int(((frame["_flag"] == 1) & ~focus_mask).sum()), int(((frame["_flag"] == 0) & ~focus_mask).sum())],
        ]
    )
    if table.min() < 0 or table.sum() < 20:
        return {"subtype": focus, "n_focus": int(focus_mask.sum()), "odds_ratio": np.nan, "p": np.nan}
    odds, p_value = fisher_exact(table)
    return {
        "subtype": focus,
        "n_focus": int(focus_mask.sum()),
        "n_flag_in_focus": int(table[0, 0]),
        "odds_ratio": float(odds),
        "p": float(p_value),
        "table": table.tolist(),
    }


def cox_hard_plus_soft(clinical, hard_labels, soft_scores, soft_name="mixed"):
    """LR test of hard labels + soft score vs hard labels alone, still with age+PAM50.

    This is the residual-overlap test: does soft membership add survival signal
    beyond the hard partition and clinical subtype?
    """
    frame = clinical.copy()
    frame["hard"] = pd.Series(hard_labels, index=frame.index).astype(str)
    frame["soft"] = pd.to_numeric(pd.Series(soft_scores, index=frame.index), errors="coerce")
    frame["OS_time"] = pd.to_numeric(frame["OS_time"], errors="coerce")
    frame["OS_event"] = pd.to_numeric(frame["OS_event"], errors="coerce")
    frame = frame.dropna(subset=["OS_time", "OS_event", "hard", "soft"])
    frame = frame[(frame["OS_time"] > 0) & frame["OS_event"].isin([0, 1])]
    empty = {"n": int(len(frame)), "n_events": int(frame["OS_event"].sum()) if len(frame) else 0, "p": np.nan, "note": ""}
    if len(frame) < 40 or frame["hard"].nunique() < 2:
        empty["note"] = "too few samples"
        return empty
    base, _ = _design_matrix(frame.rename(columns={"hard": "group"}), "group", include_group=True)
    full = base.copy()
    if soft_name == "mixed" or set(frame["soft"].unique()).issubset({0, 1, 0.0, 1.0}):
        full["soft"] = frame["soft"].astype(float).to_numpy()
    else:
        soft = frame["soft"].astype(float)
        full["soft"] = ((soft - soft.mean()) / (soft.std() + 1e-8)).to_numpy()
    try:
        base_model = _fit_cox(base, 0.0)
        full_model = _fit_cox(full, 0.0)
    except Exception:
        try:
            base_model = _fit_cox(base, 0.1)
            full_model = _fit_cox(full, 0.1)
        except Exception as exc:
            empty["note"] = str(exc)
            return empty
    lr_stat = 2.0 * (float(full_model.log_likelihood_) - float(base_model.log_likelihood_))
    p_value = float(chi2.sf(max(lr_stat, 0.0), 1))
    return {
        "n": int(len(frame)),
        "n_events": int(frame["OS_event"].sum()),
        "lr_stat": lr_stat,
        "p": p_value,
        "soft_name": soft_name,
        "note": "age+PAM50+hard vs age+PAM50+hard+soft",
    }
