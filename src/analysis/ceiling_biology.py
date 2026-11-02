"""Biology checks that reuse locked soft memberships.

Within-cluster and within-LumA survival, IntClust / Claudin-low odds ratios,
within-cluster expression contrasts on the locked gene list, and a SCAN-B
replication of the mixed-versus-single programme phenotype. Treatment covariates
are used only when the clinical table already has them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.analysis.ceiling_mechanistic import LOCKED_RATIO, bh_qvalues, load_fused_affinity
from src.analysis.q1_clustering import membership_summary
from src.evaluation.q1_survival import GENE_SETS, cox_added_value, subtype_enrichment

ENGINES = ("nocd_x", "bigclam")
TREATMENT_COLUMNS = (
    "hormone_therapy",
    "chemotherapy",
    "radio_therapy",
    "HORMONE_THERAPY",
    "CHEMOTHERAPY",
    "RADIO_THERAPY",
)


def treatment_columns(clinical):
    """Return treatment columns that are already on the clinical table."""
    found = [column for column in TREATMENT_COLUMNS if column in clinical.columns]
    usable = []
    for column in found:
        values = clinical[column].dropna()
        if values.nunique() >= 2 and len(values) >= 30:
            usable.append(column)
    return usable


def affinity_margin(fused, labels):
    """Own-cluster mean affinity minus other-cluster mean affinity."""
    affinity = np.asarray(fused, dtype=np.float64)
    labels = np.asarray(labels)
    n = affinity.shape[0]
    margin = np.zeros(n, dtype=np.float64)
    for community in np.unique(labels):
        members = np.flatnonzero(labels == community)
        others = np.flatnonzero(labels != community)
        if len(members) < 2 or len(others) == 0:
            continue
        block = affinity[np.ix_(members, members)]
        np.fill_diagonal(block, np.nan)
        own = np.nanmean(block, axis=1)
        other = affinity[np.ix_(members, others)].mean(axis=1)
        margin[members] = own - other
    return margin


def _age_only(clinical):
    """Copy clinical data with PAM50 removed so Cox adjusts for age only."""
    frame = clinical.copy()
    if "PAM50" in frame.columns:
        frame["PAM50"] = np.nan
    return frame


def _binary_cox(clinical, mask, min_each=20):
    labels = np.where(np.asarray(mask), "boundary", "other")
    if int(np.sum(mask)) < min_each or int(np.sum(~np.asarray(mask))) < min_each:
        return {"n": int(len(clinical)), "n_events": 0, "p": np.nan, "note": "too few in one class"}
    return cox_added_value(_age_only(clinical), labels, group_name="boundary")


def _load_locked(locked_dir, cohort, engine):
    root = Path(locked_dir)
    F = np.load(root / f"{cohort}_snf_{engine}_F.npy")
    hard = np.load(root / f"{cohort}_snf_labels.npy")
    return F, hard


def survival_family(clinical, F, hard, fused, ratio, engine, cohort):
    """Age-adjusted tests of mixed and boundary flags inside hard groups and LumA."""
    _, mixed, strength = membership_summary(F, ratio=ratio)
    margin = affinity_margin(fused, hard)
    high_margin = margin >= np.nanmedian(margin)
    boundary = high_margin & (strength >= ratio)
    rows = []
    pam = clinical["PAM50"].astype(str) if "PAM50" in clinical.columns else pd.Series("", index=clinical.index)

    def add(scope, mask_rows, flag, flag_name):
        if mask_rows.sum() < 40:
            return
        sub_clinical = clinical.loc[mask_rows]
        sub_flag = np.asarray(flag)[mask_rows.to_numpy()]
        result = _binary_cox(sub_clinical, sub_flag)
        rows.append({
            "cohort": cohort,
            "engine": engine,
            "scope": scope,
            "flag": flag_name,
            "n": result.get("n"),
            "n_flag": int(np.sum(sub_flag)),
            "n_events": result.get("n_events"),
            "p": result.get("p"),
            "note": result.get("note", ""),
        })

    index = clinical.index
    row_mask = pd.Series(True, index=index)
    for community in sorted(np.unique(hard)):
        in_c = pd.Series(hard == community, index=index)
        add(f"hard_{int(community)}", in_c, mixed, "mixed")
        add(f"hard_{int(community)}", in_c, boundary, "boundary")
    luma = pam.str.lower().str.replace(" ", "", regex=False).str.replace("-", "", regex=False).eq("luma")
    add("LumA", luma, mixed, "mixed")
    add("LumA", luma, boundary, "boundary")
    add("all", row_mask, boundary, "boundary")
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["cohort", "engine", "scope", "flag", "n", "n_flag", "n_events", "p", "note", "q"])
    else:
        frame["q"] = bh_qvalues(frame["p"].to_numpy())
    return frame, mixed, boundary


def orthogonal_enrichment(clinical, mixed, engine, cohort):
    """Odds ratios of the mixed flag for each IntClust level and for Claudin-low."""
    rows = []
    if "intclust" in clinical.columns:
        levels = sorted({str(v) for v in clinical["intclust"].dropna().unique() if str(v).lower() not in {"nan", ""}})
        for level in levels:
            result = subtype_enrichment(clinical, mixed, subtype_col="intclust", focus=level)
            rows.append({
                "cohort": cohort,
                "engine": engine,
                "label_family": "intclust",
                "level": level,
                **{k: result.get(k) for k in ("n_focus", "n_flag_in_focus", "odds_ratio", "p")},
            })
    result = subtype_enrichment(clinical, mixed, focus="Claudin-low")
    rows.append({
        "cohort": cohort,
        "engine": engine,
        "label_family": "PAM50",
        "level": "Claudin-low",
        **{k: result.get(k) for k in ("n_focus", "n_flag_in_focus", "odds_ratio", "p")},
    })
    result = subtype_enrichment(clinical, mixed, focus="LumA")
    rows.append({
        "cohort": cohort,
        "engine": engine,
        "label_family": "PAM50",
        "level": "LumA",
        **{k: result.get(k) for k in ("n_focus", "n_flag_in_focus", "odds_ratio", "p")},
    })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q"] = bh_qvalues(frame["p"].to_numpy())
    return frame


def _program_scores(X, genes):
    index = {gene: i for i, gene in enumerate(genes)}
    scores = {}
    for name, members in GENE_SETS.items():
        present = [index[gene] for gene in members if gene in index]
        if present:
            scores[name] = X[:, present].mean(axis=1)
    return scores


def expression_contrasts(X, genes, mixed, hard, engine, cohort, out_dir):
    """Programme scores and locked-gene t-tests, mixed versus single, inside one hard cluster."""
    scores = _program_scores(X, genes)
    mixed = np.asarray(mixed, dtype=bool)
    hard = np.asarray(hard)
    # Cluster with the most mixed patients and at least 15 single-membership patients.
    best = None
    for community in np.unique(hard):
        in_c = hard == community
        n_mixed = int((in_c & mixed).sum())
        n_single = int((in_c & ~mixed).sum())
        if n_mixed >= 15 and n_single >= 15 and (best is None or n_mixed > best[0]):
            best = (n_mixed, int(community))
    rows = []
    gene_frame = pd.DataFrame()
    if best is None:
        return pd.DataFrame(rows), gene_frame
    _, community = best
    in_c = hard == community
    a = in_c & mixed
    b = in_c & ~mixed
    for name, values in scores.items():
        stat = ttest_ind(values[a], values[b], equal_var=False)
        rows.append({
            "cohort": cohort,
            "engine": engine,
            "hard_cluster": community,
            "feature": name,
            "kind": "program",
            "n_mixed": int(a.sum()),
            "n_single": int(b.sum()),
            "delta_mixed_minus_single": float(values[a].mean() - values[b].mean()),
            "p": float(stat.pvalue),
        })
    # Overall (not within cluster) programme contrast, reported separately and not in the BH family.
    for name, values in scores.items():
        if mixed.sum() < 15 or (~mixed).sum() < 15:
            continue
        stat = ttest_ind(values[mixed], values[~mixed], equal_var=False)
        rows.append({
            "cohort": cohort,
            "engine": engine,
            "hard_cluster": -1,
            "feature": name,
            "kind": "program_overall",
            "n_mixed": int(mixed.sum()),
            "n_single": int((~mixed).sum()),
            "delta_mixed_minus_single": float(values[mixed].mean() - values[~mixed].mean()),
            "p": float(stat.pvalue),
        })
    frame = pd.DataFrame(rows)
    within = frame["kind"] == "program"
    frame["q"] = np.nan
    if within.any():
        frame.loc[within, "q"] = bh_qvalues(frame.loc[within, "p"].to_numpy())

    gene_p = ttest_ind(X[a], X[b], axis=0, equal_var=False, nan_policy="omit").pvalue
    delta = X[a].mean(axis=0) - X[b].mean(axis=0)
    gene_q = bh_qvalues(gene_p)
    gene_frame = pd.DataFrame({
        "cohort": cohort,
        "engine": engine,
        "hard_cluster": community,
        "gene": genes,
        "delta_mixed_minus_single": delta,
        "p": gene_p,
        "q": gene_q,
    })
    gene_frame = gene_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["p"])
    gene_frame["abs_delta"] = gene_frame["delta_mixed_minus_single"].abs()
    top = gene_frame.sort_values(["q", "abs_delta"], ascending=[True, False]).head(25)
    top.drop(columns=["abs_delta"]).to_csv(Path(out_dir) / f"{cohort}_{engine}_within_cluster_genes.csv", index=False)
    n_pass = int((gene_frame["q"] < 0.05).sum())
    frame = pd.concat([
        frame,
        pd.DataFrame([{
            "cohort": cohort,
            "engine": engine,
            "hard_cluster": community,
            "feature": "locked_genes_bh",
            "kind": "gene_family",
            "n_mixed": int(a.sum()),
            "n_single": int(b.sum()),
            "delta_mixed_minus_single": np.nan,
            "p": np.nan,
            "q": np.nan,
            "n_genes_q_lt_0.05": n_pass,
            "n_genes_tested": int(len(gene_frame)),
        }]),
    ], ignore_index=True)
    return frame, gene_frame


def run_biology(harmonized, graph_dir, locked_dir, out_dir, genes):
    """METABRIC-primary biology plus a SCAN-B programme replication."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    survival_frames = []
    enrich_frames = []
    program_frames = []
    treatment_report = {}
    for cohort in ("metabric", "gse96058"):
        clinical = harmonized["cohorts"][cohort]["clinical"]
        X = harmonized["cohorts"][cohort]["X"]
        fused = load_fused_affinity(graph_dir, cohort)
        found = treatment_columns(clinical)
        treatment_report[cohort] = {
            "present": found,
            "note": (
                "No hormone, chemotherapy, or radiotherapy column is on the harmonized "
                "clinical table, so treatment was not added as a covariate."
                if not found
                else "Treatment columns used only as recorded; values were not imputed."
            ),
        }
        for engine in ENGINES:
            print(f"  biology {cohort} {engine}")
            F, hard = _load_locked(locked_dir, cohort, engine)
            if F.shape[0] != len(clinical) or hard.shape[0] != len(clinical):
                raise ValueError(f"Locked {engine} labels do not match {cohort} clinical rows")
            family, mixed, _boundary = survival_family(
                clinical, F, hard, fused, LOCKED_RATIO[engine], engine, cohort
            )
            survival_frames.append(family)
            enrich_frames.append(orthogonal_enrichment(clinical, mixed, engine, cohort))
            programs, _genes = expression_contrasts(X, genes, mixed, hard, engine, cohort, out)
            program_frames.append(programs)
            if found:
                for column in found:
                    flag = clinical[column].astype(str)
                    # Stratify the mixed flag inside the treated subset when both levels exist.
                    for level in sorted(flag.dropna().unique()):
                        subset = flag == level
                        if subset.sum() < 40:
                            continue
                        result = _binary_cox(clinical.loc[subset], mixed[subset.to_numpy()])
                        survival_frames.append(pd.DataFrame([{
                            "cohort": cohort,
                            "engine": engine,
                            "scope": f"{column}={level}",
                            "flag": "mixed",
                            "n": result.get("n"),
                            "n_flag": int(mixed[subset.to_numpy()].sum()),
                            "n_events": result.get("n_events"),
                            "p": result.get("p"),
                            "note": result.get("note", ""),
                            "q": np.nan,
                        }]))

    survival = pd.concat(survival_frames, ignore_index=True) if survival_frames else pd.DataFrame()
    enrich = pd.concat(enrich_frames, ignore_index=True) if enrich_frames else pd.DataFrame()
    programs = pd.concat(program_frames, ignore_index=True) if program_frames else pd.DataFrame()
    survival.to_csv(out / "boundary_survival.csv", index=False)
    enrich.to_csv(out / "orthogonal_enrichment.csv", index=False)
    programs.to_csv(out / "program_contrasts.csv", index=False)
    (out / "treatment_status.json").write_text(json.dumps(treatment_report, indent=2), encoding="utf-8")

    def _hits(frame, cohort, q_col="q"):
        if frame.empty or q_col not in frame.columns:
            return []
        part = frame[(frame["cohort"] == cohort) & (frame[q_col] < 0.05)]
        return part.to_dict(orient="records")

    summary = {
        "treatment": treatment_report,
        "metabric_survival_hits": _hits(survival, "metabric"),
        "metabric_enrichment_hits": _hits(enrich, "metabric"),
        "metabric_program_hits": _hits(
            programs[(programs["kind"] == "program")] if not programs.empty else programs,
            "metabric",
        ),
        "scanb_program_hits": _hits(
            programs[(programs["kind"] == "program")] if not programs.empty else programs,
            "gse96058",
        ),
    }
    (out / "biology_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
