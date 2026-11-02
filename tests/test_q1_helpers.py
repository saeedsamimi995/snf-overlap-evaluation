"""Unit checks for label parsing, overlap, gene locking, and the patient graph."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.q1_clustering import membership_summary, symmetric_knn_graph
from src.preprocessing.clinical_utils import (
    normalize_pam50,
    parse_os_event,
    patient_barcode,
    to_days,
)
from src.preprocessing.harmonize import harmonize_loaded, shared_gene_symbols


def test_pam50_and_survival_parsers():
    assert normalize_pam50("pam50 subtype: LumA") == "LumA"
    assert normalize_pam50("BRCA_Basal") == "Basal"
    assert normalize_pam50("claudin-low") == "Claudin-low"
    assert normalize_pam50("Her2-enriched") == "Her2"
    assert normalize_pam50("NC") is np.nan or (isinstance(normalize_pam50("NC"), float) and np.isnan(normalize_pam50("NC")))
    assert parse_os_event("0:LIVING") == 0
    assert parse_os_event("1:DECEASED") == 1
    assert parse_os_event("LIVING") == 0
    assert patient_barcode("TCGA.3C.AAAU") == "TCGA-3C-AAAU"
    assert patient_barcode("TCGA-3C-AAAU-01") == "TCGA-3C-AAAU"
    days = to_days(pd.Series([2.0]), "months")
    assert abs(float(days.iloc[0]) - 60.88) < 1e-6


def test_mixed_membership_ratio():
    membership = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.5, 0.4, 0.0],
            [0.2, 0.2, 0.1],
        ]
    )
    hard, mixed, ratio = membership_summary(membership, ratio=0.5)
    assert list(hard[:2]) == [0, 0]
    assert hard[2] in (0, 1)  # the two leading affiliations are tied
    assert list(mixed) == [False, True, True]
    assert ratio[0] < 0.5
    assert ratio[1] >= 0.5


def test_choose_overlap_and_peaked_fit():
    from src.analysis.q1_clustering import choose_overlap_ratio, fit_cohort

    ratios = np.array([0.1] * 70 + [0.8] * 20 + [0.95] * 10)
    locked, fraction = choose_overlap_ratio(ratios)
    assert 0.1 <= fraction <= 0.4
    assert locked >= 0.5

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 20))
    X = np.vstack([centers[i] + 0.05 * rng.normal(size=(40, 20)) for i in range(4)])
    fit = fit_cohort(
        X,
        num_communities=4,
        knn_k=10,
        iterations=80,
        restarts=2,
        lr=0.05,
        seed=0,
        overlap_ratio=0.75,
        l1=0.02,
    )
    assert fit["peak_ratio"] < 0.85
    # Clean planted clusters can be fully single-community; that is a good fit.
    assert fit["mixed_fraction"] < 1.0
    assert len(set(fit["community"])) == 4


def test_knn_graph_is_symmetric_and_connected():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(25, 8))
    adjacency = symmetric_knn_graph(features, k=3)
    assert adjacency.shape == (25, 25)
    assert np.allclose(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 0)
    assert adjacency.sum() > 0


def test_gene_lock_uses_discovery_variance_before_zscore():
    discovery = pd.DataFrame(
        {
            "s1": [0.0, 5.0, 1.0],
            "s2": [0.0, 0.0, 1.0],
            "s3": [0.0, 5.0, 1.0],
        },
        index=["FLAT", "VARIABLE", "MID"],
    )
    other = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 0.0, 2.0],
        },
        index=["FLAT", "VARIABLE", "MID"],
    )
    loaded = {
        "gse96058": {"expression": discovery, "clinical": pd.DataFrame(index=["s1", "s2", "s3"])},
        "tcga": {"expression": other, "clinical": pd.DataFrame(index=["a", "b"])},
    }
    assert shared_gene_symbols(
        {"gse96058": discovery, "tcga": other}
    ) == ["FLAT", "MID", "VARIABLE"]
    harmonized = harmonize_loaded(loaded, n_genes=2, discovery="gse96058", min_shared_genes=2)
    assert harmonized["genes"][0] == "VARIABLE"
    assert len(harmonized["genes"]) == 2
    assert harmonized["cohorts"]["tcga"]["X"].shape == (2, 2)
