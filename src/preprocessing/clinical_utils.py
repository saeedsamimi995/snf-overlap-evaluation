"""Shared parsers for subtype labels, survival, and sample identifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.44

PAM50_LEVELS = ("LumA", "LumB", "Her2", "Basal", "Normal", "Claudin-low")

_PAM50_MAP = {
    "LUMA": "LumA",
    "LUMINAL A": "LumA",
    "LUMINALA": "LumA",
    "LUMB": "LumB",
    "LUMINAL B": "LumB",
    "LUMINALB": "LumB",
    "HER2": "Her2",
    "HER2 ENRICHED": "Her2",
    "HER2ENRICHED": "Her2",
    "BASAL": "Basal",
    "BASAL LIKE": "Basal",
    "BASALLIKE": "Basal",
    "NORMAL": "Normal",
    "NORMAL LIKE": "Normal",
    "NORMALLIKE": "Normal",
    "CLAUDIN LOW": "Claudin-low",
    "CLAUDINLOW": "Claudin-low",
}


def clean_gene_symbol(value):
    """Return an uppercase Hugo symbol, or None when the token is not a gene."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower().startswith("unnamed"):
        return None
    if "|" in text:
        text = text.split("|", 1)[0].strip()
    text = text.upper()
    if text in {"NAN", "NA", "NONE", "NULL", "?", "-", "--", "N/A"}:
        return None
    return text


def patient_barcode(sample_id):
    """Map a TCGA sample or file column onto a patient barcode.

    Non-TCGA identifiers are returned unchanged aside from stripping whitespace.
    ``TCGA.3C.AAAU`` and ``TCGA-3C-AAAU-01A`` both become ``TCGA-3C-AAAU``.
    """
    text = str(sample_id).strip().upper().replace(".", "-")
    parts = [part for part in text.split("-") if part]
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return str(sample_id).strip()


def normalize_pam50(value):
    """Map free-text subtype calls onto the labels used in this study."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null", "unknown", "nc"}:
        return np.nan
    lowered = text.lower()
    if lowered.startswith("pam50 subtype:"):
        text = text.split(":", 1)[1].strip()
    key = text.upper().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    if key.startswith("BRCA "):
        key = key[5:]
    key = " ".join(key.split())
    if key in _PAM50_MAP:
        return _PAM50_MAP[key]
    compact = key.replace(" ", "")
    return _PAM50_MAP.get(compact, np.nan)


def parse_os_event(value):
    """Encode death as 1 and alive/censored as 0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, (float, np.floating)) and not isinstance(value, bool):
        if np.isnan(value):
            return np.nan
        if float(value) in (0.0, 1.0):
            return int(value)
    text = str(value).strip().upper()
    if text in {"", "NAN", "NA", "NONE", "NULL"}:
        return np.nan
    if text in {"1", "1.0", "TRUE", "DEAD", "DECEASED"}:
        return 1
    if text in {"0", "0.0", "FALSE", "ALIVE", "LIVING"}:
        return 0
    if ":" in text:
        head, tail = text.split(":", 1)
        head = head.strip()
        tail = tail.strip()
        if head in {"0", "1"}:
            return int(head)
        if any(token in tail for token in ("DECEAS", "DEAD")):
            return 1
        if any(token in tail for token in ("LIV", "ALIVE")):
            return 0
    if any(token in text for token in ("DECEAS", "DEAD")):
        return 1
    if any(token in text for token in ("LIV", "ALIVE")):
        return 0
    return np.nan


def to_days(values, unit):
    """Convert a survival-time column into days."""
    numeric = pd.to_numeric(values, errors="coerce")
    if unit == "days":
        return numeric.astype(float)
    if unit == "months":
        return numeric.astype(float) * DAYS_PER_MONTH
    raise ValueError(f"Unsupported survival time unit: {unit}")


def empty_clinical(sample_ids):
    """Clinical table indexed by sample id, with the study columns present."""
    frame = pd.DataFrame(index=pd.Index(list(sample_ids), name="sample_id"))
    for column in (
        "patient_id",
        "PAM50",
        "OS_time",
        "OS_event",
        "age",
        "er_status",
        "her2_status",
        "pgr_status",
        "ki67_status",
        "intclust",
        "rfs_time",
        "rfs_event",
    ):
        frame[column] = np.nan
    return frame


def clinical_label_summary(clinical):
    """Counts used to confirm that subtype and survival labels were attached."""
    pam = clinical["PAM50"] if "PAM50" in clinical.columns else pd.Series(dtype=object)
    known = pam.notna().sum()
    counts = pam.dropna().astype(str).value_counts().to_dict()
    os_known = 0
    if "OS_time" in clinical.columns and "OS_event" in clinical.columns:
        os_known = int((clinical["OS_time"].notna() & clinical["OS_event"].notna()).sum())
    return {
        "n_samples": int(len(clinical)),
        "n_pam50": int(known),
        "pam50_counts": counts,
        "n_survival": os_known,
    }
