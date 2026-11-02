"""Load SCAN-B, TCGA-BRCA, and METABRIC expression with clinical sidecars.

METABRIC is read from the cBioPortal files already stored for the multimodal
project. The 657 MB matrix is not copied. This module writes only the clinical
table and, later, the gene-locked matrix produced by harmonization.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.clinical_utils import (
    clean_gene_symbol,
    clinical_label_summary,
    empty_clinical,
    normalize_pam50,
    parse_os_event,
    patient_barcode,
    to_days,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANCAN_PATIENT_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
    "public/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
)


def multimodal_root():
    """Directory of the sibling multimodal project, when it is present."""
    candidates = [
        Path(r"D:\Projects\multimodal-breast-cancer-subtyping"),
        PROJECT_ROOT.parents[1] / "multimodal-breast-cancer-subtyping",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def resolve_path(path):
    """Resolve a config path against the project root and the multimodal project."""
    raw = Path(str(path))
    if raw.is_file():
        return raw
    rooted = PROJECT_ROOT / raw
    if rooted.is_file():
        return rooted
    multimodal = multimodal_root() / raw
    if multimodal.is_file():
        return multimodal
    name = raw.name
    for folder in (
        multimodal_root() / "data",
        multimodal_root() / "data" / "metabric",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "tcga",
        PROJECT_ROOT / "data" / "metabric",
    ):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return rooted


def _first_column(frame, candidates):
    lookup = {str(column).upper(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def expression_sample_ids(path, kind, drop_replicates=False):
    """Read only the header and return tumor sample identifiers."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv", ".cct"} or kind == "metabric" else ","
    header = pd.read_csv(path, sep=sep, nrows=0)
    drop = {"hugo_symbol", "entrez_gene_id", "attrib_name"}
    sample_ids = []
    for column in header.columns:
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed") or name.lower() in drop:
            continue
        if drop_replicates and name.lower().endswith("repl"):
            continue
        sample_ids.append(name)
    return sample_ids


def _collapse_genes(frame):
    """Average duplicate Hugo symbols and drop rows that are not genes."""
    symbols = [clean_gene_symbol(gene) for gene in frame.index]
    frame = frame.copy()
    frame.index = symbols
    frame = frame[frame.index.notna()]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.groupby(level=0).mean()
    return frame.astype(np.float32)


def load_expression(path, kind, sample_ids=None):
    """Load a genes-by-samples matrix for one cohort.

    ``sample_ids`` limits the columns that are parsed. Headers are still scanned,
    but the value matrix stays restricted to the requested tumors.
    """
    path = Path(path)
    print(f"  Loading {kind} expression from {path}")
    if kind == "metabric":
        header = pd.read_csv(path, sep="\t", nrows=0)
        gene_col = "Hugo_Symbol" if "Hugo_Symbol" in header.columns else header.columns[0]
        use = [gene_col]
        if sample_ids is None:
            use.extend(
                column
                for column in header.columns
                if str(column).lower() not in {"hugo_symbol", "entrez_gene_id"}
            )
        else:
            wanted = set(sample_ids)
            use.extend(column for column in header.columns if str(column) in wanted)
        raw = pd.read_csv(path, sep="\t", usecols=use, low_memory=False)
        raw = raw.set_index(gene_col)
        matrix = _collapse_genes(raw)
    else:
        sep = "\t" if path.suffix.lower() in {".txt", ".tsv", ".cct"} else ","
        header = pd.read_csv(path, sep=sep, nrows=0)
        index_name = header.columns[0]
        use = [index_name]
        if sample_ids is None:
            use.extend(column for column in header.columns[1:])
        else:
            wanted = set(map(str, sample_ids))
            use.extend(column for column in header.columns[1:] if str(column) in wanted)
        raw = pd.read_csv(path, sep=sep, index_col=0, usecols=use, low_memory=False)
        # A prepared file may still carry a label row. It is not a gene.
        label_rows = [
            idx for idx in raw.index if str(idx).strip().upper() in {"PAM50", "COMMUNITY", "LABEL", "TARGET"}
        ]
        if label_rows:
            raw = raw.drop(index=label_rows)
        matrix = _collapse_genes(raw)
    if sample_ids is not None:
        ordered = [sample for sample in sample_ids if sample in matrix.columns]
        matrix = matrix.loc[:, ordered]
    print(f"    {matrix.shape[0]:,} genes x {matrix.shape[1]:,} samples")
    return matrix


def _limit_samples(sample_ids, max_samples):
    ordered = sorted(map(str, sample_ids))
    if max_samples is None or max_samples >= len(ordered):
        return ordered
    return ordered[: int(max_samples)]


def ensure_tcga_pam50_file(path):
    """Download PanCancer Atlas patient clinical data when SUBTYPE is missing locally.

    The sample-level PanCancer file does not contain SUBTYPE. cBioPortal stores
    that call on the patient table (``BRCA_LumA``, ``BRCA_Basal``, and so on).
    """
    path = Path(path)
    if path.is_file() and path.stat().st_size > 10000:
        header = pd.read_csv(path, sep="\t", comment="#", nrows=0)
        if any(str(column).upper() == "SUBTYPE" for column in header.columns):
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading TCGA PanCancer subtype table to {path}")
    request = urllib.request.Request(PANCAN_PATIENT_URL, headers={"User-Agent": "BIGCLAM-q1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())
    return path


def write_gse_clinical(expression_file, clinical_file, output_file, os_time_unit="days", max_samples=None):
    """Attach SCAN-B PAM50 and survival. Replicate columns are dropped."""
    expression_file = resolve_path(expression_file)
    clinical_file = resolve_path(clinical_file)
    sample_ids = _limit_samples(
        expression_sample_ids(expression_file, "gse96058", drop_replicates=True),
        max_samples,
    )
    wide = pd.read_csv(clinical_file, index_col=0, low_memory=False)
    wide = wide.copy()
    wide.columns = wide.columns.astype(str)
    keep = [sample for sample in sample_ids if sample in wide.columns]
    missing = len(sample_ids) - len(keep)
    if missing:
        print(f"  [GSE96058] {missing} expression samples have no clinical column")
    table = wide.loc[:, keep].T
    table.index.name = "sample_id"
    clinical = empty_clinical(keep)
    clinical["patient_id"] = clinical.index.astype(str)
    if "PAM50" in table.columns:
        clinical["PAM50"] = [normalize_pam50(value) for value in table["PAM50"]]
    if "OS_time" in table.columns:
        clinical["OS_time"] = to_days(table["OS_time"], os_time_unit).to_numpy()
    if "OS_event" in table.columns:
        clinical["OS_event"] = [parse_os_event(value) for value in table["OS_event"]]
    if "age" in table.columns:
        clinical["age"] = pd.to_numeric(table["age"], errors="coerce").to_numpy()
    for source, target in (
        ("er_status", "er_status"),
        ("her2_status", "her2_status"),
        ("pgr_status", "pgr_status"),
        ("ki67_status", "ki67_status"),
    ):
        if source in table.columns:
            clinical[target] = table[source].astype(str).to_numpy()
    _write_clinical(clinical, output_file, "GSE96058")
    return clinical, sample_ids


def write_tcga_clinical(
    clinical_file,
    expression_file,
    output_file,
    pam50_file,
    os_time_unit="months",
    max_samples=None,
):
    """Attach Firehose survival and PanCancer Atlas PAM50 calls."""
    expression_file = resolve_path(expression_file)
    clinical_file = resolve_path(clinical_file)
    pam50_file = ensure_tcga_pam50_file(resolve_path(pam50_file))
    sample_ids = _limit_samples(
        expression_sample_ids(expression_file, "tcga", drop_replicates=False),
        max_samples,
    )
    clinical_raw = pd.read_csv(clinical_file, sep="\t", low_memory=False)
    patient_col = _first_column(clinical_raw, ["Patient ID", "PATIENT_ID", "Sample ID", "SAMPLE_ID"])
    if patient_col is None:
        raise ValueError(f"No patient id column in {clinical_file}")
    clinical_raw = clinical_raw.copy()
    clinical_raw["_patient"] = clinical_raw[patient_col].map(patient_barcode)
    clinical_raw = clinical_raw.drop_duplicates("_patient", keep="first").set_index("_patient")

    os_time_col = _first_column(
        clinical_raw, ["Overall Survival (Months)", "OS_MONTHS", "OS_TIME"]
    )
    os_event_col = _first_column(
        clinical_raw, ["Overall Survival Status", "OS_STATUS", "VITAL_STATUS"]
    )
    age_col = _first_column(clinical_raw, ["Diagnosis Age", "AGE", "AGE_AT_DIAGNOSIS"])
    er_col = _first_column(clinical_raw, ["ER Status By IHC", "ER_STATUS", "ER_IHC"])
    her2_col = _first_column(clinical_raw, ["IHC-HER2", "HER2_STATUS", "HER2_SNP6"])

    pam = pd.read_csv(pam50_file, sep="\t", comment="#", low_memory=False)
    pam_id = _first_column(pam, ["PATIENT_ID", "Patient Identifier", "SAMPLE_ID"])
    pam_col = _first_column(pam, ["SUBTYPE", "PAM50", "PAM50_SUBTYPE"])
    if pam_id is None or pam_col is None:
        raise ValueError(f"PanCancer file {pam50_file} has no PATIENT_ID/SUBTYPE columns")
    pam = pam.copy()
    pam["_patient"] = pam[pam_id].map(patient_barcode)
    pam = pam.drop_duplicates("_patient", keep="first").set_index("_patient")

    rows = []
    for sample_id in sample_ids:
        barcode = patient_barcode(sample_id)
        row = {"sample_id": sample_id, "patient_id": barcode}
        if barcode in clinical_raw.index:
            source = clinical_raw.loc[barcode]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            if os_time_col is not None:
                row["OS_time"] = to_days(pd.Series([source[os_time_col]]), os_time_unit).iloc[0]
            if os_event_col is not None:
                row["OS_event"] = parse_os_event(source[os_event_col])
            if age_col is not None:
                row["age"] = pd.to_numeric(source[age_col], errors="coerce")
            if er_col is not None:
                row["er_status"] = source[er_col]
            if her2_col is not None:
                row["her2_status"] = source[her2_col]
        if barcode in pam.index:
            subtype = pam.loc[barcode, pam_col]
            if isinstance(subtype, pd.Series):
                subtype = subtype.iloc[0]
            row["PAM50"] = normalize_pam50(subtype)
        rows.append(row)
    clinical = _clinical_from_rows(rows, sample_ids)
    _write_clinical(clinical, output_file, "TCGA-BRCA")
    return clinical, sample_ids


def write_metabric_clinical(
    expression_file,
    patient_file,
    sample_file,
    output_file,
    os_time_unit="months",
    max_samples=None,
):
    """Write METABRIC subtype, IntClust, and survival without copying the matrix."""
    expression_file = resolve_path(expression_file)
    patient_file = resolve_path(patient_file)
    sample_ids = _limit_samples(
        expression_sample_ids(expression_file, "metabric", drop_replicates=False),
        max_samples,
    )
    patient = pd.read_csv(patient_file, sep="\t", comment="#", low_memory=False)
    patient_id_col = _first_column(patient, ["PATIENT_ID"])
    if patient_id_col is None:
        raise ValueError(f"No PATIENT_ID column in {patient_file}")
    patient = patient.copy()
    patient["_patient"] = patient[patient_id_col].astype(str).str.strip()
    patient = patient.drop_duplicates("_patient", keep="first").set_index("_patient")

    sample_to_patient = {sample_id: sample_id for sample_id in sample_ids}
    sample_path = resolve_path(sample_file) if sample_file else None
    if sample_path is not None and sample_path.is_file():
        sample = pd.read_csv(sample_path, sep="\t", comment="#", low_memory=False)
        sid_col = _first_column(sample, ["SAMPLE_ID"])
        pid_col = _first_column(sample, ["PATIENT_ID"])
        if sid_col and pid_col:
            sample_to_patient.update(
                {
                    str(sid).strip(): str(pid).strip()
                    for sid, pid in zip(sample[sid_col], sample[pid_col])
                }
            )

    def _col(name_options):
        return _first_column(patient, name_options)

    os_time_col = _col(["OS_MONTHS", "OS_TIME", "OVERALL_SURVIVAL_MONTHS"])
    os_event_col = _col(["OS_STATUS", "OVERALL_SURVIVAL_STATUS", "VITAL_STATUS"])
    age_col = _col(["AGE_AT_DIAGNOSIS", "AGE", "DIAGNOSIS_AGE"])
    pam_col = _col(["CLAUDIN_SUBTYPE", "PAM50", "SUBTYPE"])
    intclust_col = _col(["INTCLUST"])
    er_col = _col(["ER_IHC", "ER_STATUS"])
    her2_col = _col(["HER2_SNP6", "HER2_STATUS"])
    rfs_time_col = _col(["RFS_MONTHS"])
    rfs_event_col = _col(["RFS_STATUS"])

    rows = []
    for sample_id in sample_ids:
        patient_id = sample_to_patient.get(sample_id, sample_id)
        if patient_id not in patient.index and sample_id in patient.index:
            patient_id = sample_id
        row = {"sample_id": sample_id, "patient_id": patient_id}
        if patient_id in patient.index:
            source = patient.loc[patient_id]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            if pam_col is not None:
                row["PAM50"] = normalize_pam50(source[pam_col])
            if intclust_col is not None and pd.notna(source[intclust_col]):
                row["intclust"] = str(source[intclust_col]).strip()
            if os_time_col is not None:
                row["OS_time"] = to_days(pd.Series([source[os_time_col]]), os_time_unit).iloc[0]
            if os_event_col is not None:
                row["OS_event"] = parse_os_event(source[os_event_col])
            if age_col is not None:
                row["age"] = pd.to_numeric(source[age_col], errors="coerce")
            if er_col is not None:
                row["er_status"] = source[er_col]
            if her2_col is not None:
                row["her2_status"] = source[her2_col]
            if rfs_time_col is not None:
                row["rfs_time"] = to_days(pd.Series([source[rfs_time_col]]), os_time_unit).iloc[0]
            if rfs_event_col is not None:
                row["rfs_event"] = parse_os_event(source[rfs_event_col])
        rows.append(row)
    clinical = _clinical_from_rows(rows, sample_ids)
    _write_clinical(clinical, output_file, "METABRIC")
    return clinical, sample_ids


def _clinical_from_rows(rows, sample_ids):
    """Build the sidecar without forcing subtype strings into a float column."""
    clinical = empty_clinical(sample_ids)
    filled = pd.DataFrame(rows).set_index("sample_id")
    for column in clinical.columns:
        if column not in filled.columns:
            continue
        clinical[column] = filled[column].reindex(clinical.index).to_numpy()
    return clinical


def _write_clinical(clinical, output_file, label):
    output_file = PROJECT_ROOT / output_file if not Path(output_file).is_absolute() else Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(output_file)
    summary = clinical_label_summary(clinical)
    print(
        f"  [{label}] clinical -> {output_file} | "
        f"n={summary['n_samples']} PAM50={summary['n_pam50']} "
        f"survival={summary['n_survival']} {summary['pam50_counts']}"
    )
    if summary["n_pam50"] == 0:
        print(f"  [{label}] warning: no PAM50 labels were attached")
    return output_file


def load_q1_cohorts(q1_config, max_samples=None):
    """Load expression and clinical tables for every cohort in the study config."""
    loaded = {}
    gse_cfg = q1_config["gse96058"]
    gse_clinical, gse_samples = write_gse_clinical(
        gse_cfg["expression_file"],
        gse_cfg["clinical_file"],
        gse_cfg.get("clinical_output", "data/prepared/gse96058_clinical.csv"),
        os_time_unit=gse_cfg.get("os_time_unit", "days"),
        max_samples=max_samples,
    )
    gse_expr = load_expression(
        resolve_path(gse_cfg["expression_file"]), "gse96058", sample_ids=gse_samples
    )
    loaded["gse96058"] = {"expression": gse_expr, "clinical": gse_clinical}

    tcga_cfg = q1_config["tcga"]
    tcga_clinical, tcga_samples = write_tcga_clinical(
        tcga_cfg["clinical_file"],
        tcga_cfg["expression_file"],
        tcga_cfg.get("clinical_output", "data/prepared/tcga_clinical.csv"),
        pam50_file=tcga_cfg.get("pam50_file", "data/tcga/brca_pancan_clinical_patient.txt"),
        os_time_unit=tcga_cfg.get("os_time_unit", "months"),
        max_samples=max_samples,
    )
    tcga_expr = load_expression(
        resolve_path(tcga_cfg["expression_file"]), "tcga", sample_ids=tcga_samples
    )
    loaded["tcga"] = {"expression": tcga_expr, "clinical": tcga_clinical}

    meta_cfg = q1_config["metabric"]
    meta_out = Path(meta_cfg.get("output_dir", "data/metabric"))
    meta_clinical, meta_samples = write_metabric_clinical(
        meta_cfg["expression_file"],
        meta_cfg["patient_clinical"],
        meta_cfg.get("sample_clinical"),
        meta_out / "metabric_clinical.csv",
        os_time_unit=meta_cfg.get("os_time_unit", "months"),
        max_samples=max_samples,
    )
    meta_expr = load_expression(
        resolve_path(meta_cfg["expression_file"]), "metabric", sample_ids=meta_samples
    )
    loaded["metabric"] = {"expression": meta_expr, "clinical": meta_clinical}
    return loaded
