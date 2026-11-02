"""
Prepare clinical sidecars for TCGA-BRCA, GSE96058, and METABRIC.

Expression matrices stay where they already are. This script does not rewrite a
second full matrix, and it does not fill PAM50 with the placeholder "Unknown".
The Q1 study locks genes afterwards in ``run_q1_study.py``.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _clinical_output(output_file, default_name):
    """Keep legacy output paths from being treated as expression matrices."""
    if output_file is None:
        return default_name
    name = str(output_file)
    if name.endswith("data_target_added.csv"):
        return default_name
    return name


def prepare_tcga_brca_data(clinical_file, expression_file, output_file, pam50_file=None):
    """Write TCGA survival and PanCancer Atlas PAM50 labels.

    Parameters keep the previous signature. ``output_file`` is the clinical
    sidecar, not a second copy of the expression matrix.
    """
    from src.preprocessing.cohorts import write_tcga_clinical

    clinical, _samples = write_tcga_clinical(
        clinical_file,
        expression_file,
        _clinical_output(output_file, "data/prepared/tcga_clinical.csv"),
        pam50_file=pam50_file or "data/tcga/brca_pancan_clinical_patient.txt",
    )
    return clinical


def prepare_gse96058_data(expression_file, output_file, clinical_file=None):
    """Write SCAN-B PAM50 and survival. Replicate columns are dropped."""
    from src.preprocessing.cohorts import write_gse_clinical

    if not clinical_file:
        raise FileNotFoundError(
            "GSE96058 clinical file is required. The expression matrix does not "
            "contain PAM50 labels."
        )
    clinical, _samples = write_gse_clinical(
        expression_file,
        clinical_file,
        _clinical_output(output_file, "data/prepared/gse96058_clinical.csv"),
    )
    return clinical


def prepare_metabric(
    expression_file,
    patient_file,
    sample_file,
    output_file="data/metabric/metabric_clinical.csv",
):
    """Write METABRIC Claudin/PAM50 subtype, IntClust, and survival."""
    from src.preprocessing.cohorts import write_metabric_clinical

    clinical, _samples = write_metabric_clinical(
        expression_file,
        patient_file,
        sample_file,
        output_file,
    )
    return clinical


if __name__ == "__main__":
    import argparse

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

    parser = argparse.ArgumentParser(
        description="Prepare TCGA, GSE96058, and METABRIC clinical sidecars"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--tcga_clinical", type=str, default=None)
    parser.add_argument("--tcga_expression", type=str, default=None)
    parser.add_argument("--tcga_output", type=str, default=None)
    parser.add_argument("--tcga_pam50", type=str, default=None)
    parser.add_argument("--gse96058_expression", type=str, default=None)
    parser.add_argument("--gse96058_clinical", type=str, default=None)
    parser.add_argument("--gse96058_output", type=str, default=None)
    parser.add_argument("--metabric_expression", type=str, default=None)
    parser.add_argument("--metabric_patient", type=str, default=None)
    parser.add_argument("--metabric_sample", type=str, default=None)
    parser.add_argument("--metabric_output", type=str, default=None)
    parser.add_argument(
        "--dataset",
        choices=["tcga", "gse96058", "metabric", "both", "all"],
        default="all",
    )
    args = parser.parse_args()

    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        print(f"Loaded configuration from: {args.config}")

    prep = config.get("data_preparation", {})
    q1 = config.get("q1", {})
    tcga_cfg = {**prep.get("tcga", {}), **q1.get("tcga", {})}
    gse_cfg = {**prep.get("gse96058", {}), **q1.get("gse96058", {})}
    meta_cfg = q1.get("metabric", {})

    tcga_clinical = args.tcga_clinical or tcga_cfg.get("clinical_file", "data/brca_tcga_clinical_data.tsv")
    tcga_expression = args.tcga_expression or tcga_cfg.get(
        "expression_file",
        "data/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct",
    )
    tcga_output = args.tcga_output or tcga_cfg.get("clinical_output", "data/prepared/tcga_clinical.csv")
    tcga_pam50 = args.tcga_pam50 or tcga_cfg.get("pam50_file", "data/tcga/brca_pancan_clinical_patient.txt")

    gse_expression = args.gse96058_expression or gse_cfg.get(
        "expression_file",
        "data/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv",
    )
    gse_clinical = args.gse96058_clinical or gse_cfg.get("clinical_file")
    gse_output = args.gse96058_output or gse_cfg.get("clinical_output", "data/prepared/gse96058_clinical.csv")

    meta_expression = args.metabric_expression or meta_cfg.get("expression_file")
    meta_patient = args.metabric_patient or meta_cfg.get("patient_clinical")
    meta_sample = args.metabric_sample or meta_cfg.get("sample_clinical")
    meta_output = args.metabric_output or str(Path(meta_cfg.get("output_dir", "data/metabric")) / "metabric_clinical.csv")

    if args.dataset in {"tcga", "both", "all"}:
        prepare_tcga_brca_data(tcga_clinical, tcga_expression, tcga_output, pam50_file=tcga_pam50)
    if args.dataset in {"gse96058", "both", "all"}:
        prepare_gse96058_data(gse_expression, gse_output, gse_clinical)
    if args.dataset in {"metabric", "all"}:
        prepare_metabric(meta_expression, meta_patient, meta_sample, meta_output)

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
