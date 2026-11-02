"""Build revised manuscript.docx (title/abstract/methods/fig/refs fixes)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "paper_figures" / "includegraphics"
OUT = ROOT / "manuscript.docx"


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text):
    para = doc.add_paragraph(text)
    for run in para.runs:
        run.font.size = Pt(11)
    return para


def add_table(doc, caption, headers, rows):
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(10)
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = header
        if j == 0:
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_pr.append(OxmlElement("w:noWrap"))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(value)
            if j == 0:
                tc_pr = cell._tc.get_or_add_tcPr()
                tc_pr.append(OxmlElement("w:noWrap"))
    doc.add_paragraph("")


def add_fig(doc, name, caption, width=6.0):
    """Insert image then caption; keep them on the same page when possible."""
    path = FIG / name
    if not path.exists():
        p(doc, f"[Missing figure: {name}]")
        return
    pic_para = doc.add_paragraph()
    pic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_para.paragraph_format.keep_with_next = True
    run = pic_para.add_run()
    run.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in cap.runs:
        r.italic = True
        r.font.size = Pt(10)


def main():
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run(
        "Overlapping community detection on fused patient-similarity graphs: "
        "a locked multi-cohort evaluation with a pre-specified lead policy"
    )
    r.bold = True
    r.font.size = Pt(14)

    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors.add_run("S. Samimi* and M. Noorani\n").bold = True
    authors.add_run(
        "Department of Electrical and Computer Engineering, "
        "K.N. Toosi University of Technology, Tehran, Iran\n"
        "*Corresponding author: saeedsamimi995@gmail.com"
    )

    h(doc, "Abstract")
    p(
        doc,
        "Overlapping community detection is often added to a patient-similarity graph and then judged "
        "by whichever metric looks strongest. We lock a lead rule and a three-cohort "
        "discovery, validation, and external design, then add a pre-stated treatment-response "
        "external cohort (GSE25066, n=508). Discovery is GSE96058/SCAN-B (n=3,273); validation "
        "is TCGA-BRCA (n=1,093); external is METABRIC (n=1,980). Multi-view SNF supplies hard "
        "subtypes. Soft engines are NOCD-X, NOCD-G, SLPA, CESNA-lite, and classical BIGCLAM, "
        "which is an a priori baseline and cannot be the lead. Soft residual tests are "
        "METABRIC-primary. The treatment-response test is GSE25066. "
        "Hard SNF separates METABRIC OS after age and PAM50, with p ≈ 1.7×10⁻⁴ and BH q ≈ 8.3×10⁻⁴ "
        "within five hard methods. "
        "NOCD-X is the only lead-eligible engine that passes the quality filters, so it is the policy "
        "lead. A pre-stated treatment-response test in GSE25066 (n = 508) was null in both the LumA "
        "and hard-cluster strata under locked settings; no treatment-resistance claim is made. "
        "BIGCLAM has the stronger LumA odds ratio (≈ 3.0 versus ≈ 1.8) and a nominal residual "
        "p ≈ 0.036, but BH q ≈ 0.091. Further checks are null: SLPA's peak ratio stays 1 when "
        "mean degree falls from 34 to 6; boundary patients do not separate survival (smallest "
        "METABRIC q ≈ 0.23); BIGCLAM's L1 penalty does not move the LumA odds ratio or the residual p. "
        "A mixed-membership simulation and an IntClust benchmark both give stronger point estimates "
        "to BIGCLAM than to NOCD-X, so neither is used to defend the lead. The lead stays the "
        "pre-specified rule.",
    )

    h(doc, "1. Introduction")
    p(
        doc,
        "This paper is a locked evaluation, not a claim that one engine is the best clusterer. "
        "Classical BIGCLAM is reserved a priori as a comparison baseline and cannot be the lead, "
        "even if a biological odds ratio favours it. The checks that would have raised the claim "
        "are reported in full: SLPA density, boundary survival, the L1 ablation, an external "
        "pathological complete response test on GSE25066, and a ground-truth simulation plus "
        "IntClust labels.",
    )

    h(doc, "2. Data")
    p(
        doc,
        "Discovery: GSE96058/SCAN-B (n=3,273). Validation: TCGA-BRCA (n=1,093). External: METABRIC "
        "(n=1,980; primary residual soft tests). Treatment-response external: GSE25066 (Hatzis et al. 2011 [29]), Affymetrix "
        "HG-U133A (n=508; 99 pCR, 389 RD, 20 unlabeled). 1,400 of the locked 2,000 genes are on "
        "the array (highest-variance probe per symbol; see Methods). Locked settings are applied without retuning. No treatment columns were added "
        "to SCAN-B, TCGA, or METABRIC.",
    )

    h(doc, "3. Methods")
    add_fig(
        doc,
        "fig_pipeline.png",
        "Figure 1. Dual-output workflow with sparsified SNF kNN graph for overlapping engines.",
    )
    h(doc, "3.1 Harmonization and community-count lock", 2)
    p(
        doc,
        "Intersect gene symbols across cohorts. On discovery only, rank intersecting genes by variance "
        "and keep the top 2,000; apply that locked list to TCGA and METABRIC; z-score gene-wise within "
        "each cohort. Community count K is locked on discovery by bootstrap stability: for each "
        "candidate K∈{4,…,8}, labels are refit on bootstrap samples and mean pairwise adjusted Rand "
        "index (ARI) is recorded; we select K=5 (mean ARI ≈ 0.85), the maximum in that grid "
        "(results/q1/stability.csv).",
    )
    h(doc, "3.2 Multi-view SNF and sparsification", 2)
    p(
        doc,
        "Classical SNF message-passing fusion (Wang et al. iterative cross-diffusion; local kNN "
        "affinities, T=15 iterations, neighbourhood size k=15) on two views: Euclidean and "
        "Pearson-correlation distances converted to kNN affinities. Hard labels use scikit-learn "
        "spectral clustering on the fused affinity. The affinity argument is precomputed, and "
        "the assign-labels argument is kmeans (assign_labels=\"kmeans\"). K is 5 and the random "
        "seed is 42. For soft engines, sparsify the fused affinity to a symmetric kNN "
        "adjacency (k=15) and reconnect components by bridging with the strongest fused similarities. "
        "See SUPPLEMENTARY_METHODS.md for full hyperparameters.",
    )
    h(doc, "3.3 Overlapping engines and mixed ratios", 2)
    p(
        doc,
        "NOCD-X/G, SLPA, CESNA-lite, BIGCLAM (baseline). Mixed if second/first affiliation ≥ r. On "
        "discovery, r is chosen per engine from {0.50,…,0.95} to place mixed fraction nearest the "
        "centre of [0.10, 0.40]; locked values: NOCD-X 0.60, NOCD-G 0.90, SLPA 0.80, CESNA-lite 0.70, "
        "BIGCLAM 0.50. Differences reflect F peakiness, not arbitrary fixed cutoffs.",
    )
    h(doc, "3.4 Lead policy and evaluation", 2)
    p(
        doc,
        "BIGCLAM = baseline_comparison (not lead-eligible). Quality filters on METABRIC: mixed ∈ "
        "[0.10,0.40], peak < 0.85, LumA OR > 1. Lead = sole lead-eligible engine in the quality-pass "
        "pool. The lead designation reflects this pre-specified policy of reserving classical baselines "
        "for comparison; it is not a claim that NOCD-X outperforms BIGCLAM on every metric. "
        "Report uncorrected p and BH q for the primary families. The hard-method BH family is five methods. The METABRIC "
        "soft residual BH family is five engines. The GSE25066 BH family is two Fisher tests "
        "per engine. Soft bake-off is METABRIC-primary; GSE/TCGA soft residuals are summarised "
        "for completeness. ΔC: point estimate + 200-resample patient bootstrap 95% CI. "
        "GSE25066 keeps the locked 2,000-gene list intersected with GPL96 (highest-variance probe "
        "per symbol, outcome-blind), within-cohort z-scores, k=15, T=15, K=5, and mixed ratios "
        "0.60 (NOCD-X) and 0.50 (BIGCLAM). The pre-stated primary family is two Fisher tests of "
        "pCR, inside LumA and inside the hard cluster with the most mixed patients. BH is inside "
        "that pair. BIGCLAM is a separate family and cannot change the lead.",
    )

    h(doc, "4. Results")
    h(doc, "4.1 Hard SNF prognosis", 2)
    p(
        doc,
        "METABRIC SNF age+PAM50 Cox p ≈ 1.7×10⁻⁴ (BH q ≈ 8.3×10⁻⁴ within five hard methods). "
        "Single-view spectral cuts are weaker; SNF+k-means is competitive (sensitivity). Nested C: "
        "METABRIC 0.614→0.624 (ΔC≈0.011); TCGA 0.666→0.687 (ΔC≈0.021). Bootstrap 95% CIs for ΔC all "
        "include zero (METABRIC [−0.007, 0.010]; TCGA [−0.031, 0.033]; GSE [−0.009, 0.007]), all within "
        "bootstrap noise, and are not claimed as statistically or clinically decisive.",
    )
    add_fig(
        doc,
        "fig_cox_bars.png",
        "Figure 2. Hard partitions: age+PAM50 Cox strength (−log₁₀ uncorrected p) with analysis n on "
        "each bar. Dashed line: uncorrected p=0.05.",
        width=6.0,
    )
    add_fig(
        doc,
        "fig_hard_ablations.png",
        "Figure 3. METABRIC hard ablations: fused SNF vs single-view spectral cuts (age+PAM50 Cox).",
        width=5.5,
    )
    add_fig(doc, "fig_km_snf_gse.png", "Figure 4a. Hard SNF KM + risk table (GSE96058).", width=5.2)
    add_fig(doc, "fig_km_snf_tcga.png", "Figure 4b. Hard SNF KM + risk table (TCGA).", width=5.2)
    add_fig(doc, "fig_km_snf_metabric.png", "Figure 4c. Hard SNF KM + risk table (METABRIC).", width=5.2)

    h(doc, "4.2 Soft lead decision (METABRIC-primary)", 2)
    p(
        doc,
        "Soft residual bake-off and quality filters are METABRIC-primary. GSE96058 and TCGA "
        "own-hard+mixed residuals are reported for completeness and are mostly null after BH. "
        "The exception, also for completeness, is SLPA on TCGA (p≈0.004, q≈0.022), which still fails quality filters. BIGCLAM passes all quality filters but "
        "is not lead-eligible by the a priori baseline policy. Only NOCD-X is in the eligible "
        "quality-pass pool and is selected. This is a policy choice, not a claim that NOCD-X "
        "dominates BIGCLAM on LumA enrichment or residual p. SLPA has the smallest METABRIC residual "
        "p (0.0055; q≈0.028) but mixed≈69%, peak ratio=1, and LumA depletion: on a dense fused SNF "
        "graph, SLPA label propagation tends toward consensus. BIGCLAM residual p≈0.036 yields BH "
        "q≈0.091 across five soft engines and fails FDR at the 0.05 threshold.",
    )
    add_fig(
        doc,
        "fig_lead_decision.png",
        "Figure 5. METABRIC pass/fail lead decision. BIGCLAM passes the quality filters and stays ineligible for the lead.",
    )
    add_fig(
        doc,
        "fig_soft_bakeoff.png",
        "Figure 6. Soft bake-off on METABRIC. Left: −log₁₀ uncorrected hard-cut Cox p. Middle: "
        "−log₁₀ uncorrected own-hard+mixed residual p, with BH q annotated for the family of five "
        "METABRIC soft engines. Right: LumA odds ratios.",
    )

    h(doc, "4.3 Biology of mixed membership", 2)
    p(
        doc,
        "Mixed tumours are LumA-enriched (not Claudin-low). Programme scores (positive = higher in "
        "mixed): estrogen Δ ≈ +0.56 SD (BIGCLAM, n_mixed=504, p≈6×10⁻⁴²) and +0.36 SD (NOCD-X, "
        "n_mixed=334, p≈3×10⁻⁹), with lower basal/proliferation. These are moderate standardised "
        "shifts made highly significant by large n—a luminal-leaning mixed phenotype, not a large "
        "clinical effect size per se.",
    )
    add_fig(doc, "fig_mixed_enrichment.png", "Figure 7. Mixed×PAM50 odds ratios.", width=5.5)
    add_fig(
        doc,
        "fig_mixed_programs.png",
        "Figure 8. METABRIC programme-score bar chart (mixed minus single). Positive means higher in mixed. Not a survival curve.",
        width=5.5,
    )
    add_fig(doc, "fig_km_mixed_metabric.png", "Figure 9a. METABRIC Kaplan–Meier, mixed versus single (BIGCLAM flag), with numbers at risk.", width=5.2)
    add_fig(
        doc,
        "fig_km_within_luma_metabric.png",
        "Figure 9b. METABRIC Kaplan–Meier within LumA, mixed versus single, with numbers at risk.",
        width=5.2,
    )

    h(doc, "4.4 Closed routes", 2)
    p(
        doc,
        "Three checks that could have supported a stronger claim are null. (1) SLPA median peak "
        "ratio stays 1.0 at k = 5, 10, 15, and 30 (mean degree 6.4 to 34; mixed fractions 0.83, "
        "0.77, 0.69, 0.51). Collapse is not an under-sparsified kNN setting (Figure 10). (2) "
        "Age-adjusted mixed and boundary tests inside hard clusters and inside LumA do not survive "
        "BH; smallest METABRIC q = 0.23. There is no prognostic subgroup inside the hard cut. "
        "(3) BIGCLAM L1 in {0, 0.001, 0.01, 0.1} leaves LumA OR at 3.00 and residual p at 0.56. "
        "The LumA gap versus NOCD-X is not an L1 artifact.",
    )
    add_fig(
        doc,
        "fig_density_sweep.png",
        "Figure 10. METABRIC density sweep. Left: median second/first affiliation. Right: mixed "
        "fraction. SLPA stays at peak ratio 1 across mean degree 6–34.",
    )

    h(doc, "4.5 Treatment-response test (GSE25066)", 2)
    p(
        doc,
        "On GSE25066 (Hatzis et al. 2011 [29]), the pre-stated primary family is two Fisher tests "
        "for NOCD-X. BIGCLAM is a separate family and is not lead-eligible. The soft membership "
        "is not associated with pCR in GSE25066 under locked settings. Rebuilding the discovery "
        "graph on the 1,400 shared genes leaves the hard partition close to the locked one "
        "(ARI 0.89) and the fused affinities highly correlated (Spearman 0.98), so the locked "
        "settings are kept for GSE25066. Soft engines were not refit. Details are in the "
        "Discussion. The LumA stratum has five pCR "
        "events. The hard-cluster stratum has sixteen pCR events. The LumA contrast is event-poor, and "
        "the hard-cluster contrast is better powered and still null. There is no treatment-resistance "
        "claim.",
    )
    add_table(
        doc,
        "Table. GSE25066 pCR, mixed versus single. BH q is within each engine's family of two. "
        "The asterisk marks an odds ratio of 0: the BIGCLAM mixed group inside LumA had zero pCR events. The Fisher p is the reported test.",
        ["Engine", "Family", "Scope", "pCR mixed", "pCR single", "OR", "p", "BH q"],
        [
            ["NOCD-X", "primary", "LumA", "1/35", "4/118", "0.84", "1.00", "1"],
            ["NOCD-X", "primary", "Hard cluster 1", "2/34", "14/134", "0.54", "0.53", "1"],
            ["BIGCLAM", "baseline", "LumA", "0/29", "5/124", "0*", "0.58", "1"],
            ["BIGCLAM", "baseline", "Hard cluster 1", "3/30", "13/138", "1.07", "1.00", "1"],
        ],
    )

    h(doc, "4.6 Checks that do not override the policy", 2)
    p(
        doc,
        "A mixed-membership simulation (n=300, matched mean degree) gives mean correlation with "
        "true soft membership of 0.53 (BIGCLAM), 0.17 (SLPA), and 0.10 (graph NOCD). Mixed-label "
        "ARI is below 0.10 for every engine. IntClust agreement is real and also stronger for "
        "BIGCLAM: IntClust 8 OR 3.24, versus NOCD-X IntClust 10 OR 0.058 and IntClust 7 OR 1.95 "
        "(Figure 11). Inside BIGCLAM hard cluster 0, estrogen Δ = +0.088 SD and HER2 Δ = −0.185; "
        "SCAN-B repeats those two directions inside hard cluster 0 (estrogen Δ = +0.081 SD; "
        "HER2 Δ = −0.514; proliferation Δ = +0.165, the opposite direction; "
        "results/ceiling/program_contrasts.csv). Neither lever is used to change the lead. Doing so "
        "would either break the pre-specified baseline rule or claim a ground-truth win the "
        "simulation does not show. A reduced-budget view ablation uses the same fit as the "
        "density sweep (80 BIGCLAM iterations and one restart; 60 NOCD epochs) on the Euclidean, "
        "correlation, and fused graphs (results/ceiling/view_ablation.csv). It does not replace "
        "the locked bake-off. On that ablation the LumA enrichment is specific to the fused graph: "
        "BIGCLAM's LumA OR is 3.00 (fused), 1.35 (Euclidean), and 0.91 (correlation). NOCD-X LumA "
        "ORs on the same fits are 0.97 (fused), 0.86 (Euclidean), and 0.86 (correlation).",
    )
    add_fig(
        doc,
        "fig_intclust_or.png",
        "Figure 11. METABRIC mixed-versus-single IntClust odds ratios. Blue bars have BH q < 0.05 "
        "inside that engine's orthogonal-label family.",
    )

    h(doc, "5. Discussion")
    p(
        doc,
        "The result this study can defend is methodological. A locked protocol and a pre-specified "
        "baseline rule keep a stronger LumA odds ratio, a stronger IntClust enrichment, and a higher "
        "simulation correlation from being rewritten as the lead method. BIGCLAM shows those point "
        "estimates and still fails FDR on the residual test (q ≈ 0.091). The simulation and "
        "IntClust benchmarks favor BIGCLAM on point estimates, which is why the policy, not the "
        "metric, determines the lead. NOCD-X is the lead because "
        "it is the only eligible engine in the quality-pass pool, not because it dominates BIGCLAM.",
    )
    p(
        doc,
        "The routes that would have raised the claim are null. SLPA collapse survives sparsification "
        "down to mean degree 6, with mixed fractions rising from 0.51 to 0.83 as k falls from 30 to 5. Boundary survival inside hard clusters and inside LumA does not "
        "survive BH (smallest METABRIC q ≈ 0.23). L1 does not explain the LumA odds-ratio gap. "
        "The nested concordance gain, ΔC, has a 95% CI that includes zero on every cohort.",
    )
    p(
        doc,
        "GSE25066 was run under the locked settings and was null. Mixed status is not associated "
        "with pCR. LumA has 5 pCR events (NOCD-X 1/35 versus 4/118, OR 0.84, p = 1.00). Hard "
        "cluster 1 has 16 pCR events (NOCD-X 2/34 versus 14/134, OR 0.54, p = 0.53). The LumA "
        "contrast is event-poor. The hard-cluster contrast is better powered and is still null "
        "(BH q = 1). BIGCLAM is a separate family, is not lead-eligible, and is also null. Age, "
        "stage, grade, residual cancer burden, and distant relapse-free survival were saved and "
        "were not used as covariates. No treatment column was added to SCAN-B, TCGA, or METABRIC. "
        "The null does not validate the soft membership.",
    )
    p(
        doc,
        "The GSE25066 graph uses 1,400 of 2,000 locked genes (70%). Rebuilding the discovery graph "
        "on those 1,400 genes leaves the hard partition close to the locked one (ARI 0.89) and the "
        "fused affinities highly correlated (Spearman 0.98). Soft engines were not refit, and the "
        "gene list was not changed. Other limits: expression-only fusion; the simulation and the "
        "L1 grid use a shorter fit than the locked bake-off (80 BIGCLAM iterations and one "
        "restart, and 60 NOCD epochs, against 250 iterations with five restarts and 250 epochs); "
        "programme tests use the locked 2,000 genes rather than the 1,400 shared with GSE25066, "
        "so the programme scores are not directly comparable across the discovery and "
        "treatment-response cohorts.",
    )

    h(doc, "6. Conclusion")
    p(
        doc,
        "The hard SNF partition remains the positive result (METABRIC age+PAM50 Cox p ≈ 1.7×10⁻⁴; "
        "BH q ≈ 8.3×10⁻⁴ within five hard methods). Under locked settings, the GSE25066 "
        "treatment-response test is null in LumA (five pCR "
        "events) and in the hard cluster with the most mixed patients (sixteen pCR events). "
        "No treatment-resistance claim is made. NOCD-X remains the soft lead by the pre-specified "
        "rule, and the lead is not reversed.",
    )

    h(doc, "7. Data and code availability")
    p(
        doc,
        "Tables: lead_decision_table.csv, hard_cox_bh.csv (Cox proportional-hazards BH), soft_residual_bh.csv, "
        "delta_c_bootstrap.csv, soft_cross_cohort_summary.csv. Ceiling tables, including "
        "view_ablation.csv and program_contrasts.csv, and decision_note.md are in results/ceiling/. GSE25066 files: "
        "results/gse25066/pcr_tests.csv and results/gse25066/assignments.csv. "
        "assignments.csv contains age, ER/PR/HER2, stage, grade, pCR/RD, residual cancer burden, "
        "distant relapse-free survival, and PAM50. Only pCR, PAM50, and the hard SNF cluster "
        "entered the tests. Gene-overlap sensitivity is results/gse25066/gene_overlap_sensitivity.json. "
        "Environment: environment.yml. "
        "Hyperparameters: SUPPLEMENTARY_METHODS.md.",
    )

    h(doc, "References")
    refs = [
        "Perou CM, et al. Molecular portraits of human breast tumours. Nature. 2000;406:747–752.",
        "Sørlie T, et al. Gene expression patterns of breast carcinomas distinguish tumor subclasses "
        "with clinical implications. Proc Natl Acad Sci USA. 2001;98:10869–10874.",
        "Parker JS, et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. "
        "J Clin Oncol. 2009;27:1160–1167.",
        "Curtis C, et al. The genomic and transcriptomic architecture of 2,000 breast tumours "
        "reveals novel subgroups. Nature. 2012;486:346–352.",
        "Dawson SJ, Rueda OM, Aparicio S, Caldas C. A new genome-driven integrated classification "
        "of breast cancer and its implications. EMBO J. 2013;32:617–628.",
        "The Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. "
        "Nature. 2012;490:61–70.",
        "Hoadley KA, et al. Cell-of-origin patterns dominate the molecular classification of 10,000 "
        "tumors from 33 types of cancer. Cell. 2018;173:291–304.e6.",
        "Cerami E, et al. The cBio cancer genomics portal: an open platform for exploring "
        "multidimensional cancer genomics data. Cancer Discov. 2012;2:401–404.",
        "Brueffer C, et al. Clinical value of RNA sequencing–based classifiers for prediction of "
        "the five conventional breast cancer biomarkers: a report from the population-based "
        "multicenter SCAN-B study. JCO Precis Oncol. 2018;2:1–18.",
        "Saal LH, et al. The Sweden Cancerome Analysis Network – Breast (SCAN-B) Initiative: a "
        "large-scale multicenter infrastructure towards implementation of breast cancer genomic "
        "analyses in the clinical routine. Genome Med. 2015;7:20. (GSE96058).",
        "Wang B, et al. Similarity network fusion for aggregating data types on a genomic scale. "
        "Nat Methods. 2014;11:333–337.",
        "Xu T, et al. CancerSubtypes: an R/Bioconductor package for molecular cancer subtype "
        "identification, validation and visualization. Bioinformatics. 2017;33:3131–3133.",
        "Ruan P, et al. Using association signal annotations to boost similarity network fusion. "
        "Bioinformatics. 2019;35:3718–3726.",
        "Rappoport N, Shamir R. NEighborhood-based Multi-Omics clustering (NEMO). Bioinformatics. "
        "2019;35:3059–3065.",
        "Gliozzo J, Soto Gomez MA, Bonometti A, Patak A, Casiraghi E, Valentini G. miss-SNF: a "
        "multimodal patient similarity network integration approach to handle completely missing "
        "data sources. Bioinformatics. 2025;41:btaf150.",
        "Shakyawar SK, Sajja BR, Patel JC, Guda C. iCluF: an unsupervised iterative "
        "cluster-fusion method for patient stratification using multiomics data. "
        "Bioinform Adv. 2024;4:vbae015. doi:10.1093/bioadv/vbae015.",
        "Yang J, Leskovec J. Overlapping community detection at scale: a nonnegative matrix "
        "factorization approach. In: Proceedings of the Sixth ACM International Conference on Web "
        "Search and Data Mining (WSDM). 2013:587–596.",
        "Yang J, McAuley J, Leskovec J. Community detection in networks with node attributes. In: "
        "IEEE International Conference on Data Mining (ICDM). 2013:1151–1156.",
        "Xie J, Szymanski BK, Liu X. SLPA: uncovering overlapping communities in social networks "
        "via a speaker-listener interaction dynamic process. In: 2011 IEEE 11th International "
        "Conference on Data Mining Workshops (ICDMW). 2011:344–349. doi:10.1109/ICDMW.2011.154.",
        "Shchur O, Günnemann S. Overlapping community detection with graph neural networks. In: "
        "Deep Learning on Graphs Workshop (DLG@KDD). 2019. arXiv:1909.12201.",
        "Airoldi EM, Blei DM, Fienberg SE, Xing EP. Mixed membership stochastic blockmodels. "
        "J Mach Learn Res. 2008;9:1981–2014.",
        "Ciaramella A, Nardone D, Staiano A. Data integration by fuzzy similarity-based "
        "hierarchical clustering. BMC Bioinformatics. 2020;21(Suppl 10):350. "
        "doi:10.1186/s12859-020-03567-6.",
        "Liu W, Murphy TB, Brennan L. Simplex-structured matrix factorisation: application of soft "
        "clustering to metabolomic data. Sci Rep. 2025;15:17817. doi:10.1038/s41598-025-02361-9.",
        "Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected "
        "communities. Sci Rep. 2019;9:5233. doi:10.1038/s41598-019-41695-z.",
        "Lee DD, Seung HS. Learning the parts of objects by non-negative matrix factorization. "
        "Nature. 1999;401:788–791.",
        "Cox DR. Regression models and life-tables. J R Stat Soc Series B. 1972;34:187–220. "
        "doi:10.1111/j.2517-6161.1972.tb00899.x.",
        "Harrell FE Jr, et al. Evaluating the yield of medical tests. JAMA. 1982;247:2543–2546.",
        "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful "
        "approach to multiple testing. J R Stat Soc Series B. 1995;57:289–300. "
        "doi:10.1111/j.2517-6161.1995.tb02031.x.",
        "Hatzis C, et al. A genomic predictor of response and survival following taxane-anthracycline "
        "chemotherapy for invasive breast cancer. JAMA. 2011;305:1873–1881.",
    ]
    for i, ref in enumerate(refs, 1):
        para = p(doc, f"[{i}] {ref}")
        para.paragraph_format.left_indent = Inches(0.3)

    doc.save(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
