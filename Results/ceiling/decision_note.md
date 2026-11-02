# Ceiling decision

Primary identity: **methods paper**. The lead policy is not reversed.

Simulation and IntClust were both run. Neither gives NOCD-X a win that could justify the lead on performance.

- Simulation mean correlation with planted soft membership: BIGCLAM 0.53, SLPA 0.17, graph NOCD 0.10. Mixed-label ARI < 0.10 for all.
- IntClust enrichment is real and stronger for BIGCLAM (IntClust 8 OR 3.24) than for NOCD-X (IntClust 7 OR 1.95; IntClust 10 depletion OR 0.058).

Nulls, stated as nulls:

- SLPA peak ratio stays 1.0 from mean degree 6.4 to 34.
- Boundary survival: smallest METABRIC BH q = 0.23.
- BIGCLAM L1 in {0, 0.001, 0.01, 0.1} leaves LumA OR at 3.00 and residual p at 0.56.
- No treatment columns on SCAN-B, TCGA, or METABRIC.
- GSE25066 pCR, locked settings, no retuning: NOCD-X LumA 1/35 vs 4/118 (p = 1.00) and hard cluster 1 2/34 vs 14/134 (p = 0.53); BH q = 1 in that two-test family. BIGCLAM's separate family is also null. LumA has 5 pCR events; the hard cluster has 16 and is still null. No treatment-resistance claim. Age, stage, grade, RCB, and DRFS were saved and not used. Discovery SNF rebuilt on the 1,400 array genes: hard-label ARI 0.89, fused-affinity Spearman 0.98. The paper stays a methods paper.

Supported lesson: a classical baseline can look biologically stronger without surviving FDR (locked residual q ≈ 0.091). NOCD-X remains the policy lead because it is baseline-ineligible for BIGCLAM to take that role, not because NOCD-X is the better estimator.
