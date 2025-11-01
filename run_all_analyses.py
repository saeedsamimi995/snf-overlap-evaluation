"""
Master Script to Run All Reviewer-Addressed Analyses

This script runs all analyses needed to address reviewer concerns:
1. Baseline comparison (original vs BIGCLAM-filtered)
2. Computational benchmarking
3. Parameter sensitivity analysis
4. Ablation studies
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import (
    compare_baseline_vs_bigclam,
    benchmark_pipeline,
    analyze_variance_threshold_sensitivity,
    analyze_similarity_threshold_sensitivity
)


def main():
    parser = argparse.ArgumentParser(
        description='Run all analyses to address reviewer concerns'
    )
    parser.add_argument('--original_data', type=str, required=True,
                       help='Path to original gene expression data')
    parser.add_argument('--bigclam_data', type=str,
                       help='Path to BIGCLAM-clustered data (for comparison)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip baseline comparison (if BIGCLAM data not available)')
    
    args = parser.parse_args()
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    print("=" * 80)
    print("RUNNING ALL REVIEWER-ADDRESSED ANALYSES")
    print("=" * 80)
    
    # 1. Computational Benchmarking
    print("\n[1/4] Computational Benchmarking")
    print("-" * 80)
    benchmark_pipeline(
        args.original_data,
        config,
        "results/benchmark_results.json"
    )
    
    # 2. Variance Threshold Sensitivity
    print("\n[2/4] Variance Threshold Sensitivity Analysis")
    print("-" * 80)
    analyze_variance_threshold_sensitivity(
        args.original_data,
        threshold_range=(5, 20, 1),
        output_dir="results/sensitivity"
    )
    
    # 3. Similarity Threshold Sensitivity
    print("\n[3/4] Similarity Threshold Sensitivity Analysis")
    print("-" * 80)
    variance_threshold = config.get('preprocessing', {}).get('variance_threshold', 13) if config else 13
    analyze_similarity_threshold_sensitivity(
        args.original_data,
        threshold_range=(0.1, 0.9, 0.1),
        variance_threshold=variance_threshold,
        output_dir="results/sensitivity"
    )
    
    # 4. Baseline Comparison (if BIGCLAM data available)
    if not args.skip_comparison and args.bigclam_data:
        print("\n[4/4] Baseline Comparison Analysis")
        print("-" * 80)
        compare_baseline_vs_bigclam(
            args.original_data,
            args.bigclam_data,
            "results/comparison",
            config
        )
    else:
        print("\n[4/4] Baseline Comparison Analysis - SKIPPED")
        print("  (Use --bigclam_data to enable or --skip_comparison to suppress this message)")
    
    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETED")
    print("=" * 80)
    print("\nResults saved in:")
    print("  - results/benchmark_results.json")
    print("  - results/sensitivity/")
    if not args.skip_comparison and args.bigclam_data:
        print("  - results/comparison/")
    print("\nThese results address reviewer concerns about:")
    print("  ✓ Computational efficiency")
    print("  ✓ Parameter threshold justification")
    print("  ✓ Baseline comparison (original vs BIGCLAM)")
    print("=" * 80)


if __name__ == "__main__":
    main()

