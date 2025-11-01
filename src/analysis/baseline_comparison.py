"""
Baseline Comparison Script

This script compares the performance of classifiers on:
1. Original data (before BIGCLAM clustering)
2. BIGCLAM-filtered/clustered data

Addresses Reviewer 1 concern: "What happens if one applies SVM or MLP to the original data?"
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import (
    load_data, apply_variance_threshold,
    augment_data, split_data, normalize_data, encode_labels
)
from src.classifiers import train_mlp, train_svm
from src.evaluation import save_mlp_results, save_svm_results
import json
from datetime import datetime


def compare_baseline_vs_bigclam(
    original_data_path,
    bigclam_data_path,
    output_dir="results/comparison",
    config=None
):
    """
    Compare classifier performance on original vs BIGCLAM-clustered data.
    
    Args:
        original_data_path: Path to original gene expression data
        bigclam_data_path: Path to BIGCLAM-clustered data
        output_dir: Directory to save comparison results
        config: Configuration dictionary (optional)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("BASELINE COMPARISON: Original Data vs BIGCLAM-Filtered Data")
    print("=" * 80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'original_data': {},
        'bigclam_data': {}
    }
    
    # Process Original Data
    print("\n[1] Processing ORIGINAL data...")
    X_orig, data_orig = load_data(original_data_path)
    
    # Apply variance threshold (same as BIGCLAM pipeline)
    X_reduced_orig, _, _ = apply_variance_threshold(
        X_orig, data_orig, 
        threshold=config.get('preprocessing', {}).get('variance_threshold', 13) if config else 13
    )
    
    # For original data, use PAM50 labels if available, otherwise create dummy labels
    # This assumes the original data has a label column
    if 'Community' not in X_reduced_orig.columns:
        # If no community column, we'll use original labels or create from data
        print("  Note: Original data doesn't have community labels, using original structure")
    
    # Prepare original data for classification
    X_orig_aug = augment_data(
        X_reduced_orig,
        noise_mean=config.get('augmentation', {}).get('noise_mean', 0.0) if config else 0.0,
        noise_std=config.get('augmentation', {}).get('noise_std', 0.1) if config else 0.1
    )
    
    X_train_orig, X_valid_orig, X_test_orig, y_train_orig, y_valid_orig, y_test_orig = split_data(
        X_orig_aug,
        test_size=config.get('data_splitting', {}).get('test_size', 0.2) if config else 0.2,
        valid_size=config.get('data_splitting', {}).get('valid_size', 0.2) if config else 0.2,
        random_state=config.get('data_splitting', {}).get('random_state', 42) if config else 42
    )
    
    X_train_orig_norm, X_valid_orig_norm, X_test_orig_norm, _ = normalize_data(
        X_train_orig, X_valid_orig, X_test_orig
    )
    
    y_train_orig_oh, y_valid_orig_oh, y_test_orig_oh, _ = encode_labels(
        y_train_orig, y_valid_orig, y_test_orig
    )
    
    # Train MLP on original data
    print("\n[2] Training MLP on ORIGINAL data...")
    mlp_orig = train_mlp(
        X_train_orig_norm, y_train_orig_oh,
        X_valid_orig_norm, y_valid_orig_oh,
        X_test_orig_norm, y_test_orig_oh,
        num_runs=config.get('classifiers', {}).get('mlp', {}).get('num_runs', 5) if config else 5,
        models_dir=f"{output_dir}/models_original"
    )
    results['original_data']['mlp'] = {
        'test_metrics': mlp_orig['best_test_metrics'],
        'valid_metrics': mlp_orig['best_valid_metrics']
    }
    
    # Train SVM on original data
    print("\n[3] Training SVM on ORIGINAL data...")
    svm_orig = train_svm(
        X_train_orig_norm, y_train_orig,
        X_valid_orig_norm, y_valid_orig,
        X_test_orig_norm, y_test_orig,
        num_runs=config.get('classifiers', {}).get('svm', {}).get('num_runs', 5) if config else 5,
        models_dir=f"{output_dir}/models_original"
    )
    results['original_data']['svm'] = {
        'test_metrics': svm_orig['test_metrics'],
        'valid_metrics': svm_orig['valid_metrics']
    }
    
    # Process BIGCLAM Data
    print("\n[4] Processing BIGCLAM-clustered data...")
    X_bigclam = pd.read_csv(bigclam_data_path)
    
    # Prepare BIGCLAM data for classification
    X_bigclam_aug = augment_data(
        X_bigclam,
        noise_mean=config.get('augmentation', {}).get('noise_mean', 0.0) if config else 0.0,
        noise_std=config.get('augmentation', {}).get('noise_std', 0.1) if config else 0.1
    )
    
    X_train_big, X_valid_big, X_test_big, y_train_big, y_valid_big, y_test_big = split_data(
        X_bigclam_aug,
        test_size=config.get('data_splitting', {}).get('test_size', 0.2) if config else 0.2,
        valid_size=config.get('data_splitting', {}).get('valid_size', 0.2) if config else 0.2,
        random_state=config.get('data_splitting', {}).get('random_state', 42) if config else 42
    )
    
    X_train_big_norm, X_valid_big_norm, X_test_big_norm, _ = normalize_data(
        X_train_big, X_valid_big, X_test_big
    )
    
    y_train_big_oh, y_valid_big_oh, y_test_big_oh, _ = encode_labels(
        y_train_big, y_valid_big, y_test_big
    )
    
    # Train MLP on BIGCLAM data
    print("\n[5] Training MLP on BIGCLAM-clustered data...")
    mlp_big = train_mlp(
        X_train_big_norm, y_train_big_oh,
        X_valid_big_norm, y_valid_big_oh,
        X_test_big_norm, y_test_big_oh,
        num_runs=config.get('classifiers', {}).get('mlp', {}).get('num_runs', 5) if config else 5,
        models_dir=f"{output_dir}/models_bigclam"
    )
    results['bigclam_data']['mlp'] = {
        'test_metrics': mlp_big['best_test_metrics'],
        'valid_metrics': mlp_big['best_valid_metrics']
    }
    
    # Train SVM on BIGCLAM data
    print("\n[6] Training SVM on BIGCLAM-clustered data...")
    svm_big = train_svm(
        X_train_big_norm, y_train_big,
        X_valid_big_norm, y_valid_big,
        X_test_big_norm, y_test_big,
        num_runs=config.get('classifiers', {}).get('svm', {}).get('num_runs', 5) if config else 5,
        models_dir=f"{output_dir}/models_bigclam"
    )
    results['bigclam_data']['svm'] = {
        'test_metrics': svm_big['test_metrics'],
        'valid_metrics': svm_big['valid_metrics']
    }
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print("\nMLP Performance:")
    print(f"  Original Data - Test MCC: {results['original_data']['mlp']['test_metrics'][-1]:.4f}")
    print(f"  BIGCLAM Data - Test MCC: {results['bigclam_data']['mlp']['test_metrics'][-1]:.4f}")
    improvement_mlp = results['bigclam_data']['mlp']['test_metrics'][-1] - results['original_data']['mlp']['test_metrics'][-1]
    print(f"  Improvement: {improvement_mlp:+.4f} ({improvement_mlp/results['original_data']['mlp']['test_metrics'][-1]*100:+.2f}%)")
    
    print("\nSVM Performance:")
    print(f"  Original Data - Test MCC: {results['original_data']['svm']['test_metrics']['mcc']:.4f}")
    print(f"  BIGCLAM Data - Test MCC: {results['bigclam_data']['svm']['test_metrics']['mcc']:.4f}")
    improvement_svm = results['bigclam_data']['svm']['test_metrics']['mcc'] - results['original_data']['svm']['test_metrics']['mcc']
    print(f"  Improvement: {improvement_svm:+.4f} ({improvement_svm/results['original_data']['svm']['test_metrics']['mcc']*100:+.2f}%)")
    
    # Save results
    with open(f"{output_dir}/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}/comparison_results.json")
    
    return results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description='Compare baseline vs BIGCLAM-filtered data performance'
    )
    parser.add_argument('--original_data', type=str, required=True,
                       help='Path to original gene expression data')
    parser.add_argument('--bigclam_data', type=str, required=True,
                       help='Path to BIGCLAM-clustered data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load config
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    compare_baseline_vs_bigclam(
        args.original_data,
        args.bigclam_data,
        args.output_dir,
        config
    )

