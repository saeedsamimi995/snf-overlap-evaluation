"""
Parameter Sensitivity Analysis

Tests different parameter values to justify threshold choices.
Addresses Reviewer 1 & 2 concerns about threshold justification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import load_data, apply_variance_threshold
from src.utils.graph_utils import create_adjacency_and_assignments
from src.bigclam import train_bigclam


def analyze_variance_threshold_sensitivity(data_path, threshold_range=(5, 20, 1), output_dir="results/sensitivity"):
    """
    Analyze sensitivity to variance threshold parameter.
    
    Args:
        data_path: Path to input data
        threshold_range: (start, end, step) for threshold values
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Variance Threshold Sensitivity Analysis")
    print("=" * 80)
    
    X, data = load_data(data_path)
    
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        try:
            X_reduced, _, _ = apply_variance_threshold(X, data, threshold=threshold)
            n_features = X_reduced.shape[1]
            reduction_ratio = n_features / X.shape[1]
            
            results.append({
                'threshold': float(threshold),
                'n_features': int(n_features),
                'reduction_ratio': float(reduction_ratio)
            })
            print(f"  Features retained: {n_features} ({reduction_ratio*100:.1f}%)")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['threshold'], df_results['n_features'], marker='o')
    plt.axvline(x=13, color='r', linestyle='--', label='Selected threshold (13)')
    plt.xlabel('Variance Threshold', fontsize=12)
    plt.ylabel('Number of Features Retained', fontsize=12)
    plt.title('Feature Selection Sensitivity to Variance Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_threshold_sensitivity.png", dpi=300)
    plt.close()
    
    # Save results
    df_results.to_csv(f"{output_dir}/variance_threshold_results.csv", index=False)
    
    print(f"\nResults saved to: {output_dir}/")
    return df_results


def analyze_similarity_threshold_sensitivity(data_path, threshold_range=(0.1, 0.9, 0.1), 
                                            variance_threshold=13, output_dir="results/sensitivity"):
    """
    Analyze sensitivity to similarity threshold for graph construction.
    
    Args:
        data_path: Path to input data
        threshold_range: (start, end, step) for threshold values
        variance_threshold: Variance threshold for feature selection
        output_dir: Directory to save results
    """
    import os
    os.makedirs(os.path.dirname(output_dir) if os.path.dirname(output_dir) else '.', exist_ok=True)
    
    print("\nSimilarity Threshold Sensitivity Analysis")
    print("=" * 80)
    
    X, data = load_data(data_path)
    X_reduced, _, _ = apply_variance_threshold(X, data, threshold=variance_threshold)
    
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting similarity threshold: {threshold:.2f}")
        try:
            A, _ = create_adjacency_and_assignments(X_reduced, threshold=threshold)
            n_edges = np.sum(A) // 2
            n_nodes = A.shape[0]
            density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            
            results.append({
                'threshold': float(threshold),
                'n_edges': int(n_edges),
                'density': float(density),
                'avg_degree': float(n_edges * 2 / n_nodes) if n_nodes > 0 else 0
            })
            print(f"  Edges: {n_edges}, Density: {density:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(df_results['threshold'], df_results['n_edges'], marker='o', color='blue')
    axes[0].axvline(x=0.4, color='r', linestyle='--', label='Selected threshold (0.4)')
    axes[0].set_xlabel('Similarity Threshold', fontsize=12)
    axes[0].set_ylabel('Number of Edges', fontsize=12)
    axes[0].set_title('Graph Edge Count vs Similarity Threshold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(df_results['threshold'], df_results['density'], marker='s', color='green')
    axes[1].axvline(x=0.4, color='r', linestyle='--', label='Selected threshold (0.4)')
    axes[1].set_xlabel('Similarity Threshold', fontsize=12)
    axes[1].set_ylabel('Graph Density', fontsize=12)
    axes[1].set_title('Graph Density vs Similarity Threshold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_threshold_sensitivity.png", dpi=300)
    plt.close()
    
    # Save results
    df_results.to_csv(f"{output_dir}/similarity_threshold_results.csv", index=False)
    
    print(f"\nResults saved to: {output_dir}/")
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter sensitivity analysis')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--variance_threshold', action='store_true',
                       help='Analyze variance threshold sensitivity')
    parser.add_argument('--similarity_threshold', action='store_true',
                       help='Analyze similarity threshold sensitivity')
    parser.add_argument('--all', action='store_true',
                       help='Run all sensitivity analyses')
    parser.add_argument('--output_dir', type=str, default='results/sensitivity',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.all or args.variance_threshold:
        analyze_variance_threshold_sensitivity(args.data, output_dir=args.output_dir)
    
    if args.all or args.similarity_threshold:
        analyze_similarity_threshold_sensitivity(args.data, output_dir=args.output_dir)

