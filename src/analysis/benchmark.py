"""
Computational Benchmarking Script

Measures and reports computational efficiency of the BIGCLAM pipeline.
Addresses Reviewer 1 & 2 concern about computational efficiency discussion.
"""

import time
import psutil
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import load_data, apply_variance_threshold
from src.utils.graph_utils import create_adjacency_and_assignments
from src.bigclam import train_bigclam
from src.classifiers import train_mlp, train_svm


class BenchmarkTimer:
    """Context manager for timing and memory profiling."""
    
    def __init__(self, name, record_memory=True):
        self.name = name
        self.record_memory = record_memory
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        
    def __enter__(self):
        process = psutil.Process(os.getpid())
        if self.record_memory:
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        process = psutil.Process(os.getpid())
        if self.record_memory:
            self.end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = self.end_memory - self.start_memory
        else:
            memory_used = None
        
        elapsed = self.end_time - self.start_time
        print(f"  {self.name}: {elapsed:.2f}s", end="")
        if memory_used is not None:
            print(f" (Memory: {memory_used:.2f} MB)")
        else:
            print()
        
        self.elapsed_time = elapsed
        self.memory_used = memory_used


def benchmark_pipeline(data_path, config=None, output_file="results/benchmark_results.json"):
    """
    Benchmark the complete BIGCLAM pipeline.
    
    Args:
        data_path: Path to input data
        config: Configuration dictionary
        output_file: Path to save benchmark results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024**3,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        'timing': {},
        'memory': {}
    }
    
    print("=" * 80)
    print("COMPUTATIONAL BENCHMARKING")
    print("=" * 80)
    print(f"System: {results['system_info']['cpu_count']} CPUs, "
          f"{results['system_info']['total_memory_gb']:.2f} GB RAM")
    if results['system_info']['gpu_available']:
        print(f"GPU: {results['system_info']['gpu_name']}")
    print("=" * 80)
    
    # 1. Data Loading
    print("\n[1] Data Loading and Preprocessing")
    with BenchmarkTimer("Data loading") as timer:
        X, data = load_data(data_path)
        n_samples, n_features = X.shape
    results['timing']['data_loading'] = timer.elapsed_time
    results['data_shape'] = {'samples': int(n_samples), 'features': int(n_features)}
    
    # 2. Feature Selection
    print("\n[2] Feature Selection")
    threshold = config.get('preprocessing', {}).get('variance_threshold', 13) if config else 13
    with BenchmarkTimer("Variance thresholding") as timer:
        X_reduced, _, _ = apply_variance_threshold(X, data, threshold=threshold)
        n_features_reduced = X_reduced.shape[1]
    results['timing']['feature_selection'] = timer.elapsed_time
    results['reduction'] = {
        'original_features': int(n_features),
        'reduced_features': int(n_features_reduced),
        'reduction_ratio': float(n_features_reduced / n_features)
    }
    
    # 3. Graph Construction
    print("\n[3] Graph Construction")
    similarity_threshold = config.get('preprocessing', {}).get('similarity_threshold', 0.4) if config else 0.4
    with BenchmarkTimer("Adjacency matrix creation") as timer:
        A, p2c = create_adjacency_and_assignments(X_reduced, threshold=similarity_threshold)
        n_edges = np.sum(A) // 2  # Undirected graph
    results['timing']['graph_construction'] = timer.elapsed_time
    results['graph_stats'] = {
        'nodes': int(A.shape[0]),
        'edges': int(n_edges),
        'density': float(n_edges / (A.shape[0] * (A.shape[0] - 1) / 2))
    }
    
    # 4. BIGCLAM Training
    print("\n[4] BIGCLAM Training")
    max_communities = config.get('bigclam', {}).get('max_communities', 10) if config else 10
    iterations = config.get('bigclam', {}).get('iterations', 100) if config else 100
    lr = config.get('bigclam', {}).get('learning_rate', 0.08) if config else 0.08
    
    with BenchmarkTimer("BIGCLAM training") as timer:
        F, best_num_communities = train_bigclam(
            A, max_communities=max_communities, iterations=iterations, lr=lr
        )
    results['timing']['bigclam_training'] = timer.elapsed_time
    results['bigclam'] = {
        'communities_found': int(best_num_communities),
        'max_communities_tested': int(max_communities),
        'iterations_per_community': int(iterations)
    }
    
    # Calculate total time
    total_time = sum(results['timing'].values())
    results['timing']['total'] = total_time
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Pipeline Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nTime Breakdown:")
    for step, time_val in results['timing'].items():
        if step != 'total':
            percentage = (time_val / total_time) * 100
            print(f"  {step.replace('_', ' ').title()}: {time_val:.2f}s ({percentage:.1f}%)")
    
    print(f"\nData Reduction: {n_features} → {n_features_reduced} features "
          f"({results['reduction']['reduction_ratio']*100:.1f}% retained)")
    print(f"Graph: {results['graph_stats']['nodes']} nodes, "
          f"{results['graph_stats']['edges']} edges "
          f"(density: {results['graph_stats']['density']:.4f})")
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nBenchmark results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Benchmark BIGCLAM pipeline performance')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                       help='Output file for benchmark results')
    
    args = parser.parse_args()
    
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    benchmark_pipeline(args.data, config, args.output)

