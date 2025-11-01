"""
Main pipeline script for BIGCLAM Breast Cancer Subtype Clustering.

This script executes the complete pipeline:
1. Data loading and preprocessing
2. Graph construction
3. BIGCLAM training
4. Classification with MLP and SVM
5. Evaluation and visualization

Usage:
    python main.py --config config/config.yaml
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# Import project modules
from src.preprocessing import (
    load_data, apply_variance_threshold,
    augment_data, split_data, normalize_data, encode_labels
)
from src.utils.graph_utils import (
    create_adjacency_and_assignments, gen_json, save_graph_data
)
from src.bigclam import train_bigclam, BIGCLAM
from src.classifiers import train_mlp, train_svm
from src.evaluation import save_mlp_results, save_svm_results


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_bigclam_pipeline(config):
    """
    Run the complete BIGCLAM pipeline.
    
    Args:
        config (dict): Configuration dictionary.
    """
    print("=" * 80)
    print("BIGCLAM Breast Cancer Subtype Clustering Pipeline")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs(config['data']['save_path'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    
    # Step 1: Data Loading and Preprocessing
    print("\n[Step 1] Loading and preprocessing data...")
    X, data = load_data(config['data']['input_file'])
    
    X_reduced, selected_feature_names, df_selected_features = apply_variance_threshold(
        X, data, threshold=config['preprocessing']['variance_threshold']
    )
    
    # Step 2: Graph Construction
    print("\n[Step 2] Constructing graph...")
    df_for_graph = X_reduced.copy()
    
    # Create adjacency matrix and community assignments
    A, p2c = create_adjacency_and_assignments(
        df_for_graph, threshold=config['preprocessing']['similarity_threshold']
    )
    
    # Generate and save JSON data
    data_json = gen_json(A, p2c)
    with open('data.json', 'w') as f:
        json.dump(data_json, f, indent=4)
    
    # Save graph data
    save_graph_data(A, p2c, config['data']['save_path'])
    
    # Step 3: BIGCLAM Training
    print("\n[Step 3] Training BIGCLAM model...")
    F, best_num_communities = train_bigclam(
        A,
        max_communities=config['bigclam']['max_communities'],
        iterations=config['bigclam']['iterations'],
        lr=config['bigclam']['learning_rate']
    )
    
    # Get community assignments
    F_argmax = np.argmax(F, axis=1)
    
    # Add community labels to dataframe
    df_reduced = X_reduced.copy()
    df_reduced['Community'] = F_argmax
    
    # Save labeled data
    output_file = config['data']['output_file']
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df_reduced.to_csv(output_file, index=False)
    print(f"Labeled data saved to: {output_file}")
    
    # Generate JSON with BIGCLAM assignments
    data_json_with_bigclam = gen_json(A, p2c, F_argmax)
    with open('data_bigclam.json', 'w') as f:
        json.dump(data_json_with_bigclam, f, indent=4)
    
    # Step 4: Data Preparation for Classification
    print("\n[Step 4] Preparing data for classification...")
    
    # Load the clustered dataset
    X_labeled = pd.read_csv(output_file)
    
    # Augment data
    X_augmented = augment_data(
        X_labeled,
        noise_mean=config['augmentation']['noise_mean'],
        noise_std=config['augmentation']['noise_std']
    )
    
    # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        X_augmented,
        test_size=config['data_splitting']['test_size'],
        valid_size=config['data_splitting']['valid_size'],
        random_state=config['data_splitting'].get('random_state', None)
    )
    
    # Normalize data
    X_train_norm, X_valid_norm, X_test_norm, scaler = normalize_data(
        X_train, X_valid, X_test
    )
    
    # Encode labels
    y_train_onehot, y_valid_onehot, y_test_onehot, encoder = encode_labels(
        y_train, y_valid, y_test
    )
    
    # Step 5: MLP Classification
    print("\n[Step 5] Training MLP classifier...")
    mlp_results = train_mlp(
        X_train_norm, y_train_onehot,
        X_valid_norm, y_valid_onehot,
        X_test_norm, y_test_onehot,
        num_runs=config['classifiers']['mlp']['num_runs'],
        num_epochs=config['classifiers']['mlp']['num_epochs'],
        lr=config['classifiers']['mlp']['learning_rate'],
        patience=config['classifiers']['mlp']['patience'],
        weight_decay=config['classifiers']['mlp']['weight_decay'],
        dropout_rate=config['classifiers']['mlp']['dropout_rate'],
        hidden_layers=tuple(config['classifiers']['mlp']['hidden_layers']),
        models_dir=config['output']['models_dir']
    )
    
    # Save MLP results
    save_mlp_results(
        mlp_results,
        results_dir=config['output']['results_dir'],
        y_train_onehot=y_train_onehot,
        y_valid_onehot=y_valid_onehot,
        y_test_onehot=y_test_onehot
    )
    
    # Step 6: SVM Classification
    print("\n[Step 6] Training SVM classifier...")
    svm_results = train_svm(
        X_train_norm, y_train,
        X_valid_norm, y_valid,
        X_test_norm, y_test,
        num_runs=config['classifiers']['svm']['num_runs'],
        kernel=config['classifiers']['svm']['kernel'],
        C=config['classifiers']['svm']['C'],
        gamma=config['classifiers']['svm']['gamma'],
        probability=config['classifiers']['svm']['probability'],
        models_dir=config['output']['models_dir']
    )
    
    # Save SVM results
    save_svm_results(
        svm_results,
        results_dir=config['output']['results_dir']
    )
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {config['output']['results_dir']}")
    print(f"Models saved to: {config['output']['models_dir']}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='BIGCLAM Breast Cancer Subtype Clustering Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a configuration file or use the default: config/config.yaml")
        return
    
    config = load_config(args.config)
    
    # Run pipeline
    try:
        run_bigclam_pipeline(config)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise


if __name__ == '__main__':
    main()

