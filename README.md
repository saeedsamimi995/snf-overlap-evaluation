# New Clustering of Breast Cancer Subtypes Using Gene Expression Data and the BIGCLAM Method

A comprehensive implementation of the BIGCLAM (Big Community Affiliation Model) algorithm for clustering breast cancer subtypes from gene expression data. This project includes data preprocessing, graph construction, community detection, and evaluation using MLP and SVM classifiers.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project implements a pipeline for clustering breast cancer subtypes using the BIGCLAM method on gene expression data. The workflow includes:

1. **Data Preprocessing**: Feature selection using variance threshold
2. **Graph Construction**: Building similarity graphs from gene expression profiles
3. **BIGCLAM Training**: GPU-accelerated community detection
4. **Classification**: Evaluation using MLP and SVM classifiers
5. **Visualization**: Confusion matrices and ROC curves

The implementation is tested on the [GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058) breast cancer gene expression dataset.

## Features

- **GPU-Accelerated BIGCLAM**: Optimized PyTorch implementation for efficient computation
- **Modular Architecture**: Clean, professional code structure suitable for research publications
- **Comprehensive Evaluation**: MLP and SVM classifiers with extensive metrics
- **Reproducible Pipeline**: Configuration-based approach for easy parameter tuning
- **Visualization Tools**: Automated generation of confusion matrices and ROC curves

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended for BIGCLAM training)
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd New-Clustering-of-Breast-Cancer-Subtypes-Using-Gene-Expression-Data-and-the-BIGCLAM-Method
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config/
│   └── config.yaml          # Configuration file with hyperparameters
├── data/                     # Data directory (create and add your data here)
│   └── reduced_data_BC_label.csv
├── notebooks/                # Jupyter notebooks for exploratory analysis
│   └── bigclam_training_v4_0.ipynb
├── src/                      # Source code modules
│   ├── analysis/            # Reviewer-addressed analysis scripts
│   ├── bigclam/             # BIGCLAM model implementation
│   ├── classifiers/         # MLP and SVM classifiers
│   ├── evaluation/          # Evaluation and visualization tools
│   ├── preprocessing/       # Data loading and preprocessing
│   └── utils/               # Utility functions
├── docs/                     # Documentation
│   └── METHODOLOGY.md       # Detailed methodology documentation
├── models/                   # Saved model files (created during training)
├── results/                  # Output visualizations and results
├── main.py                   # Main pipeline script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

### Quick Start

1. **Prepare your data**:
   - Place your gene expression data CSV file in the `data/` directory
   - Update the `input_file` path in `config/config.yaml`
   - The CSV should have samples as columns and genes as rows, with a target column (e.g., PAM50)

2. **Run the complete pipeline**:
```bash
python main.py --config config/config.yaml
```

### Configuration

Edit `config/config.yaml` to adjust parameters:

```yaml
# Data paths
data:
  input_file: "data/gene_expression_data_target_added.csv"
  output_file: "data/reduced_data_BC_label.csv"

# Preprocessing
preprocessing:
  variance_threshold: 13
  similarity_threshold: 0.4

# BIGCLAM parameters
bigclam:
  max_communities: 10
  iterations: 100
  learning_rate: 0.08

# Classifier parameters
classifiers:
  mlp:
    num_runs: 10
    num_epochs: 200
    learning_rate: 0.001
  svm:
    num_runs: 10
    kernel: "rbf"
    C: 0.1
```

### Running Individual Components

You can also import and use individual modules:

```python
from src.preprocessing import load_data, apply_variance_threshold
from src.bigclam import train_bigclam
from src.classifiers import train_mlp, train_svm

# Load and preprocess data
X, data = load_data("data/your_data.csv")
X_reduced, _, _ = apply_variance_threshold(X, data, threshold=13)

# Train BIGCLAM
F, num_communities = train_bigclam(A, max_communities=10)
```

### Analysis Scripts for Reviewer Concerns

The repository includes specialized analysis scripts to address reviewer concerns:

1. **Baseline Comparison**: Compare classifiers on original vs BIGCLAM-filtered data
   ```bash
   python -m src.analysis.baseline_comparison \
     --original_data data/gene_expression_data_target_added.csv \
     --bigclam_data data/reduced_data_BC_label.csv
   ```

2. **Computational Benchmarking**: Measure pipeline performance and efficiency
   ```bash
   python -m src.analysis.benchmark \
     --data data/gene_expression_data_target_added.csv
   ```

3. **Parameter Sensitivity Analysis**: Justify threshold selections
   ```bash
   python -m src.analysis.parameter_sensitivity \
     --data data/gene_expression_data_target_added.csv --all
   ```

4. **Run All Analyses**: Execute all reviewer-addressed analyses
   ```bash
   python run_all_analyses.py \
     --original_data data/gene_expression_data_target_added.csv \
     --bigclam_data data/reduced_data_BC_label.csv
   ```

See `docs/METHODOLOGY.md` for detailed methodological documentation addressing reviewer concerns.

## Configuration

The project uses YAML configuration files for easy parameter management. Key configuration sections:

- **Data**: File paths and data settings
- **Preprocessing**: Feature selection and graph construction parameters
- **BIGCLAM**: Model training hyperparameters
- **Classifiers**: MLP and SVM training parameters
- **Output**: Directories for saving results and models

## Output

The pipeline generates:

- **Labeled Dataset**: `data/reduced_data_BC_label.csv` with community assignments
- **Models**: Trained MLP and SVM models in `models/`
- **Visualizations**: Confusion matrices and ROC curves in `results/`
- **Graph Data**: Adjacency matrices and community assignments in `data/BIGCLAM_data/`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bigclam_breast_cancer,
  title={New Clustering of Breast Cancer Subtypes Using Gene Expression Data and the BIGCLAM Method},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repository}
}
```

This project builds on the methodology from:
- Leskovec, J., & Sosič, R. (2016). SNAP: A general-purpose network analysis and graph-mining library. *ACM Transactions on Intelligent Systems and Technology (TIST)*, 8(1), 1-20.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Special thanks to Dr. Mohammad Teshnehlab and Dr. Saeed Pirmoradi for their guidance and support
- Dataset: [GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058)
- BIGCLAM methodology based on Leskovec et al. (2013)

## Contributing

Contributions are welcome! If you have suggestions or find issues, please open an issue or submit a pull request.

## Contact

For questions or support, please open an issue on the GitHub repository.
