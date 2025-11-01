# Quick Start Guide

This guide will help you get started with the BIGCLAM Breast Cancer Clustering project quickly.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for faster BIGCLAM training)
- 8GB+ RAM

## Installation Steps

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place your gene expression data CSV file in the `data/` directory. The expected format:
- Samples as columns
- Genes/features as rows
- Optional: Last column contains PAM50 labels or other targets

**Example file structure:**
```
data/
  └── gene_expression_data_target_added.csv
```

### 4. Configure Parameters

Edit `config/config.yaml` to adjust parameters for your dataset:

```yaml
data:
  input_file: "data/gene_expression_data_target_added.csv"  # Update this path

preprocessing:
  variance_threshold: 13  # Adjust based on your data

bigclam:
  max_communities: 10  # Maximum communities to explore
  iterations: 100      # Training iterations
```

### 5. Run the Pipeline

```bash
python main.py --config config/config.yaml
```

## Expected Output

After running the pipeline, you'll find:

1. **Processed Data**: `data/reduced_data_BC_label.csv` - Dataset with community assignments
2. **Models**: `models/` - Trained MLP and SVM models
3. **Visualizations**: `results/` - Confusion matrices and ROC curves
4. **Graph Data**: `data/BIGCLAM_data/` - Adjacency matrices and community assignments

## Using Individual Components

### Load and Preprocess Data Only

```python
from src.preprocessing import load_data, apply_variance_threshold

X, data = load_data("data/your_data.csv")
X_reduced, selected_features, df_features = apply_variance_threshold(X, data, threshold=13)
```

### Train BIGCLAM Only

```python
from src.utils.graph_utils import create_adjacency_and_assignments
from src.bigclam import train_bigclam
import numpy as np

# Create graph
A, p2c = create_adjacency_and_assignments(df, threshold=0.4)

# Train BIGCLAM
F, num_communities = train_bigclam(A, max_communities=10, iterations=100)
F_argmax = np.argmax(F, axis=1)  # Get community assignments
```

### Train Classifiers Only

```python
from src.classifiers import train_mlp, train_svm
from src.preprocessing import augment_data, split_data, normalize_data, encode_labels

# Assuming you have X_labeled dataframe with 'Community' column
X_aug = augment_data(X_labeled)
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X_aug)
X_train_norm, X_valid_norm, X_test_norm, _ = normalize_data(X_train, X_valid, X_test)
y_train_oh, y_valid_oh, y_test_oh, _ = encode_labels(y_train, y_valid, y_test)

# Train MLP
mlp_results = train_mlp(X_train_norm, y_train_oh, X_valid_norm, y_valid_oh, 
                        X_test_norm, y_test_oh)

# Train SVM
svm_results = train_svm(X_train_norm, y_train, X_valid_norm, y_valid, 
                        X_test_norm, y_test)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**: Reduce `max_communities` or `iterations` in config
2. **CUDA Out of Memory**: BIGCLAM will automatically fall back to CPU
3. **File Not Found**: Check that your data file path in `config.yaml` is correct
4. **Import Errors**: Ensure you've activated the virtual environment and installed dependencies

### Getting Help

- Check the main `README.md` for detailed documentation
- Review the configuration file comments
- Examine the example notebook in `notebooks/`

## Next Steps

- Explore the Jupyter notebook in `notebooks/` for interactive analysis
- Modify hyperparameters in `config/config.yaml` for your specific use case
- Check out the visualization functions in `src/evaluation/`

## Citation

If you use this code, please cite it appropriately (see `CITATION.md`).

