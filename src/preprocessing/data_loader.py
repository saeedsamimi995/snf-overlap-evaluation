"""Data loading and preprocessing functions."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

LABEL_ROWS = {"PAM50", "COMMUNITY", "LABEL", "TARGET"}


def load_data(file_path):
    """Load a genes-by-samples CSV and return samples by genes.

    A trailing PAM50 or community row is removed from the feature matrix and
    returned on ``data.attrs['labels']``. It is not left inside ``X``.

    Args:
        file_path (str): Path to the CSV file containing gene expression data.

    Returns:
        tuple: (X, data) where X is samples by genes and data is genes by samples.
    """
    df = pd.read_csv(file_path, index_col=0)
    labels = None
    if len(df.index) and str(df.index[-1]).strip().upper() in LABEL_ROWS:
        labels = df.iloc[-1].copy()
        df = df.iloc[:-1]
    df = df.apply(pd.to_numeric, errors="coerce")
    data = df.copy()
    X = data.T
    X.index = X.index.astype(str)
    if labels is not None:
        data.attrs["labels"] = labels
        X.attrs["labels"] = labels
    return X, data


def apply_variance_threshold(X, data, threshold=13):
    """Apply an absolute variance threshold.

    The legacy classifier path still uses this cutoff. The Q1 study does not:
    a single absolute cutoff is not comparable across RNA-seq and microarray.

    Args:
        X (pd.DataFrame): Samples by genes.
        data (pd.DataFrame): Genes by samples, kept for callers that pass it.
        threshold (float): Variance threshold value.

    Returns:
        tuple: (X_reduced, selected_feature_names, df_selected_features)
    """
    del data  # Feature names now come from X. The label row is no longer in X.
    values = X.apply(pd.to_numeric, errors="coerce")
    filled = values.copy()
    for column in filled.columns:
        if filled[column].isna().all():
            filled[column] = 0.0
        else:
            filled[column] = filled[column].fillna(filled[column].mean())

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(filled.to_numpy(dtype=float))
    selected = selector.get_support(indices=True)
    selected_feature_names = values.columns.to_numpy()[selected]
    X_reduced = values.iloc[:, selected]
    df_selected_features = pd.DataFrame(
        {
            "Selected Feature Index": selected,
            "Feature Name": selected_feature_names,
        }
    )
    print("Original data shape:", X.shape)
    print("Reduced data shape:", X_reduced.shape)
    print(f"Selected {len(selected)} features out of {X.shape[1]}")
    return X_reduced, selected_feature_names, df_selected_features
