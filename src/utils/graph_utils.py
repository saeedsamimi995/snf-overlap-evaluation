"""Graph construction utilities for BIGCLAM."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def create_adjacency_and_assignments(df, threshold=0.4):
    """
    Create adjacency matrix and community assignments from dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with features and optional community column.
        threshold (float): Similarity threshold for edge creation.
        
    Returns:
        tuple: (A, p2c) where A is the adjacency matrix and p2c is list of community sets.
    """
    label_column = None
    for candidate in ("Community", "PAM50", "label"):
        if candidate in df.columns:
            label_column = candidate
            break
    if label_column is None and len(df.columns):
        last = df.columns[-1]
        if not pd.api.types.is_numeric_dtype(df[last]):
            label_column = last

    features = df.drop(columns=[label_column]) if label_column is not None else df
    feature_values = features.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    N = feature_values.shape[0]
    similarity_matrix = cosine_similarity(feature_values)

    # Create adjacency matrix based on similarity threshold
    A = (similarity_matrix > threshold).astype(int) - np.eye(N).astype(int)

    if label_column is not None:
        community_assignments = []
        for value in df[label_column].tolist():
            try:
                community_assignments.append({int(value)})
            except (TypeError, ValueError):
                community_assignments.append({0})
    else:
        community_assignments = [{i} for i in range(N)]

    return A, community_assignments


def gen_json(A, p2c, F_argmax=None):
    """
    Generate JSON data for graph visualization.
    
    Args:
        A (np.ndarray or torch.Tensor): Adjacency matrix.
        p2c (list): List of community sets for each node.
        F_argmax (np.ndarray, optional): Assigned communities from BIGCLAM.
        
    Returns:
        dict: JSON-formatted graph data with nodes and links.
    """
    # Convert torch tensor to numpy if needed
    if hasattr(A, 'cpu'):
        A = A.cpu().numpy()
    
    N = A.shape[0]
    data = {'nodes': [], 'links': []}
    
    for i in range(N):
        grp = ''.join(map(str, sorted(p2c[i])))
        node = {'id': str(i), 'group': str(grp)}
        if F_argmax is not None:
            node.update({'assigned': str(F_argmax[i])})
        data['nodes'].append(node)
        
        friends = np.where(A[i])[0]
        for friend in friends:
            inter = 2 - len(p2c[i].intersection(p2c[friend]))
            data['links'].append({
                'source': str(i),
                'target': str(friend),
                'value': str(inter * 5 + 1),
                'distance': str(inter * 15 + 1)
            })
    
    return data


def save_graph_data(A, p2c, save_path, adj_filename='adj.npy', p2c_filename='p2c.pkl'):
    """
    Save adjacency matrix and community assignments.
    
    Args:
        A (np.ndarray): Adjacency matrix.
        p2c (list): Community assignments.
        save_path (str): Directory path to save files.
        adj_filename (str): Filename for adjacency matrix.
        p2c_filename (str): Filename for community assignments.
    """
    import os
    import pickle
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Make adjacency matrix symmetric
    A_symmetric = A + A.T
    
    # Save files
    np.save(os.path.join(save_path, adj_filename), A_symmetric)
    pickle.dump(p2c, open(os.path.join(save_path, p2c_filename), "wb"))
    
    print(f"Graph data saved to: {save_path}")

