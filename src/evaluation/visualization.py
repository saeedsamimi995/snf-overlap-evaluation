"""Visualization functions for evaluation results."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os


def plot_confusion_matrix(cm_mean, cm_std=None, title='Confusion Matrix', 
                         cmap=plt.cm.Blues, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm_mean (np.ndarray): Mean confusion matrix.
        cm_std (np.ndarray, optional): Standard deviation of confusion matrix.
        title (str): Plot title.
        cmap: Matplotlib colormap.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_mean, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(cm_mean))
    plt.xticks(tick_marks, tick_marks, rotation=45, fontsize=12)
    plt.yticks(tick_marks, tick_marks, fontsize=12)
    
    # Add text annotations
    for i in range(len(cm_mean)):
        for j in range(len(cm_mean)):
            plt.text(
                j, i, f'{int(cm_mean[i, j])}',
                horizontalalignment="center",
                color="white" if cm_mean[i, j] > cm_mean.max() / 2 else "black",
                fontsize=12
            )
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_pred, title='ROC Curve', save_path=None):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true (np.ndarray): True labels (one-hot encoded).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """
    # Convert torch tensor to numpy if needed
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    elif hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    output_size = y_true.shape[1]
    
    for i in range(output_size):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'Class {i + 1} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_cm_stats(cms):
    """
    Calculate mean and standard deviation of confusion matrices.
    
    Args:
        cms (list): List of confusion matrices.
        
    Returns:
        tuple: (mean, std) confusion matrices.
    """
    cms = np.array(cms)
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)
    return cm_mean, cm_std


def save_mlp_results(results, results_dir='results', y_train_onehot=None, 
                    y_valid_onehot=None, y_test_onehot=None):
    """
    Save MLP evaluation results including confusion matrices and ROC curves.
    
    Args:
        results (dict): Results dictionary from train_mlp function.
        results_dir (str): Directory to save results.
        y_train_onehot, y_valid_onehot, y_test_onehot: One-hot encoded labels.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate statistics
    train_cm_mean, train_cm_std = calculate_cm_stats(results['train_cms'])
    valid_cm_mean, valid_cm_std = calculate_cm_stats(results['valid_cms'])
    test_cm_mean, test_cm_std = calculate_cm_stats(results['test_cms'])
    
    # Save confusion matrices
    plot_confusion_matrix(
        train_cm_mean, train_cm_std, 
        title='Confusion Matrix (Train)',
        save_path=f'{results_dir}/MLP_train_conf_matrix.svg'
    )
    plot_confusion_matrix(
        valid_cm_mean, valid_cm_std,
        title='Confusion Matrix (Valid)',
        save_path=f'{results_dir}/MLP_valid_conf_matrix.svg'
    )
    plot_confusion_matrix(
        test_cm_mean, test_cm_std,
        title='Confusion Matrix (Test)',
        save_path=f'{results_dir}/MLP_test_conf_matrix.svg'
    )
    
    # Save ROC curves
    if y_train_onehot is not None:
        plot_roc_curve(
            y_train_onehot, results['best_train_outputs'],
            title='ROC Curve (Train)',
            save_path=f'{results_dir}/MLP_train_roc.svg'
        )
    if y_valid_onehot is not None:
        plot_roc_curve(
            y_valid_onehot, results['best_valid_outputs'],
            title='ROC Curve (Valid)',
            save_path=f'{results_dir}/MLP_valid_roc.svg'
        )
    if y_test_onehot is not None:
        plot_roc_curve(
            y_test_onehot, results['best_test_outputs'],
            title='ROC Curve (Test)',
            save_path=f'{results_dir}/MLP_test_roc.svg'
        )
    
    # Print metrics
    print(f"MLP Train Metrics: {results['best_train_metrics']}")
    print(f"MLP Valid Metrics: {results['best_valid_metrics']}")
    print(f"MLP Test Metrics: {results['best_test_metrics']}")


def save_svm_results(results, results_dir='results'):
    """
    Save SVM evaluation results including confusion matrices and ROC curves.
    
    Args:
        results (dict): Results dictionary from train_svm function.
        results_dir (str): Directory to save results.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    def plot_svm_conf_matrix(conf_matrices, filename, title):
        mean_conf_matrix = np.mean(conf_matrices, axis=0)
        plot_confusion_matrix(mean_conf_matrix, title=title, save_path=filename)
    
    def plot_svm_roc_curve(roc_curve_data, auc_data, filename):
        plt.figure(figsize=(10, 8))
        
        for class_idx in range(len(roc_curve_data[0])):
            mean_fpr = np.mean([curve[class_idx][0] for curve in roc_curve_data], axis=0)
            mean_tpr = np.mean([curve[class_idx][1] for curve in roc_curve_data], axis=0)
            mean_auc = np.mean([auc[class_idx] for auc in auc_data])
            plt.plot(mean_fpr, mean_tpr, label=f'Class {class_idx + 1} (AUC = {mean_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()
    
    # Plot and save confusion matrices
    plot_svm_conf_matrix(
        results['conf_matrices_train'],
        f'{results_dir}/SVM_train_conf_matrix.svg',
        'Confusion Matrix (Train)'
    )
    plot_svm_conf_matrix(
        results['conf_matrices_valid'],
        f'{results_dir}/SVM_valid_conf_matrix.svg',
        'Confusion Matrix (Valid)'
    )
    plot_svm_conf_matrix(
        results['conf_matrices_test'],
        f'{results_dir}/SVM_test_conf_matrix.svg',
        'Confusion Matrix (Test)'
    )
    
    # Plot and save ROC curves
    plot_svm_roc_curve(
        results['roc_curves_train'],
        results['auc_train'],
        f'{results_dir}/SVM_train_roc_curve.svg'
    )
    plot_svm_roc_curve(
        results['roc_curves_valid'],
        results['auc_valid'],
        f'{results_dir}/SVM_valid_roc_curve.svg'
    )
    plot_svm_roc_curve(
        results['roc_curves_test'],
        results['auc_test'],
        f'{results_dir}/SVM_test_roc_curve.svg'
    )
    
    # Print metrics
    print(f"SVM Train Metrics: {results['train_metrics']}")
    print(f"SVM Valid Metrics: {results['valid_metrics']}")
    print(f"SVM Test Metrics: {results['test_metrics']}")

