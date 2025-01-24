import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Evaluate model performance and visualize results

    Parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities
        model_name (str): Name of the model for reporting

    Returns:
        tuple: Precision, recall, F1 score, and predictions
    """
    # Ensure evaluationPlots directory exists
    os.makedirs('evaluationPlots', exist_ok=True)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save plot
    plot_path = os.path.join('evaluationPlots', f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()

    return precision, recall, f1, y_pred_proba


def plot_roc_curves(y_test, predictions_dict):
    """
    Plot ROC curves for multiple models

    Parameters:
        y_test (array): True test labels
        predictions_dict (dict): Dictionary of model predictions
    """
    # Ensure evaluationPlots directory exists
    os.makedirs('evaluationPlots', exist_ok=True)

    plt.figure(figsize=(10, 8))

    for model_name, y_pred_proba in predictions_dict.items():
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend(loc="lower right")

    # Save plot
    plot_path = os.path.join('evaluationPlots', 'roc_curves.png')
    plt.savefig(plot_path)
    plt.close()