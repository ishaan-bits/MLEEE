"""
Model evaluation module.
Computes metrics tailored for imbalanced classification.
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Comprehensive evaluation for imbalanced classification.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (for ROC-AUC)
    model_name : str
        Name of the model

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC if probabilities available
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['roc_auc'] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, output_dict=True)

    return metrics


def print_evaluation_results(metrics_dict):
    """
    Pretty print evaluation results.

    Parameters:
    -----------
    metrics_dict : dict
        Metrics dictionary from evaluate_model()
    """
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {metrics_dict['model']}")
    print("="*80)

    print(f"\nAccuracy: {metrics_dict['accuracy']:.4f}")
    print(
        f"Precision: {metrics_dict['precision']:.4f}  (of predicted positives, how many are correct)")
    print(
        f"Recall: {metrics_dict['recall']:.4f}   (of actual positives, how many we caught)")
    print(
        f"F1-Score: {metrics_dict['f1_score']:.4f}   (harmonic mean of precision & recall)")

    if metrics_dict.get('roc_auc'):
        print(f"ROC-AUC: {metrics_dict['roc_auc']:.4f}")

    cm = np.array(metrics_dict['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"                Predicted No-Fail  Predicted Fail")
    print(f"Actual No-Fail  {cm[0, 0]:16d}  {cm[0, 1]:14d}")
    print(f"Actual Fail     {cm[1, 0]:16d}  {cm[1, 1]:14d}")

    print(f"\nTrue Negatives (TN): {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP): {cm[1, 1]}")


def compare_models(metrics_baseline, metrics_advanced):
    """
    Compare two models side-by-side.

    Parameters:
    -----------
    metrics_baseline : dict
        Baseline model metrics
    metrics_advanced : dict
        Advanced model metrics
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    metrics_to_compare = ['accuracy', 'precision',
                          'recall', 'f1_score', 'roc_auc']

    comparison_data = []
    for metric in metrics_to_compare:
        baseline_val = metrics_baseline.get(metric, 'N/A')
        advanced_val = metrics_advanced.get(metric, 'N/A')

        if isinstance(baseline_val, float) and isinstance(advanced_val, float):
            diff = advanced_val - baseline_val
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = "—"

        comparison_data.append({
            'Metric': metric.upper(),
            'Baseline (LR)': f"{baseline_val:.4f}" if isinstance(baseline_val, float) else baseline_val,
            'Advanced (RF)': f"{advanced_val:.4f}" if isinstance(advanced_val, float) else advanced_val,
            'Difference': diff_str
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    print("\nINTERPRETATION:")
    print("─" * 80)
    print("Precision: Of predicted failures, how many are actual failures?")
    print("Recall: Of actual failures, how many do we detect?")
    print("F1-Score: Balanced metric (prefer this for imbalanced data)")
    print("ROC-AUC: Overall ranking quality (1.0 = perfect, 0.5 = random)")


def save_results_json(all_metrics, output_path):
    """
    Save all metrics to JSON file.

    Parameters:
    -----------
    all_metrics : dict
        Dictionary of all metrics
    output_path : str or Path
        Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[SUCCESS] Results saved: {output_path}")


if __name__ == "__main__":
    from preprocessing import preprocess_dataset
    from models import BaselineModel, AdvancedModel, load_model
    from data_loader import load_dataset

    data_path = Path(__file__).parent.parent / "archive" / "ai4i2020.csv"

    if data_path.exists():
        # Load and preprocess data
        df = load_dataset(data_path)
        X_train, X_test, y_train, y_test, scaler, info = preprocess_dataset(df)

        # Load trained models
        baseline_path = Path(__file__).parent.parent / \
            "models" / "baseline_lr.pkl"
        advanced_path = Path(__file__).parent.parent / \
            "models" / "advanced_rf.pkl"

        if baseline_path.exists() and advanced_path.exists():
            baseline_model = load_model(baseline_path)
            advanced_model = load_model(advanced_path)

            # Evaluate both models
            y_pred_baseline = baseline_model.predict(X_test)
            y_pred_proba_baseline = baseline_model.predict_proba(X_test)
            metrics_baseline = evaluate_model(
                y_test, y_pred_baseline, y_pred_proba_baseline, "Logistic Regression")

            y_pred_advanced = advanced_model.predict(X_test)
            y_pred_proba_advanced = advanced_model.predict_proba(X_test)
            metrics_advanced = evaluate_model(
                y_test, y_pred_advanced, y_pred_proba_advanced, "Random Forest")

            # Print results
            print_evaluation_results(metrics_baseline)
            print_evaluation_results(metrics_advanced)
            compare_models(metrics_baseline, metrics_advanced)

            # Save results
            all_results = {
                'baseline': metrics_baseline,
                'advanced': metrics_advanced
            }
            save_results_json(all_results, Path(
                __file__).parent.parent / "results" / "metrics.json")
        else:
            print("Trained models not found. Run models.py first.")
    else:
        print(f"Dataset not found at {data_path}")
