"""
Model training module.
Implements Logistic Regression (baseline) and Random Forest (advanced) models.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class BaselineModel:
    """Logistic Regression baseline model."""

    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        self.name = "Logistic Regression"

    def train(self, X_train, y_train):
        """Train the model."""
        print(f"\n[Training] {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"[SUCCESS] {self.name} training complete")
        return self.model

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Generate prediction probabilities."""
        return self.model.predict_proba(X)


class AdvancedModel:
    """Random Forest advanced model."""

    def __init__(self, random_state=42, n_estimators=100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        self.name = "Random Forest Classifier"

    def train(self, X_train, y_train):
        """Train the model."""
        print(f"\n[Training] {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"[SUCCESS] {self.name} training complete")
        return self.model

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Generate prediction probabilities."""
        return self.model.predict_proba(X)


def save_model(model_obj, filepath):
    """
    Save trained model to pickle file.

    Parameters:
    -----------
    model_obj : BaselineModel or AdvancedModel
        Model object to save
    filepath : str or Path
        Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model_obj.model, f)

    print(f"[SUCCESS] Model saved: {filepath}")


def load_model(filepath):
    """
    Load trained model from pickle file.

    Parameters:
    -----------
    filepath : str or Path
        Path to the saved model

    Returns:
    --------
    sklearn model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    from preprocessing import preprocess_dataset
    from data_loader import load_dataset

    data_path = Path(__file__).parent.parent / "archive" / "ai4i2020.csv"

    if data_path.exists():
        # Load and preprocess data
        df = load_dataset(data_path)
        X_train, X_test, y_train, y_test, scaler, info = preprocess_dataset(df)

        # Train baseline model
        baseline = BaselineModel()
        baseline.train(X_train, y_train)
        save_model(baseline, Path(__file__).parent.parent /
                   "models" / "baseline_lr.pkl")

        # Train advanced model
        advanced = AdvancedModel()
        advanced.train(X_train, y_train)
        save_model(advanced, Path(__file__).parent.parent /
                   "models" / "advanced_rf.pkl")

        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETE")
        print("="*80)
    else:
        print(f"Dataset not found at {data_path}")
