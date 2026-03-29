"""
Preprocessing pipeline for AI4I 2020 dataset.
Handles normalization, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(df, target_col='Machine failure', test_size=0.2, random_state=42):
    """
    Preprocess the dataset: handle missing values, encode categoricals, normalize numerics.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    target_col : str
        Name of the target column
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple of (X_train, X_test, y_train, y_test, scaler, encoder_info)
    """

    # Create a copy to avoid modifying original
    df = df.copy()

    print("="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)

    # Step 1: Drop identifiers and target leakage columns
    print("\n[Step 1] Dropping identifiers and target leakage columns...")
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=columns_to_drop)
    print(f"Dropped columns: {columns_to_drop}")
    print(f"Remaining columns: {df.shape[1]}")

    # Step 2: Separate target and features
    print("\n[Step 2] Separating target and features...")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"Target distribution:\n{y.value_counts()}")

    # Step 3: Handle categorical features (Product Type)
    print("\n[Step 3] Encoding categorical features...")
    if 'Type' in X.columns:
        X_encoded = pd.get_dummies(
            X, columns=['Type'], prefix='Type', drop_first=False)
        print(f"One-hot encoded 'Type' column")
        print(f"New feature count: {X_encoded.shape[1]}")
    else:
        X_encoded = X.copy()

    # Step 4: Identify numeric columns and normalize
    print("\n[Step 4] Normalizing numeric features...")
    numeric_cols = X_encoded.select_dtypes(
        include=[np.number]).columns.tolist()
    print(f"Numeric columns to scale: {numeric_cols}")

    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
    print(f"Applied StandardScaler to {len(numeric_cols)} numeric features")

    # Step 5: Train/test split
    print("\n[Step 5] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(
        f"Train set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Train target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)

    return X_train, X_test, y_train, y_test, scaler, {
        'numeric_cols': numeric_cols,
        'encoded_cols': list(X_encoded.columns),
        'dropped_cols': columns_to_drop
    }


def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Save processed train and test sets to CSV files.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Feature matrices
    y_train, y_test : pd.Series
        Target vectors
    output_dir : str or Path
        Directory to save processed data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine X and y for easier storage
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"\n[SUCCESS] Processed train data saved: {train_path}")
    print(f"[SUCCESS] Processed test data saved: {test_path}")


if __name__ == "__main__":
    from data_loader import load_dataset

    data_path = Path(__file__).parent.parent / "archive" / "ai4i2020.csv"

    if data_path.exists():
        df = load_dataset(data_path)
        X_train, X_test, y_train, y_test, scaler, info = preprocess_dataset(df)

        output_dir = Path(__file__).parent.parent / "data" / "processed"
        save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    else:
        print(f"Dataset not found at {data_path}")
