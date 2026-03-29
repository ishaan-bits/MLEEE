"""
Data loading module for AI4I 2020 Predictive Maintenance Dataset.
Loads CSV and performs initial exploratory inspection.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(filepath):
    """
    Load the AI4I 2020 dataset from CSV.

    Parameters:
    -----------
    filepath : str or Path
        Path to the ai4i2020.csv file

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    return df


def explore_dataset(df):
    """
    Print initial dataset exploration statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to explore
    """
    print("="*80)
    print("DATASET EXPLORATION")
    print("="*80)

    print(f"\nDataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "="*80)
    print("DATA TYPES")
    print("="*80)
    print(df.dtypes)

    print("\n" + "="*80)
    print("MISSING VALUES")
    print("="*80)
    print(df.isnull().sum())

    print("\n" + "="*80)
    print("FIRST 5 ROWS")
    print("="*80)
    print(df.head())

    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(df.describe())

    print("\n" + "="*80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*80)
    if 'Machine failure' in df.columns:
        counts = df['Machine failure'].value_counts()
        percentages = df['Machine failure'].value_counts(normalize=True) * 100
        print(f"\nClass Distribution:")
        for label in counts.index:
            print(
                f"  Class {label}: {counts[label]} samples ({percentages[label]:.2f}%)")

    print("\n" + "="*80)
    print("COLUMN NAMES")
    print("="*80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    return df


if __name__ == "__main__":
    # Load dataset from archive folder
    data_path = Path(__file__).parent.parent / "archive" / "ai4i2020.csv"

    if data_path.exists():
        df = load_dataset(data_path)
        explore_dataset(df)
    else:
        print(f"Dataset not found at {data_path}")
