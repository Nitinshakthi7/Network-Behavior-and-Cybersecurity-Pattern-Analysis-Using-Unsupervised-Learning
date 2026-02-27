"""
preprocessing.py
----------------
Data loading, cleaning, categorical encoding, and feature scaling.

Pipeline order:
    load_data → handle_missing → encode_categorical → scale_features
    (orchestrated by preprocess())

Labels are separated before any feature transformation and are never
passed to any model. They are returned separately for post-hoc validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the UNSW-NB15 dataset from a CSV file.

    Returns:
        df (DataFrame): Raw dataset
    """
    df = pd.read_csv(filepath)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical columns → column median (robust to outliers)
    - Categorical columns → column mode (most frequent value)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    """
    Label-encode all remaining object-type columns.

    Note on encoding choice:
        Label encoding is used instead of one-hot encoding to avoid
        dimensionality explosion in a 43-feature dataset. The trade-off
        (implied ordinality) is accepted for distance-based clustering.

    Parameters:
        df      : DataFrame
        exclude : List of column names to skip (e.g. already removed labels)
    """
    if exclude is None:
        exclude = []

    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                if c not in exclude]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df, cat_cols


def scale_features(X: np.ndarray) -> tuple:
    """
    Standardise all features to mean=0, std=1 using StandardScaler.

    Critical for K-Means and DBSCAN: both use Euclidean distance,
    which is dominated by unscaled high-magnitude features.

    Returns:
        X_scaled (ndarray): Standardised feature matrix
        scaler (StandardScaler): Fitted scaler (retained for inverse transform if needed)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def preprocess(df: pd.DataFrame, label_cols: list) -> tuple:
    """
    Full preprocessing pipeline.

    Steps:
        1. Handle missing values
        2. Separate label columns (returned separately, never used for training)
        3. Encode categorical features
        4. Scale all numerical features

    Parameters:
        df         : Raw DataFrame (may include labels)
        label_cols : Columns to exclude from features

    Returns:
        X_scaled      : Scaled feature matrix (ndarray)
        labels_df     : DataFrame of label columns (for evaluation only)
        feature_names : List of feature column names (matches X_scaled columns)
        scaler        : Fitted StandardScaler
    """
    df = handle_missing(df)

    # Separate labels — never seen by any model
    labels_df = df[[c for c in label_cols if c in df.columns]].copy()
    df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")

    # Encode categoricals
    df, _ = encode_categorical(df)

    feature_names = df.columns.tolist()
    X_scaled, scaler = scale_features(df.values)

    return X_scaled, labels_df, feature_names, scaler
