import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath):
    """
    Load the UNSW-NB15 dataset from a CSV file.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    - Numerical columns: fill with column median
    - Categorical columns: fill with column mode
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[INFO] Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
    return df


def encode_categorical(df, exclude_cols=None):
    """
    Label-encode all categorical (object-type) columns.
    exclude_cols: list of column names to skip (e.g. label columns)
    """
    if exclude_cols is None:
        exclude_cols = []

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"[INFO] Encoded categorical columns: {cat_cols}")
    return df


def scale_features(X):
    """
    Apply StandardScaler to normalize all feature values.
    Returns the scaled numpy array and the fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[INFO] Feature scaling applied. Shape: {X_scaled.shape}")
    return X_scaled, scaler


def preprocess(df, label_cols=None):
    """
    Full preprocessing pipeline:
    1. Handle missing values
    2. Separate and drop label columns
    3. Encode categorical features
    4. Scale numerical features

    Parameters:
        df         : Raw DataFrame
        label_cols : List of label/target columns to exclude from features

    Returns:
        X_scaled      : Scaled feature matrix (numpy array)
        labels_df     : DataFrame of label columns (for evaluation only)
        feature_names : List of feature column names
    """
    if label_cols is None:
        label_cols = []

    # Step 1 – Handle missing values
    df = handle_missing_values(df)

    # Step 2 – Separate labels (kept for evaluation only, not used in training)
    labels_df = df[label_cols].copy() if label_cols else pd.DataFrame()
    df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")

    # Step 3 – Encode categorical columns
    df = encode_categorical(df)

    # Step 4 – Scale features
    feature_names = df.columns.tolist()
    X_scaled, scaler = scale_features(df.values)

    return X_scaled, labels_df, feature_names
