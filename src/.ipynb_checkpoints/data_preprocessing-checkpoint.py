import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def validate_dataframe(df, required_columns, target_column):
    """Check that a dataframe meets basic requirements."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    return True

def clean_data(df, numeric_columns, categorical_columns):
    """Clean a dataframe by handling missing values and encoding categoricals."""
    df = df.copy()
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df['age'] = df['age'].astype(int)
    df['notifications_per_day'] = df['notifications_per_day'].astype(int)
    df['app_opens_per_day'] = df['app_opens_per_day'].astype(int)

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("UnKnown")

    return df

def encode_categoricals(df, columns):
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df

def preprocessor(minmax_cols, standard_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("minmax", MinMaxScaler(), minmax_cols),
            ("standard", StandardScaler(), standard_cols)
        ],
        remainder="passthrough"
    )
    return preprocessor
    

def check_data_quality(df, numeric_columns):
    """Return a dictionary of data quality metrics."""
    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report