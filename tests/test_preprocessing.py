import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data_preprocessing import clean_data, encode_categoricals, validate_dataframe

def test_clean_data_handles_missing_values():
    """Verify clean_data handles numeric and categorical missing values."""
    df = pd.DataFrame({
        "age": [20, 30, np.nan, 40, 50],
        "notifications_per_day": [1, 2, 3, 4, 5],
        "app_opens_per_day": [5, 4, 3, 2, 1],
        "score": [1.0, np.nan, 3.0, 4.0, 5.0],
        "category": ["A", "B", np.nan, "A", "B"]
    })

    result = clean_data(df, numeric_columns=["score"], categorical_columns=["category"])
    
    assert result["score"].isna().sum() == 0, "Numeric missing values should be filled"
    assert result["score"].iloc[1] == 3.5, "Missing numeric should be filled with median"
    
    assert result["category"].isna().sum() == 0, "Categorical missing values should be filled"
    # Given the clean_data logic, categorical missing is filled with "UnKnown"
    assert "UnKnown" in result["category"].values

def test_encode_categoricals_correctly():
    """Verify encode_categoricals properly one-hot encodes."""
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red"]
    })
    
    result = encode_categoricals(df, columns=["color"])
    assert "color_red" in result.columns or "color_green" in result.columns or "color_blue" in result.columns
    assert len(result.columns) >= 2, "Should create dummy columns for the colors (drop_first=True yields n-1)"

def test_clean_data_does_not_modify_original():
    """Verify clean_data does not modify the original dataframe inplace."""
    df = pd.DataFrame({
        "age": [20.0, 30.0],
        "notifications_per_day": [1, 2],
        "app_opens_per_day": [5, 4],
        "score": [1.0, np.nan],
    })
    
    df_copy = df.copy()
    result = clean_data(df, numeric_columns=["score"], categorical_columns=[])
    
    # Original should still have NaN
    assert pd.isna(df.iloc[1]["score"])
    pd.testing.assert_frame_equal(df, df_copy)

def test_validate_dataframe_raises_missing_columns():
    """Verify validate_dataframe raises ValueError for missing columns."""
    df = pd.DataFrame({"existing_col": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, required_columns=["missing_col"], target_column="target")

def test_validate_dataframe_raises_missing_target():
    """Verify validate_dataframe raises ValueError when target is missing."""
    df = pd.DataFrame({"col1": [1, 2]})
    with pytest.raises(ValueError, match="Target column .* not found"):
        validate_dataframe(df, required_columns=["col1"], target_column="target")

def test_validate_dataframe_raises_empty():
    """Verify validate_dataframe raises ValueError for empty dataframe."""
    df = pd.DataFrame(columns=["col1", "target"])
    with pytest.raises(ValueError, match="Dataframe is empty"):
        validate_dataframe(df, required_columns=["col1"], target_column="target")