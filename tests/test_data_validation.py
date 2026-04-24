import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train import CONFIG

@pytest.fixture(scope="module")
def data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/main/Smartphone_Usage_And_Addiction_Analysis_3600_Rows.csv"))
    df = pd.read_csv(data_path)
    return df

def test_data_expected_columns(data):
    """Verify the expected columns are present in the dataset."""
    expected_cols = set(CONFIG["numeric_columns"] + CONFIG["categorical_columns"] + [CONFIG["target"]])
    missing_cols = expected_cols - set(data.columns)
    assert not missing_cols, f"Missing expected columns in dataset: {missing_cols}"

def test_data_target_expected_values(data):
    """Verify target variable contains only expected values."""
    target_col = CONFIG["target"]
    unique_vals = set(data[target_col].dropna().unique())
    expected_vals = {"Low", "Medium", "High"}
    
    # We verify it's a subset or exact match to expected values for classification
    assert unique_vals.issubset(expected_vals), f"Unexpected values in target column: {unique_vals - expected_vals}"

def test_data_numeric_ranges(data):
    """Verify that numeric features are within expected reasonable ranges."""
    # Age shouldn't be negative, and max age usually < 120
    assert data["age"].min() >= 0
    assert data["age"].max() < 120
    
    # Hours cannot be negative
    hour_cols = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'weekend_screen_time']
    for col in hour_cols:
        assert data[col].min() >= 0, f"{col} has negative values"
        
    # Same for counts
    count_cols = ['notifications_per_day', 'app_opens_per_day']
    for col in count_cols:
        assert data[col].min() >= 0, f"{col} has negative values"
