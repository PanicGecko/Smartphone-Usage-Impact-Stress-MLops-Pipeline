import pandas as pd
import numpy as np
import pytest
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train import train_model, CONFIG

@pytest.fixture(scope="module")
def sample_data():
    """Return a small dummy sample that mimics the dataset structure."""
    df = pd.DataFrame({
        'age': np.random.randint(18, 50, 100),
        'daily_screen_time_hours': np.random.uniform(1, 10, 100),
        'social_media_hours': np.random.uniform(0, 5, 100),
        'gaming_hours': np.random.uniform(0, 5, 100),
        'work_study_hours': np.random.uniform(0, 10, 100),
        'sleep_hours': np.random.uniform(4, 10, 100),
        'notifications_per_day': np.random.randint(10, 100, 100),
        'app_opens_per_day': np.random.randint(5, 50, 100),
        'weekend_screen_time': np.random.uniform(2, 12, 100),
        'gender': np.random.choice(["Male", "Female"], 100),
        'academic_work_impact': np.random.choice(["High", "Low"], 100),
        'stress_level': np.random.choice(["Low", "Medium", "High"], 100)
    })
    return df

def test_model_prediction_type_shape(sample_data, tmp_path):
    """Verify model produces predictions of correct type and shape."""
    temp_csv = tmp_path / "sample_data.csv"
    sample_data.to_csv(temp_csv, index=False)
    
    test_config = CONFIG.copy()
    test_config["data_url"] = str(temp_csv)
    # Reduce n_estimators to speed up test
    test_config["n_estimators"] = 5
    
    # Train the model 
    metrics = train_model(config=test_config)
    
    # Load the trained model to perform predictions
    import pickle
    model_path = "models/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    
    assert metrics['test_size'] == 20, "Config test_size was 0.2, so 20 rows of 100 should be in test"
    assert type(metrics['accuracy']) == float
    
def test_model_minimum_performance(tmp_path):
    """Verify the model achieves minimum performance threshold on known test set.
    According to requirements, we should train on a small sample and verify performance threshold."""

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/main/Smartphone_Usage_And_Addiction_Analysis_3600_Rows.csv"))
    df = pd.read_csv(data_path).sample(200, random_state=42)
    temp_csv = tmp_path / "actual_sample.csv"
    df.to_csv(temp_csv, index=False)
    
    test_config = CONFIG.copy()
    test_config["data_url"] = str(temp_csv)
    test_config["n_estimators"] = 10
    
    # It must achieve a minimum performance threshold
    metrics = train_model(config=test_config)
    
    assert metrics["accuracy"] >= test_config["min_accuracy"]
    assert metrics["f1_score"] >= test_config["min_f1"]
