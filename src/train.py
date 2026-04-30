import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
from pathlib import Path
import yaml
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Add src to path so we can import local modules
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality, select_columns, decode_target, encode_target, norm_preprocessor
from evaluation import evaluate_model, check_thresholds, save_metrics, save_model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
CONFIG = load_config(DEFAULT_CONFIG_PATH)

def load_data(relative_path):
    full_path = PROJECT_ROOT / relative_path
    print(f"Loading data from {full_path}...")
    return pd.read_csv(full_path)

def train_model(config=None):
    """Full training pipeline. Returns metrics dictionary."""
    if config is None:
        config = CONFIG

    # Load
    df = load_data(config["data_url"])

    # Drop ID column
    df = select_columns(df, columns=config["numeric_columns"] + config["categorical_columns"] + [config["target"]])

    # Validate
    required = config["numeric_columns"] + config["categorical_columns"]
    validate_dataframe(df, required, config["target"])

    # Data quality check
    quality = check_data_quality(df, config["numeric_columns"])
    print(f"Data quality: {quality['total_nulls']} nulls, {quality['duplicate_rows']} duplicates")

    # Clean
    df = clean_data(df, config["numeric_columns"], config["categorical_columns"])

    # Encode
    df = encode_categoricals(df, config["categorical_columns"])
    df = encode_target(df, config["target"])

    # Normalize/standardize
    preprocessor = norm_preprocessor(config["minmax_columns"], config["standard_columns"])

    df['age'] = df['age'].astype(int)
    df['notifications_per_day'] = df['notifications_per_day'].astype(int)
    df['app_opens_per_day'] = df['app_opens_per_day'].astype(int)

    # Split
    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    mlflow.set_experiment("smartphone-stress-impact-prediction")

    # Start MLflow
    mlflow.set_experiment("smartphone-stress-impact-prediction")
    with mlflow.start_run():
        mlflow.log_param("model", config["model_type"])
        mlflow.log_param("random_state", config["random_state"])

        if config["model_type"] == "Logistic Regression":
            mlflow.log_param("C", config["C"])
            mlflow.log_param("max_iter", str(config["max_iter"]))
        elif config["model_type"] == "RandomForest":
            mlflow.log_param("n_estimators", str(config["n_estimators"]))
            mlflow.log_param("max_depth", str(config["max_depth"]))
        elif config["model_type"] == "Gradient Boosting":
            mlflow.log_param("learning_rate", str(config["learning_rate"]))
            mlflow.log_param("max_depth", str(config["max_depth"]))
            mlflow.log_param("max_iter", str(config["max_iter"]))

        # Train
        print("Training random forest...")
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=config["random_state"]
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        metrics["n_features"] = X_train.shape[1]

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        # ── Log the trained model as an artifact ──
        mlflow.sklearn.log_model(model, "model")

        # Check thresholds
        check_thresholds(metrics, {"accuracy": config["min_accuracy"], "f1_score": config["min_f1"]})

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        save_model(model, model_path)

        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        metrics_path = "metrics/results.json"
        save_metrics(metrics, metrics_path)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    metrics = train_model(config)

    # Exit with error if thresholds not met
    if metrics["accuracy"] < config["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold")
        sys.exit(1)
    if metrics["f1_score"] < config["min_f1"]:
        print(f"\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")
