import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }
    return metrics
