"""Evaluation metrics for labeled binary predictions."""
import numpy as np
def compute_academic_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute binary metrics from labeled evaluation samples; positive class is 1."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected equally sized, non-empty labeled and predicted samples.")
    if any(value not in (0, 1) for value in (*y_true, *y_pred)):
        raise ValueError("Labels and predictions must be binary (0 or 1).")
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {"sample_size": len(y_true), "class_distribution": {"positive": sum(y_true), "negative": len(y_true) - sum(y_true)}, "Accuracy": round((tp + tn) / len(y_true), 4), "Precision": round(precision, 4), "Recall": round(recall, 4), "F1_Score": round(f1, 4), "MCC": round(mcc, 4), "Confusion_Matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}}

