import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    if len(y_true) == 0:
        return np.array([])
    return confusion_matrix(y_true, y_pred)
