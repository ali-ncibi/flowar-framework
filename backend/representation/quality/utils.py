from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


def encode_labels(
    labels: NDArray[np.int_] | NDArray[np.str_],
) -> tuple[NDArray[np.int_], dict[int, Any]]:
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_to_int[label] for label in labels])
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    return encoded, int_to_label


def normalize_features(feature_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    scaler = StandardScaler()
    return scaler.fit_transform(feature_matrix)


def get_class_samples(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> dict[int, NDArray[np.float64]]:
    class_samples = {}
    for class_id in np.unique(labels):
        mask = labels == class_id
        class_samples[class_id] = feature_col[mask]
    return class_samples


def compute_class_means(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> dict[int, float]:
    class_samples = get_class_samples(feature_col, labels)
    return {
        class_id: float(np.mean(samples)) for class_id, samples in class_samples.items()
    }


def compute_class_variances(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> dict[int, float]:
    class_samples = get_class_samples(feature_col, labels)
    return {
        class_id: float(np.var(samples)) for class_id, samples in class_samples.items()
    }


def compute_feature_entropy(
    feature_col: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    hist, _ = np.histogram(feature_col, bins=n_bins)
    hist = hist[hist > 0]
    probabilities = hist / hist.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def compute_conditional_entropy(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    class_samples = get_class_samples(feature_col, labels)
    unique_labels = np.unique(labels)
    n_samples = len(labels)
    conditional_entropy = 0.0
    for class_id in unique_labels:
        class_data = class_samples[class_id]
        p_class = len(class_data) / n_samples
        hist, _ = np.histogram(class_data, bins=n_bins)
        hist = hist[hist > 0]
        probabilities = hist / hist.sum()
        class_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        conditional_entropy += p_class * class_entropy
    return conditional_entropy


def compute_label_entropy(labels: NDArray[np.int_]) -> float:
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy
