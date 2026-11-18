import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import silhouette_samples

from . import utils


def fisher_discriminant_ratio(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    class_means = utils.compute_class_means(feature_col, labels)
    class_variances = utils.compute_class_variances(feature_col, labels)
    overall_mean = np.mean(feature_col)
    unique_labels = np.unique(labels)
    between_var = 0.0
    for class_id in unique_labels:
        mask = labels == class_id
        n_class = np.sum(mask)
        between_var += n_class * (class_means[class_id] - overall_mean) ** 2
    between_var /= len(labels)
    within_var = np.mean(list(class_variances.values()))
    if within_var < 1e-10:
        return 0.0
    return float(between_var / within_var)


def kullback_leibler_divergence(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    overall_hist_raw, bin_edges = np.histogram(feature_col, bins=n_bins)
    overall_hist: NDArray[np.float64] = overall_hist_raw.astype(np.float64)
    overall_hist += 1e-10
    overall_hist /= overall_hist.sum()
    class_samples = utils.get_class_samples(feature_col, labels)
    unique_labels = np.unique(labels)
    kl_divergences: list[float] = []
    for class_id in unique_labels:
        class_data = class_samples[class_id]
        class_hist_raw, _ = np.histogram(class_data, bins=bin_edges)
        class_hist: NDArray[np.float64] = class_hist_raw.astype(np.float64)
        class_hist += 1e-10
        class_hist /= class_hist.sum()
        kl = np.sum(class_hist * np.log(class_hist / overall_hist))
        kl_divergences.append(float(kl))
    return float(np.mean(kl_divergences))


def mutual_information_score(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    feature_entropy = utils.compute_feature_entropy(feature_col, n_bins)
    conditional_entropy = utils.compute_conditional_entropy(feature_col, labels, n_bins)
    mi = feature_entropy - conditional_entropy
    return float(max(0.0, mi))


def silhouette_score_feature(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    X = feature_col.reshape(-1, 1)
    silhouette_vals = silhouette_samples(X, labels)
    return float(np.mean(np.asarray(silhouette_vals)))


def class_separability_index(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    class_means = utils.compute_class_means(feature_col, labels)
    unique_labels = sorted(np.unique(labels))
    if len(unique_labels) < 2:
        return 0.0
    inter_class_distances: list[float] = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1 :]:
            dist = abs(class_means[label1] - class_means[label2])
            inter_class_distances.append(dist)
    intra_class_distances: list[float] = []
    class_samples = utils.get_class_samples(feature_col, labels)
    for class_id in unique_labels:
        class_data = class_samples[class_id]
        if len(class_data) > 1:
            mean = class_means[class_id]
            distances = np.abs(class_data - mean)
            intra_class_distances.extend(distances)
    if len(intra_class_distances) == 0 or np.mean(intra_class_distances) < 1e-10:
        return 0.0
    inter_mean = np.mean(inter_class_distances) if inter_class_distances else 0.0
    intra_mean = np.mean(intra_class_distances)
    return float(inter_mean / intra_mean)


def class_overlap_ratio(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    class_samples = utils.get_class_samples(feature_col, labels)
    unique_labels = np.unique(labels)
    overlapping_count = 0.0
    total_count = len(feature_col)
    for class_id in unique_labels:
        class_data = class_samples[class_id]
        class_min = np.min(class_data)
        class_max = np.max(class_data)
        for other_class_id in unique_labels:
            if other_class_id == class_id:
                continue
            other_data = class_samples[other_class_id]
            overlap = np.sum((other_data >= class_min) & (other_data <= class_max))
            overlapping_count += overlap
    overlap_ratio = overlapping_count / (total_count * (len(unique_labels) - 1))
    return float(np.clip(overlap_ratio, 0.0, 1.0))


def compute_all_quality_metrics(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> dict[str, float]:
    return {
        "fisher_discriminant_ratio": fisher_discriminant_ratio(feature_col, labels),
        "kullback_leibler_divergence": kullback_leibler_divergence(feature_col, labels),
        "mutual_information": mutual_information_score(feature_col, labels),
        "silhouette_score": silhouette_score_feature(feature_col, labels),
        "class_separability_index": class_separability_index(feature_col, labels),
        "class_overlap_ratio": class_overlap_ratio(feature_col, labels),
    }


def compute_all_features_metrics(
    feature_matrix: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> dict[int, dict[str, float]]:
    metrics = {}
    n_features = feature_matrix.shape[1]
    for feature_idx in range(n_features):
        feature_col = feature_matrix[:, feature_idx]
        metrics[feature_idx] = compute_all_quality_metrics(feature_col, labels)
    return metrics
