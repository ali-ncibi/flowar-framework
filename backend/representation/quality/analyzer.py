from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from . import metrics, utils, visualization


class FeatureQualityAnalyzer:
    def __init__(
        self,
        feature_matrix: NDArray[np.float64],
        labels: NDArray[np.int_] | NDArray[np.str_],
        feature_names: list[str] | None = None,
        label_mapping: dict[int, Any] | None = None,
    ):
        self.feature_matrix = feature_matrix
        self.n_samples, self.n_features = feature_matrix.shape
        self.label_mapping_auto: dict[int, Any] | None
        if labels.dtype.kind in ["U", "O"]:
            self.labels, self.label_mapping_auto = utils.encode_labels(labels)
        else:
            self.labels = labels.astype(np.int_)
            self.label_mapping_auto = None
        self.label_mapping = label_mapping or self.label_mapping_auto
        self.feature_names = feature_names or [
            f"Feature_{i}" for i in range(self.n_features)
        ]
        self.all_metrics = metrics.compute_all_features_metrics(
            self.feature_matrix, self.labels
        )

    def compute_all_metrics(self) -> dict[int, dict[str, float]]:
        return self.all_metrics

    def rank_features(
        self,
        metric_name: str = "fisher_discriminant_ratio",
        ascending: bool = False,
    ) -> list[tuple[int, float]]:
        ranked = [
            (f_idx, self.all_metrics[f_idx].get(metric_name, 0.0))
            for f_idx in range(self.n_features)
        ]
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked

    def get_top_features(
        self,
        metric_name: str = "fisher_discriminant_ratio",
        top_k: int = 10,
    ) -> list[int]:
        ranked = self.rank_features(metric_name, ascending=False)
        return [f_idx for f_idx, _ in ranked[:top_k]]

    def get_feature_summary(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": len(np.unique(self.labels)),
            "feature_names": self.feature_names,
            "label_mapping": self.label_mapping,
        }

    def plot_feature_quality_bars(self, feature_idx: int) -> go.Figure:
        if not 0 <= feature_idx < self.n_features:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self.n_features})"
            )

        return visualization.plot_feature_quality_bars(
            self.all_metrics[feature_idx],
            feature_idx=feature_idx,
        )

    def plot_feature_ranking(
        self,
        metric_name: str = "fisher_discriminant_ratio",
    ) -> go.Figure:
        return visualization.plot_feature_quality_comparison(
            self.all_metrics,
            metric_name=metric_name,
        )

    def plot_class_distributions(self, feature_idx: int) -> go.Figure:
        if not 0 <= feature_idx < self.n_features:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self.n_features})"
            )
        feature_col = self.feature_matrix[:, feature_idx]
        return visualization.plot_class_distributions(
            feature_col,
            self.labels,
            label_mapping=self.label_mapping,
            feature_name=self.feature_names[feature_idx],
        )

    def plot_class_scatter(self, feature_idx: int) -> go.Figure:
        if not 0 <= feature_idx < self.n_features:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self.n_features})"
            )

        feature_col = self.feature_matrix[:, feature_idx]
        return visualization.plot_class_scatter(
            feature_col,
            self.labels,
            label_mapping=self.label_mapping,
            feature_name=self.feature_names[feature_idx],
        )

    def plot_feature_matrix_heatmap(
        self, top_k: int = 15, n_samples: int = 20
    ) -> go.Figure:
        return visualization.plot_feature_matrix_heatmap(
            self.feature_matrix,
            top_k=top_k,
            n_samples=n_samples,
        )

    def plot_pca_3d(self) -> go.Figure:
        return visualization.plot_pca_3d_colored(
            self.feature_matrix,
            self.labels,
            label_mapping=self.label_mapping,
        )

    def plot_correlation_heatmap(self) -> go.Figure:
        return visualization.plot_feature_correlation_heatmap(
            self.feature_matrix,
            feature_names=self.feature_names,
        )

    def plot_metrics_heatmap(self) -> go.Figure:
        return visualization.plot_metrics_heatmap(self.all_metrics)

    def get_metrics_table(self) -> dict[int, dict[str, float]]:
        return self.all_metrics

    def get_analysis_report(self) -> str:
        report = []
        report.append("=" * 70)
        report.append("FEATURE QUALITY ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        report.append("DATASET SUMMARY")
        report.append("-" * 70)
        report.append(f"  Number of Samples:  {self.n_samples}")
        report.append(f"  Number of Features: {self.n_features}")
        report.append(f"  Number of Classes:  {len(np.unique(self.labels))}")
        report.append("")
        metrics_to_report = [
            "fisher_discriminant_ratio",
            "kullback_leibler_divergence",
            "mutual_information",
            "silhouette_score",
            "class_separability_index",
            "class_overlap_ratio",
        ]
        report.append("TOP-5 FEATURES BY METRIC")
        report.append("-" * 70)
        for metric_name in metrics_to_report:
            ranked = self.rank_features(metric_name, ascending=False)
            top_5 = ranked[:5]
            readable_name = metric_name.replace("_", " ").title()
            report.append(f"\n{readable_name}:")
            for rank, (f_idx, score) in enumerate(top_5, 1):
                fname = (
                    self.feature_names[f_idx]
                    if f_idx < len(self.feature_names)
                    else f"F{f_idx}"
                )
                report.append(f"  {rank}. {fname} (F{f_idx}): {score:.4f}")
        report.append("")
        report.append("=" * 70)
        return "\n".join(report)
