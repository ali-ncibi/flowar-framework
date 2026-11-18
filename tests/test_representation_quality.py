import numpy as np
import plotly.graph_objects as go
import pytest

from backend.representation.quality import (
    FeatureQualityAnalyzer,
    compute_all_features_metrics,
    compute_all_quality_metrics,
    visualization,
)
from backend.representation.quality import metrics as qmetrics
from backend.representation.quality import utils as qutils


@pytest.fixture
def feature_matrix():
    rng = np.random.RandomState(42)
    X = np.vstack(
        [
            rng.normal(0.0, 0.1, size=(10, 3)),
            rng.normal(5.0, 0.1, size=(10, 3)),
        ]
    ).astype(np.float64)
    return X


@pytest.fixture
def labels():
    return np.array([0] * 10 + [1] * 10, dtype=np.int_)


class TestUtils:
    def test_encode_labels_int(self, labels):
        enc, mapping = qutils.encode_labels(labels)
        assert enc.tolist() == labels.tolist()
        assert mapping == {0: 0, 1: 1}

    def test_encode_labels_str(self):
        enc, mapping = qutils.encode_labels(np.array(["b", "a", "b"]))
        assert enc.tolist() == [1, 0, 1]
        assert mapping == {0: "a", 1: "b"}

    def test_normalize_features(self, feature_matrix):
        out = qutils.normalize_features(feature_matrix)
        assert out.shape == feature_matrix.shape
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-8)

    def test_get_class_samples(self, feature_matrix, labels):
        samples = qutils.get_class_samples(feature_matrix[:, 0], labels)
        assert set(samples) == {0, 1}
        assert len(samples[0]) == 10

    def test_class_means_variances(self, feature_matrix, labels):
        col = feature_matrix[:, 0]
        means = qutils.compute_class_means(col, labels)
        assert means[0] == pytest.approx(col[:10].mean())
        variances = qutils.compute_class_variances(col, labels)
        assert variances[1] == pytest.approx(col[10:].var())

    def test_entropies(self, feature_matrix, labels):
        col = feature_matrix[:, 0]
        assert qutils.compute_feature_entropy(col) > 0
        assert qutils.compute_conditional_entropy(col, labels) >= 0
        assert qutils.compute_label_entropy(labels) == pytest.approx(1.0, abs=1e-6)


class TestMetrics:
    def test_fisher(self, feature_matrix, labels):
        assert qmetrics.fisher_discriminant_ratio(feature_matrix[:, 0], labels) > 0

    def test_fisher_zero_within_var(self):
        col = np.array([1.0, 1.0, 2.0, 2.0])
        labels = np.array([0, 0, 1, 1])
        assert qmetrics.fisher_discriminant_ratio(col, labels) == 0.0

    def test_kl_divergence(self, feature_matrix, labels):
        assert qmetrics.kullback_leibler_divergence(feature_matrix[:, 0], labels) >= 0

    def test_mutual_information(self, feature_matrix, labels):
        assert qmetrics.mutual_information_score(feature_matrix[:, 0], labels) >= 0

    def test_silhouette(self, feature_matrix, labels):
        score = qmetrics.silhouette_score_feature(feature_matrix[:, 0], labels)
        assert -1.0 <= score <= 1.0

    def test_separability(self, feature_matrix, labels):
        assert qmetrics.class_separability_index(feature_matrix[:, 0], labels) > 0

    def test_separability_single_class(self):
        col = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 0])
        assert qmetrics.class_separability_index(col, labels) == 0.0

    def test_separability_no_intra(self):
        col = np.array([1.0, 5.0])
        labels = np.array([0, 1])
        # each class has 1 sample -> no intra distances -> 0.0
        assert qmetrics.class_separability_index(col, labels) == 0.0

    def test_separability_zero_intra_mean(self):
        col = np.array([1.0, 1.0, 5.0, 5.0])
        labels = np.array([0, 0, 1, 1])
        # intra distances all zero -> first guard returns 0.0
        assert qmetrics.class_separability_index(col, labels) == 0.0

    def test_overlap_ratio(self, feature_matrix, labels):
        r = qmetrics.class_overlap_ratio(feature_matrix[:, 0], labels)
        assert 0.0 <= r <= 1.0

    def test_overlap_ratio_single_sample_classes(self):
        col = np.array([0.0, 10.0])
        labels = np.array([0, 1])
        assert qmetrics.class_overlap_ratio(col, labels) == 0.0

    def test_compute_all_quality_metrics(self, feature_matrix, labels):
        m = compute_all_quality_metrics(feature_matrix[:, 0], labels)
        assert set(m) == {
            "fisher_discriminant_ratio",
            "kullback_leibler_divergence",
            "mutual_information",
            "silhouette_score",
            "class_separability_index",
            "class_overlap_ratio",
        }

    def test_compute_all_features_metrics(self, feature_matrix, labels):
        m = compute_all_features_metrics(feature_matrix, labels)
        assert set(m) == {0, 1, 2}


class TestAnalyzer:
    def test_init_int_labels(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        assert an.n_samples == 20
        assert an.n_features == 3
        assert an.label_mapping_auto is None
        assert an.label_mapping is None
        assert an.feature_names == ["Feature_0", "Feature_1", "Feature_2"]

    def test_init_str_labels(self, feature_matrix):
        labels = np.array(["x"] * 10 + ["y"] * 10)
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        assert an.label_mapping_auto == {0: "x", 1: "y"}
        assert an.label_mapping == {0: "x", 1: "y"}

    def test_init_custom_names_and_mapping(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(
            feature_matrix,
            labels,
            feature_names=["a", "b", "c"],
            label_mapping={0: "A", 1: "B"},
        )
        assert an.feature_names == ["a", "b", "c"]
        assert an.label_mapping == {0: "A", 1: "B"}

    def test_compute_all_metrics(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        assert an.compute_all_metrics() is an.all_metrics

    def test_rank_and_top(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        ranked = an.rank_features("fisher_discriminant_ratio")
        assert len(ranked) == 3
        ranked_asc = an.rank_features("fisher_discriminant_ratio", ascending=True)
        assert ranked_asc[0][1] <= ranked_asc[-1][1]
        top = an.get_top_features("fisher_discriminant_ratio", top_k=2)
        assert len(top) == 2

    def test_rank_missing_metric_defaults_zero(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        ranked = an.rank_features("nonexistent_metric")
        assert all(v == 0.0 for _, v in ranked)

    def test_feature_summary(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        s = an.get_feature_summary()
        assert s["n_samples"] == 20 and s["n_classes"] == 2

    def test_plots(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        assert isinstance(an.plot_feature_quality_bars(0), go.Figure)
        assert isinstance(an.plot_feature_ranking(), go.Figure)
        assert isinstance(an.plot_class_distributions(0), go.Figure)
        assert isinstance(an.plot_class_scatter(1), go.Figure)
        assert isinstance(an.plot_feature_matrix_heatmap(), go.Figure)
        assert isinstance(an.plot_pca_3d(), go.Figure)
        assert isinstance(an.plot_correlation_heatmap(), go.Figure)
        assert isinstance(an.plot_metrics_heatmap(), go.Figure)

    def test_plot_index_out_of_range(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        with pytest.raises(ValueError, match="out of range"):
            an.plot_feature_quality_bars(99)
        with pytest.raises(ValueError, match="out of range"):
            an.plot_class_distributions(-1)
        with pytest.raises(ValueError, match="out of range"):
            an.plot_class_scatter(3)

    def test_metrics_table_and_report(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        assert an.get_metrics_table() is an.all_metrics
        report = an.get_analysis_report()
        assert "FEATURE QUALITY ANALYSIS REPORT" in report
        assert "Fisher Discriminant Ratio" in report


class TestVisualizationDirect:
    def test_bars_empty_metrics(self):
        fig = visualization.plot_feature_quality_bars({})
        assert isinstance(fig, go.Figure)

    def test_bars_zero_metrics(self):
        fig = visualization.plot_feature_quality_bars({"a": 0.0, "b": 0.0})
        assert isinstance(fig, go.Figure)

    def test_bars_feature_idx_none(self):
        fig = visualization.plot_feature_quality_bars({"a": 1.0}, feature_idx=None)
        assert isinstance(fig, go.Figure)

    def test_comparison_missing_metric(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        fig = visualization.plot_feature_quality_comparison(
            an.all_metrics, metric_name="nope"
        )
        assert isinstance(fig, go.Figure)

    def test_comparison_zero(self, feature_matrix, labels):
        zero_metrics = {i: {"m": 0.0} for i in range(3)}
        fig = visualization.plot_feature_quality_comparison(
            zero_metrics, metric_name="m"
        )
        assert isinstance(fig, go.Figure)

    def test_distributions_no_mapping(self, feature_matrix, labels):
        fig = visualization.plot_class_distributions(
            feature_matrix[:, 0], labels, label_mapping=None
        )
        assert isinstance(fig, go.Figure)

    def test_distributions_with_mapping(self, feature_matrix, labels):
        fig = visualization.plot_class_distributions(
            feature_matrix[:, 0], labels, label_mapping={0: "A", 1: "B"}
        )
        assert isinstance(fig, go.Figure)

    def test_scatter_mapping_partial(self, feature_matrix, labels):
        fig = visualization.plot_class_scatter(
            feature_matrix[:, 0],
            labels,
            label_mapping={0: "A"},  # 1 not in mapping
        )
        assert isinstance(fig, go.Figure)

    def test_pca_single_feature(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)
        labels = np.array([0, 0, 1, 1])
        fig = visualization.plot_pca_3d_colored(X, labels)
        assert isinstance(fig, go.Figure)

    def test_pca_two_features(self):
        X = np.array(
            [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 9.0]], dtype=np.float64
        )
        labels = np.array([0, 0, 1, 1])
        fig = visualization.plot_pca_3d_colored(
            X, labels, label_mapping={0: "A", 1: "B"}
        )
        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap_names_mismatch(self, feature_matrix):
        fig = visualization.plot_feature_correlation_heatmap(
            feature_matrix, feature_names=["only_one"]
        )
        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap_no_names(self, feature_matrix):
        fig = visualization.plot_feature_correlation_heatmap(feature_matrix)
        assert isinstance(fig, go.Figure)

    def test_metrics_heatmap_empty(self):
        fig = visualization.plot_metrics_heatmap({})
        assert isinstance(fig, go.Figure)

    def test_metrics_heatmap_zero(self, feature_matrix, labels):
        an = FeatureQualityAnalyzer(feature_matrix, labels)
        zero = {i: {k: 0.0 for k in v} for i, v in an.all_metrics.items()}
        fig = visualization.plot_metrics_heatmap(zero)
        assert isinstance(fig, go.Figure)
