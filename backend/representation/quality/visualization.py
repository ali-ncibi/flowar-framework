"""Visualization functions for feature quality analysis."""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def plot_feature_quality_bars(
    metrics_dict: dict[str, float],
    feature_idx: int | None = None,
) -> go.Figure:
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    max_val = max(metric_values) if metric_values else 1.0
    if max_val > 0:
        normalized_values = [v / max_val for v in metric_values]
    else:
        normalized_values = metric_values
    metric_min = min(metric_values) if metric_values else 0.0
    metric_max = max(metric_values) if metric_values else 1.0
    color_values = [
        (v - metric_min) / (metric_max - metric_min) if metric_max > metric_min else 0.5
        for v in metric_values
    ]

    fig = go.Figure(
        data=go.Bar(
            y=metric_names,
            x=normalized_values,
            orientation="h",
            marker=dict(
                color=color_values,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                line=dict(color="black", width=0.5),
                showscale=False,
            ),
            text=[f"{val:.3f}" for val in metric_values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{customdata:.3f}<extra></extra>",
            customdata=metric_values,
        )
    )

    fig.update_layout(
        title={
            "text": (
                f"Feature Quality Metrics (Feature {feature_idx})"
                if feature_idx is not None
                else "Feature Quality Metrics"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        xaxis_title="Normalized Score",
        width=1200,
        height=600,
        xaxis=dict(range=[0, 1.15], gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        hovermode="closest",
        font=dict(size=10),
        showlegend=False,
    )

    return fig


def plot_feature_quality_comparison(
    all_metrics: dict[int, dict[str, float]],
    metric_name: str = "fisher_discriminant_ratio",
) -> go.Figure:
    feature_indices = sorted(all_metrics.keys())
    metric_values = [
        all_metrics[f_idx].get(metric_name, 0.0) for f_idx in feature_indices
    ]
    max_val = max(metric_values) if metric_values else 1.0
    min_val = min(metric_values) if metric_values else 0.0
    normalized_values = [
        (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        for v in metric_values
    ]

    fig = go.Figure(
        data=go.Bar(
            x=[f"F{i}" for i in feature_indices],
            y=metric_values,
            marker=dict(
                color=normalized_values,
                colorscale="RdYlGn",
                cmin=0,
                cmax=1,
                line=dict(color="black", width=0.5),
                showscale=False,
            ),
            text=[f"{val:.2f}" for val in metric_values],
            textposition="outside",
            hovertemplate="<b>Feature %{x}</b><br>Score: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"Feature Ranking: {metric_name.replace('_', ' ').title()}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        xaxis_title="Feature Index",
        yaxis_title="Score",
        width=1200,
        height=500,
        hovermode="x unified",
        font=dict(size=10),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        showlegend=False,
    )

    return fig


def plot_class_distributions(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
    label_mapping: dict[int, Any] | None = None,
    feature_name: str = "Feature",
) -> go.Figure:
    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(unique_labels)))
    colors_hex = [
        f"rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)"
        for c in colors
    ]

    fig = go.Figure()
    for label_id, color in zip(unique_labels, colors_hex, strict=False):
        mask = labels == label_id
        data = feature_col[mask]
        label_name = (
            label_mapping[label_id]
            if label_mapping and label_id in label_mapping
            else f"Class {label_id}"
        )
        fig.add_trace(
            go.Histogram(
                x=data,
                name=label_name,
                marker=dict(color=color, line=dict(color="black", width=0.5)),
                nbinsx=15,
                opacity=0.75,
            )
        )

    fig.update_layout(
        title={
            "text": f"Class Distributions: {feature_name}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        xaxis_title=feature_name,
        yaxis_title="Frequency",
        width=1200,
        height=600,
        barmode="overlay",
        hovermode="x unified",
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        xaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        font=dict(size=10),
    )

    return fig


def plot_class_scatter(
    feature_col: NDArray[np.float64],
    labels: NDArray[np.int_],
    label_mapping: dict[int, Any] | None = None,
    feature_name: str = "Feature",
) -> go.Figure:
    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(unique_labels)))
    colors_hex = [
        f"rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)"
        for c in colors
    ]

    fig = go.Figure()
    for label_id, color in zip(unique_labels, colors_hex, strict=False):
        mask = labels == label_id
        data = feature_col[mask]
        x_positions = np.arange(len(data))
        label_name = (
            label_mapping[label_id]
            if label_mapping and label_id in label_mapping
            else f"Class {label_id}"
        )
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=data,
                mode="markers",
                name=label_name,
                marker=dict(
                    color=color,
                    size=8,
                    line=dict(color="black", width=0.5),
                ),
                hovertemplate="<b>%{fullData.name}</b><br>Sample: %{x}<br>Value: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": f"Class Separation: {feature_name}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        xaxis_title="Sample Index",
        yaxis_title=feature_name,
        width=1200,
        height=500,
        hovermode="closest",
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        xaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", gridwidth=1),
        font=dict(size=10),
    )

    return fig
    plt.tight_layout()
    return fig


def plot_feature_matrix_heatmap(
    feature_matrix: NDArray[np.float64],
    top_k: int = 15,
    n_samples: int = 20,
) -> go.Figure:
    n_rows = min(n_samples, feature_matrix.shape[0])
    n_cols = min(top_k, feature_matrix.shape[1])
    sample_matrix = feature_matrix[:n_rows, :n_cols]
    sample_matrix_norm = (sample_matrix - sample_matrix.min(axis=0)) / (
        sample_matrix.max(axis=0) - sample_matrix.min(axis=0) + 1e-10
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=sample_matrix_norm,
            x=list(range(n_cols)),
            y=list(range(n_rows)),
            colorscale="RdYlGn",
            colorbar=dict(
                title="Normalized Value",
                thickness=15,
                len=0.7,
            ),
            hovertemplate="Feature %{x}<br>Sample %{y}<br>Value: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"Feature Matrix Heatmap (Top {n_cols} features, {n_rows} samples)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        xaxis_title="Feature Index",
        yaxis_title="Sample Index",
        width=1200,
        height=600,
        font=dict(size=10),
    )

    return fig


def plot_pca_3d_colored(
    feature_matrix: NDArray[np.float64],
    labels: NDArray[np.int_],
    label_mapping: dict[int, Any] | None = None,
) -> go.Figure:
    n_components = min(3, feature_matrix.shape[1])
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(feature_matrix)
    explained_variance_ratio = np.asarray(
        pca.explained_variance_ratio_, dtype=np.float64
    ).tolist()
    pc1_variance = (
        explained_variance_ratio[0] if len(explained_variance_ratio) > 0 else 0.0
    )
    pc2_variance = (
        explained_variance_ratio[1] if len(explained_variance_ratio) > 1 else 0.0
    )
    pc3_variance = (
        explained_variance_ratio[2] if len(explained_variance_ratio) > 2 else 0.0
    )
    if n_components < 3:
        pca_features = np.pad(
            pca_features,
            ((0, 0), (0, 3 - n_components)),
            mode="constant",
        )

    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(unique_labels)))
    colors_hex = [
        f"rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.7)"
        for c in colors
    ]

    fig = go.Figure()
    for label_id, color in zip(unique_labels, colors_hex, strict=False):
        mask = labels == label_id
        label_name = (
            label_mapping[label_id]
            if label_mapping and label_id in label_mapping
            else f"Class {label_id}"
        )
        fig.add_trace(
            go.Scatter3d(
                x=pca_features[mask, 0],
                y=pca_features[mask, 1],
                z=pca_features[mask, 2],
                mode="markers",
                name=label_name,
                marker=dict(
                    size=6,
                    color=color,
                    line=dict(color="black", width=0.5),
                    opacity=0.8,
                ),
                hovertemplate="<b>%{fullData.name}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "3D PCA: Feature Space Colored by Activity Label",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        scene=dict(
            xaxis=dict(
                title=f"PC1 ({pc1_variance:.1%})",
                backgroundcolor="rgba(230, 230, 230, 0.9)",
                gridcolor="white",
                showbackground=True,
            ),
            yaxis=dict(
                title=f"PC2 ({pc2_variance:.1%})",
                backgroundcolor="rgba(230, 230, 230, 0.9)",
                gridcolor="white",
                showbackground=True,
            ),
            zaxis=dict(
                title=f"PC3 ({pc3_variance if n_components > 2 else 0:.1%})",
                backgroundcolor="rgba(230, 230, 230, 0.9)",
                gridcolor="white",
                showbackground=True,
            ),
        ),
        width=1200,
        height=800,
        showlegend=True,
        hovermode="closest",
        font=dict(size=10),
    )

    return fig


def plot_feature_correlation_heatmap(
    feature_matrix: NDArray[np.float64],
    feature_names: list[str] | None = None,
) -> go.Figure:
    corr_matrix = np.atleast_2d(np.corrcoef(feature_matrix.T))
    n_features = feature_matrix.shape[1]
    if feature_names and len(feature_names) == n_features:
        labels = feature_names
    else:
        labels = [f"F{i}" for i in range(n_features)]
    hover_text = []
    for i in range(n_features):
        row = []
        for j in range(n_features):
            row.append(f"{corr_matrix[i, j]:.2f}")
        hover_text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=hover_text,
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar=dict(
                title="Correlation<br>Coefficient",
                thickness=15,
                len=0.7,
            ),
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Feature Correlation Matrix",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        xaxis_title="Feature",
        yaxis_title="Feature",
        width=1200,
        height=1000,
        font=dict(size=10),
        xaxis=dict(tickangle=-45),
    )

    return fig


def plot_metrics_heatmap(
    all_metrics: dict[int, dict[str, float]],
) -> go.Figure:
    feature_indices = sorted(all_metrics.keys())
    metric_names = (
        list(all_metrics[feature_indices[0]].keys()) if feature_indices else []
    )
    matrix = np.zeros((len(metric_names), len(feature_indices)))
    for j, f_idx in enumerate(feature_indices):
        for i, m_name in enumerate(metric_names):
            matrix[i, j] = all_metrics[f_idx].get(m_name, 0.0)
    for i in range(len(metric_names)):
        row = matrix[i, :]
        max_val = np.max(row)
        if max_val > 0:
            matrix[i, :] = row / max_val
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"F{i}" for i in feature_indices],
            y=[m.replace("_", " ").title() for m in metric_names],
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Normalized<br>Score", thickness=20, len=0.7),
            hovertemplate="<b>%{y}</b><br>Feature %{x}<br>Score: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Feature Quality Metrics Heatmap (Normalized)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "black"},
        },
        xaxis_title="Feature Index",
        yaxis_title="Quality Metric",
        width=1200,
        height=500,
        hovermode="closest",
        font=dict(size=10),
    )

    return fig
