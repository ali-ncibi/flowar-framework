import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.dataset.methods.base import SmartHomeDataset
from backend.representation.quality import FeatureQualityAnalyzer
from frontend.components import data_loader
from frontend.components.shared import (
    SEGMENTATION_METHODS,
    handle_annotation_execution,
    handle_representation_execution,
    handle_segmentation_execution,
    setup_representation_params,
    setup_segmentation_params,
)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


st.set_page_config(page_title="Segment Representation", layout="wide")
st.title("Segment Representation")
data_loader.render()
dataset: SmartHomeDataset = st.session_state["dataset"]
tab1, tab2 = st.tabs(["⚙️ Execute Feature Extraction", "🔍 Feature Analysis"])
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1.container(border=True):
        st.subheader("Segmentation Configuration")
        seg_method_name = st.selectbox(
            "Select Segmentation Method", SEGMENTATION_METHODS
        )
        seg_params = setup_segmentation_params(seg_method_name)
    with col1.container(border=True):
        st.subheader("Annotation Configuration")
        st.caption("Select how to annotate segments with activity labels. ")
        annotation_method_name = st.selectbox(
            "Annotation Method",
            [
                "MostOverlappingAnnotator",
                "AnyOverlappingAnnotator",
                "AtSegmentEndAnnotator",
            ],
            help="Method to assign activity labels to segments",
        )
    with col2.container(border=True):
        st.subheader("Representation Configuration")
        representation_type = st.radio(
            "Representation Type",
            options=["Vector", "Matrix"],
            help="Vector: traditional feature extraction. Matrix: 2D sensor interaction patterns.",
        )
        if representation_type == "Vector":
            if st.radio("Use Textual?", options=["No", "Yes"]) == "Yes":
                rep_method_name = "TextualVectorExtractor"
            else:
                rep_method_name = "StatsVectorExtractor"
        else:
            rep_method_name = st.selectbox(
                "Matrix Representation Method",
                [
                    "FuzzyMatrixExtractor",
                    "TextualMatrixExtractor",
                ],
            )
        rep_method_params = setup_representation_params(rep_method_name)

        if st.button(
            "🚀 Execute Feature Extraction",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Starting segmentation and annotation..."):
                    seg_res = handle_segmentation_execution(
                        dataset, seg_method_name, seg_params
                    )
                    segment_labels, activity_mapping = handle_annotation_execution(
                        dataset,
                        seg_res.segments,
                        annotation_method_name,
                    )
                    extracted, mask = handle_representation_execution(
                        dataset,
                        seg_res.segments,
                        representation_type,
                        rep_method_name,
                        rep_method_params,
                    )
                    if extracted is None:
                        st.error("Feature extraction failed")
                        st.stop()
                    st.success(f"✓ Extracted matrices: shape {extracted.shape}")
                    n_segments = extracted.shape[0]
                    extracted_reshaped = extracted.reshape(n_segments, -1)
                    st.info(f"ℹ Flattened for analysis: {extracted_reshaped.shape}")
                st.subheader("Detected Segments")
                segment_data = []
                for i, segment in enumerate(seg_res.segments):
                    segment_data.append(
                        {
                            "Segment #": i + 1,
                            "Start": segment.interval.start.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "End": segment.interval.end.strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration (s)": round(segment.interval.duration, 2),
                            "Num Events": segment.num_events,
                        }
                    )
                df_segments = pd.DataFrame(segment_data)
                st.dataframe(df_segments, width="stretch", hide_index=True)
                st.subheader("🔹 Computed Features Matrix")
                assert extracted_reshaped is not None, (
                    "Extracted features should not be None"
                )
                st.info(
                    f"Feature matrix shape: {extracted_reshaped.shape[0]} segments × {extracted_reshaped.shape[1]} features"
                )
                st.dataframe(extracted_reshaped, width="stretch", hide_index=True)
                st.session_state.last_feature_matrix = extracted_reshaped
                st.session_state.last_segments = seg_res.segments
                st.session_state.last_dataset = dataset
                st.session_state.last_segment_labels = segment_labels
                st.session_state.last_activity_mapping = activity_mapping
                st.session_state.last_sensor_activations = dataset.sensor_activations
                st.session_state.last_representation_type = representation_type

            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
                st.error(traceback.format_exc())

with tab2:
    st.subheader("📊 Feature Quality Analysis")

    if "last_feature_matrix" not in st.session_state:
        st.info("👈 Run segmentation and feature extraction first to analyze features")
    elif "last_segment_labels" not in st.session_state:
        st.warning(
            "⚠️ Segment labels not available. Run segmentation with annotation enabled."
        )
    else:
        feature_matrix = st.session_state.last_feature_matrix
        segment_labels = st.session_state.last_segment_labels
        activity_mapping = st.session_state.last_activity_mapping
        feature_names = ["F" + str(i) for i in range(feature_matrix.shape[1])]
        analyzer = FeatureQualityAnalyzer(
            feature_matrix=feature_matrix,
            labels=segment_labels,
            # feature_names=feature_names[: feature_matrix.shape[1]],
            label_mapping=dict(activity_mapping) if activity_mapping else None,
        )

        analysis_tabs_list = [
            "Quality Metrics",
            "Feature Rankings",
            "Class Visualizations",
            "Feature Space",
            "Correlation",
        ]
        analysis_tabs = st.tabs(analysis_tabs_list)
        with analysis_tabs[0]:
            st.subheader("Feature Quality Metrics Heatmap")
            st.caption(
                "Shows how each feature performs across different quality metrics. "
                "Green = good, Red = poor (values normalized per metric)."
            )
            fig_heatmap = analyzer.plot_metrics_heatmap()
            st.plotly_chart(fig_heatmap, width="stretch")
            with st.expander("📖 Metric Explanations"):
                st.markdown("""
                **Fisher Discriminant Ratio**: Ratio of between-class to within-class variance. Higher is better.

                **Kullback-Leibler Divergence**: Average divergence of class distributions from overall distribution. Higher indicates more distinctive classes.

                **Mutual Information**: Dependency between feature and class labels. Higher indicates stronger correlation with classes.

                **Silhouette Score**: How well samples cluster with their class. Range [-1, 1], higher is better.

                **Class Separability Index**: Ratio of inter-class to intra-class distances. Higher means better separation.

                **Class Overlap Ratio**: Proportion of samples overlapping with other classes. Lower is better (0-1).
                """)
        with analysis_tabs[1]:
            st.subheader("Feature Rankings")
            ranking_metric = st.selectbox(
                "Select metric to rank features:",
                [
                    "fisher_discriminant_ratio",
                    "kullback_leibler_divergence",
                    "mutual_information",
                    "silhouette_score",
                    "class_separability_index",
                    "class_overlap_ratio",
                ],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            fig_ranking = analyzer.plot_feature_ranking(ranking_metric)
            st.plotly_chart(fig_ranking, width="stretch")
            st.subheader(
                f"Top-10 Features by {ranking_metric.replace('_', ' ').title()}"
            )
            ranked = analyzer.rank_features(ranking_metric, ascending=False)
            top_10 = ranked[:10]
            ranking_data = []
            for rank, (f_idx, score) in enumerate(top_10, 1):
                ranking_data.append(
                    {
                        "Rank": rank,
                        "Feature": feature_names[f_idx]
                        if f_idx < len(feature_names)
                        else f"F{f_idx}",
                        "Index": f_idx,
                        "Score": f"{score:.4f}",
                    }
                )
            df_ranking = pd.DataFrame(ranking_data)
            st.dataframe(df_ranking, width="stretch", hide_index=True)
        with analysis_tabs[2]:
            st.subheader("Class Distributions & Separability")
            col_viz1, col_viz2 = st.columns(2)
            selected_feature_idx = st.selectbox(
                "Select feature to visualize class distributions:",
                options=list(range(feature_matrix.shape[1])),
                format_func=lambda x: (
                    f"{feature_names[x] if x < len(feature_names) else f'F{x}'} (F{x})"
                ),
            )
            with col_viz1:
                st.write("**Class Distributions (Histogram)**")
                fig_dist = analyzer.plot_class_distributions(selected_feature_idx)
                st.plotly_chart(fig_dist, width="stretch")
            with col_viz2:
                st.write("**Class Separation (Scatter)**")
                fig_scatter = analyzer.plot_class_scatter(selected_feature_idx)
                st.plotly_chart(fig_scatter, width="stretch")
            st.subheader(
                f"Quality Metrics for {feature_names[selected_feature_idx] if selected_feature_idx < len(feature_names) else f'F{selected_feature_idx}'}"
            )
            fig_bars = analyzer.plot_feature_quality_bars(selected_feature_idx)
            st.plotly_chart(fig_bars, width="stretch")
        with analysis_tabs[3]:
            st.subheader("Feature Space Projection (PCA)")
            st.caption(
                "3D PCA projection of the feature space, colored by activity label. "
                "Well-separated clusters indicate good feature quality for distinguishing activities."
            )
            try:
                fig_pca = analyzer.plot_pca_3d()
                st.plotly_chart(fig_pca, width="stretch")
            except Exception as e:
                st.warning(f"Could not generate PCA visualization: {str(e)}")
            st.subheader("Feature Matrix Heatmap (Sample)")
            st.caption("Visualization of feature values for a subset of segments")
            col_heat1, col_heat2 = st.columns(2)
            with col_heat1:
                top_k = st.number_input(
                    "Number of top features to show",
                    min_value=5,
                    max_value=30,
                    value=15,
                )
            with col_heat2:
                n_samples = st.number_input(
                    "Number of samples to show", min_value=5, max_value=50, value=20
                )

            fig_heat = analyzer.plot_feature_matrix_heatmap(
                top_k=top_k, n_samples=n_samples
            )
            st.plotly_chart(fig_heat, width="stretch")
        with analysis_tabs[4]:
            st.subheader("Feature Correlation Matrix")
            st.caption(
                "Shows correlation between features. Values close to ±1 indicate strong correlation. "
                "Highly correlated features may be redundant."
            )
            try:
                fig_corr = analyzer.plot_correlation_heatmap()
                if st.toggle("Show correlation values", value=False):
                    fig_corr.update_traces(
                        texttemplate="%{z:.2f}",
                        textfont={"size": 8},
                    )
                    st.plotly_chart(fig_corr, width="stretch")
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {str(e)}")
