import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.annotation.methods import (
    AnyOverlappingAnnotator,
    AtSegmentEndAnnotator,
    BaseAnnotationMethod,
    MostOverlappingAnnotator,
)
from backend.classification.methods import (
    BaseClassificationMethod,
    DecisionTreeClassifier,
    DeepSequenceGridClassifier,
    KNeighborsClassifier,
)
from backend.classification.optimization import (
    HyperparameterOptimizer,
    get_default_ranges,
)
from backend.classification.quality import CrossValidationEvaluator
from backend.dataset.methods.base import SmartHomeDataset
from backend.representation.methods import (
    BaseRepresentationMethod,
    FuzzyMatrixExtractor,
    StatsVectorExtractor,
    TextualMatrixExtractor,
    TextualVectorExtractor,
)
from backend.segmentation.methods import (
    BaseSegmentationMethod,
    CorrWan,
    CorrXu,
    FixedCount,
    FixedTime,
    RulsifPeCpd,
    RulsifPlsbdCpd,
    SepCpd,
    W2vCcsCPD,
    W2vCcsdCPD,
    W2vCsCPD,
)
from frontend.components import data_loader
from frontend.components.shared import (
    SEGMENTATION_METHODS,
    handle_annotation_execution,
    handle_classification_execution,
    handle_representation_execution,
    handle_segmentation_execution,
    setup_classification_params,
    setup_representation_params,
    setup_segmentation_params,
)

CLASSIFICATION_METHODS = [
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "DeepSequenceGridClassifier",
]

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


st.set_page_config(page_title="Segment Classification", layout="wide")
st.title("Segment Classification")
data_loader.render()
dataset: SmartHomeDataset = st.session_state["dataset"]
tab1, tab2 = st.tabs(["⚙️ Execute Classification", "🔍 Hyperparameter Search"])
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
    with col2.container(border=True):
        clf_method_name = st.selectbox(
            "Classification Method",
            [
                "KNeighborsClassifier",
                "DecisionTreeClassifier",
                # "DeepSequenceGridClassifier",
            ],
        )

    with col2.container(border=True):
        st.subheader("Execute Classification Pipeline")
        clf_params = setup_classification_params(clf_method_name)

        if st.button(
            "🚀 Execute Classification",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Starting segmentation and annotation..."):
                    seg_res = handle_segmentation_execution(
                        dataset, seg_method_name, seg_params
                    )
                    labels, labels_dict = handle_annotation_execution(
                        dataset,
                        seg_res.segments,
                        annotation_method_name,
                    )
                    features, masks = handle_representation_execution(
                        dataset,
                        seg_res.segments,
                        representation_type,
                        rep_method_name,
                        rep_method_params,
                    )
                    if features is None:
                        st.error("Feature extraction failed")
                        st.stop()
                    st.success(f"✓ Extracted matrices: shape {features.shape}")

                    res = handle_classification_execution(
                        features,
                        labels,
                        masks,
                        clf_method_name,
                        clf_params,
                    )

                    if res is None:
                        st.error("Classification failed")
                        st.stop()

                    classifier = res["classifier"]
                    accuracy = res["accuracy"]
                    predictions = res["predictions"]

                    st.session_state.seg_method_name = seg_method_name
                    st.session_state.seg_method_params = seg_params
                    st.session_state.annotation_method_name = annotation_method_name
                    st.session_state.rep_method_name = rep_method_name
                    st.session_state.rep_method_params = rep_method_params

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
                st.dataframe(
                    pd.DataFrame(features.reshape(features.shape[0], -1)),
                    width="stretch",
                    hide_index=True,
                )
                st.subheader("✅ Classification Results")
                acc = accuracy
                preds = predictions
                st.write(f"**Balanced Accuracy:** {acc:.4f}")
                st.write(predictions)

            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
                st.error(traceback.format_exc())

with tab2:
    if (
        "seg_method_name" not in st.session_state
        or "rep_method_name" not in st.session_state
    ):
        st.info(
            "Please execute a classification first to set up the segmentation and representation methods."
        )
        st.stop()
    col1, col2 = st.columns([1, 1])
    with col1.container(border=True):
        st.subheader("Hyperparameter Optimization")
        st.caption(
            "Automatically find optimal hyperparameters of the Classifier using Bayesian optimization "
            "and cross-validation with Balanced Accuracy score."
        )
        opt_method = st.selectbox(
            "Select Method for Optimization",
            CLASSIFICATION_METHODS,
            key="opt_method_select",
        )
        default_ranges = get_default_ranges(opt_method)
        if not default_ranges:
            st.error(f"No parameter ranges defined for {opt_method}")
            st.stop()

    with col2.container(border=True):
        st.subheader("Configure Hyperparameter Search Ranges")
        st.caption("Adjust the min/max ranges for each hyperparameter")
        classifier_param_ranges: dict[str, dict[str, Any]] = {}
        for param_name, param_info in default_ranges.items():
            st.caption(f"Adjust **{param_name}** parameter range:")
            values: list | None
            min_val: float | None
            max_val: float | None
            values, min_val, max_val = None, None, None
            coll1, coll2 = st.columns([1, 1])
            with coll1:
                if param_info["type"] == "categorical":
                    st.caption("Categorical parameter - select value in right column")
                elif param_info["type"] == "int":
                    min_val = st.number_input(
                        f"Min {param_name}",
                        value=int(param_info["min"]),
                        min_value=int(param_info["min"]),
                        max_value=int(param_info["max"]),
                        key=f"opt_min_{param_name}",
                    )
                elif param_info["type"] == "float":
                    min_val = st.number_input(
                        f"Min {param_name}",
                        value=float(param_info["min"]),
                        format="%.4f",
                        key=f"opt_min_{param_name}",
                    )
            with coll2:
                if param_info["type"] == "categorical":
                    values = st.multiselect(
                        f"Select value for {param_name}",
                        options=param_info["values"],
                        key=f"opt_cat_{param_name}",
                        default=param_info["values"],
                    )
                elif param_info["type"] == "int":
                    max_val = st.number_input(
                        f"Max {param_name}",
                        value=int(param_info["max"]),
                        min_value=int(param_info["min"]),
                        max_value=int(param_info["max"]),
                        key=f"opt_max_{param_name}",
                    )
                elif param_info["type"] == "float":
                    max_val = st.number_input(
                        f"Max {param_name}",
                        value=float(param_info["max"]),
                        format="%.4f",
                        key=f"opt_max_{param_name}",
                    )
            if param_info["type"] == "categorical":
                classifier_param_ranges[param_name] = {
                    "type": "categorical",
                    "values": values,
                }
            else:
                classifier_param_ranges[param_name] = {
                    "type": param_info["type"],
                    "min": min_val,
                    "max": max_val,
                }

    with col1.container(border=True):
        st.subheader("Optimization Settings")
        col_opt1, col_opt2 = st.columns([2, 2])
        with col_opt1:
            n_iterations = st.slider("Iterations", 10, 200, 10, key="opt_iterations")
        with col_opt2:
            cv_folds = st.number_input(
                "CV Folds to Use",
                1,
                len(dataset.leave_one_day_out_split()),
                min(3, len(dataset.leave_one_day_out_split())),
                key="opt_cv_folds",
            )
        if st.button("Start Optimization", key="opt_start_button", width="stretch"):
            with st.spinner("Starting hyperparameter optimization..."):
                try:
                    if opt_method == "KNeighborsClassifier":
                        classifier_class: type[BaseClassificationMethod] = (
                            KNeighborsClassifier
                        )
                    elif opt_method == "DecisionTreeClassifier":
                        classifier_class = DecisionTreeClassifier
                    elif opt_method == "DeepSequenceGridClassifier":
                        classifier_class = DeepSequenceGridClassifier
                    else:
                        st.error(f"Unknown classification method: {opt_method}")
                        st.stop()

                    evaluator = CrossValidationEvaluator(dataset, cv_folds)

                    segmenter_name = st.session_state.seg_method_name
                    if segmenter_name == "CorrWan":
                        segmenter_class: type[BaseSegmentationMethod] = CorrWan
                    elif segmenter_name == "CorrXu":
                        segmenter_class = CorrXu
                    elif segmenter_name == "SepCpd":
                        segmenter_class = SepCpd
                    elif segmenter_name == "RulsifPeCpd":
                        segmenter_class = RulsifPeCpd
                    elif segmenter_name == "RulsifPlsbdCpd":
                        segmenter_class = RulsifPlsbdCpd
                    elif segmenter_name == "W2vCcsCPD":
                        segmenter_class = W2vCcsCPD
                    elif segmenter_name == "W2vCcsdCPD":
                        segmenter_class = W2vCcsdCPD
                    elif segmenter_name == "W2vCsCPD":
                        segmenter_class = W2vCsCPD
                    elif segmenter_name == "FixedCount":
                        segmenter_class = FixedCount
                    elif segmenter_name == "FixedTime":
                        segmenter_class = FixedTime
                    else:
                        st.error(f"Unknown segmentation method: {segmenter_name}")
                        st.stop()
                    segmenter_params = st.session_state.get("seg_method_params", {})

                    rep_method_name = st.session_state.rep_method_name
                    if rep_method_name == "StatsVectorExtractor":
                        rep_method_class: type[BaseRepresentationMethod] = (
                            StatsVectorExtractor
                        )
                    elif rep_method_name == "TextualVectorExtractor":
                        rep_method_class = TextualVectorExtractor
                    elif rep_method_name == "FuzzyMatrixExtractor":
                        rep_method_class = FuzzyMatrixExtractor
                    elif rep_method_name == "TextualMatrixExtractor":
                        rep_method_class = TextualMatrixExtractor
                    else:
                        st.error(f"Unknown representation method: {rep_method_name}")
                        st.stop()
                    rep_method_params = st.session_state.get("rep_method_params", {})

                    annotator_class: type[BaseAnnotationMethod] | None = None
                    annotation_method_name = st.session_state.annotation_method_name
                    if annotation_method_name == "MostOverlappingAnnotator":
                        annotator_class = MostOverlappingAnnotator
                    elif annotation_method_name == "AnyOverlappingAnnotator":
                        annotator_class = AnyOverlappingAnnotator
                    elif annotation_method_name == "AtSegmentEndAnnotator":
                        annotator_class = AtSegmentEndAnnotator
                    else:
                        st.error(f"Unknown annotation method: {annotation_method_name}")
                        st.stop()

                    optimizer = HyperparameterOptimizer(
                        evaluator,
                        segmenter_class,
                        segmenter_params,
                        rep_method_class,
                        rep_method_params,
                        annotator_class,
                    )
                    result = optimizer.optimize(
                        classifier_class=classifier_class,
                        classifier_param_ranges=classifier_param_ranges,
                        n_calls=int(n_iterations),
                        n_initial_points=max(3, int(n_iterations) // 10),
                        random_state=42,
                    )
                    st.success("Optimization completed successfully!")
                    st.subheader("Optimization Results")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(
                            "Best Balanced Accuracy Score", f"{result.best_score:.4f}"
                        )
                        st.write("**Best Hyperparameters:**")
                        for param_name, param_value in result.best_params.items():
                            param_type = classifier_param_ranges[param_name]["type"]
                            st.write(f"- {param_name}: {param_value}")
                    with col_res2:
                        st.write("**Search Space:**")
                        for param_name, param_config in classifier_param_ranges.items():
                            if param_config["type"] == "categorical":
                                st.write(f"- {param_name}: {param_config['values']}")
                            else:
                                st.write(
                                    f"- {param_name}: [{param_config['min']}, {param_config['max']}]"
                                )
                    st.subheader("Optimization History")
                    history_data = []
                    for iter_result in result.iteration_results:
                        history_data.append(
                            {
                                "Iteration": iter_result["iteration"] + 1,
                                "Score": f"{iter_result['mean_accuracy']:.4f}",
                                "Params": str(
                                    {k: (v) for k, v in iter_result["params"].items()}
                                )[:80],
                            }
                        )
                    df_history = pd.DataFrame(history_data)
                    st.dataframe(df_history, width="stretch", hide_index=True)
                    if result.iteration_results:
                        st.subheader("Convergence Plot")
                        iterations = [
                            r["iteration"] + 1 for r in result.iteration_results
                        ]
                        accuracy_scores = [
                            r["mean_accuracy"] for r in result.iteration_results
                        ]
                        param_texts = []
                        for r in result.iteration_results:
                            params = r.get("params", {})
                            param_str = "<br>".join(
                                [f"{k}: {v}" for k, v in params.items()]
                            )
                            param_texts.append(param_str)
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=iterations,
                                y=accuracy_scores,
                                mode="lines+markers",
                                name="Accuracy Score",
                                line=dict(width=2),
                                marker=dict(size=8),
                                customdata=param_texts,
                                hovertemplate="<b>Iteration %{x}</b><br>Accuracy Score: %{y:.4f}<br><br><b>Parameters:</b><br>%{customdata}<extra></extra>",
                            )
                        )
                        fig.add_hline(
                            y=result.best_score,
                            line_dash="dash",
                            line_color="red",
                            name="Best Accuracy",
                            annotation_text="Best Accuracy",
                            annotation_position="right",
                        )
                        fig.update_layout(
                            title=f"Optimization Convergence - {opt_method}",
                            xaxis_title="Iteration",
                            yaxis_title="Accuracy Score",
                            hovermode="closest",
                            height=500,
                            width=800,
                        )
                        st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")
                    st.error(traceback.format_exc())
