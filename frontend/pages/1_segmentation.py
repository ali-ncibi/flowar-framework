import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.dataset.methods.base import SmartHomeDataset
from frontend.components import data_loader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from backend.segmentation.methods import (  # noqa: E402
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
from backend.segmentation.optimization import (  # noqa: E402
    HyperparameterOptimizer,
    get_default_ranges,
)
from backend.segmentation.quality import CrossValidationEvaluator  # noqa: E402
from frontend.components.shared import (  # noqa: E402
    SEGMENTATION_METHODS,
    handle_segmentation_execution,
    setup_segmentation_params,
)

st.set_page_config(page_title="Segmentation", layout="wide")
st.title("Segmentation")
data_loader.render()
dataset: SmartHomeDataset = st.session_state["dataset"]
tab1, tab2 = st.tabs(["⚙️ Execute Segmentation", "🔍 Hyperparameter Search"])
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1.container(border=True):
        st.subheader("Segmentation Method")
        method_name = st.selectbox("Select Segmentation Method", SEGMENTATION_METHODS)
    with col2.container(border=True):
        st.subheader("Configure Parameters")
        params = setup_segmentation_params(method_name)
    with col1.container(border=True):
        st.subheader("Execute Segmentation")
        st.caption("On the whole loaded dataset (for illustration purposes).")
        if st.button("🚀 Execute Segmentation", type="primary", width="stretch"):
            try:
                res = handle_segmentation_execution(dataset, method_name, params)
                segment_data = []
                for i, segment in enumerate(res.segments):
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
            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")
                st.error(traceback.format_exc())
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1.container(border=True):
        st.subheader("Hyperparameter Optimization")
        st.caption(
            "Automatically find optimal hyperparameters of the Segmenter using Bayesian optimization "
            "and cross-validation with F1 score (boundary detection accuracy)."
        )
        opt_method = st.selectbox(
            "Select Method for Optimization",
            SEGMENTATION_METHODS,
            key="opt_method_select",
        )
        default_ranges = get_default_ranges(opt_method)
        if not default_ranges:
            st.error(f"No parameter ranges defined for {opt_method}")
            st.stop()
    with col2.container(border=True):
        st.subheader("Configure Hyperparameter Search Ranges")
        st.caption("Adjust the min/max ranges for each hyperparameter")
        user_param_ranges: dict[str, dict[str, Any]] = {}
        for param_name, param_info in default_ranges.items():
            st.caption(f"Adjust **{param_name}** parameter range:")
            coll1, coll2 = st.columns([1, 1])
            with coll1:
                if param_info["type"] == "int":
                    min_val: float = st.number_input(
                        f"Min {param_name}",
                        value=int(param_info["min"]),
                        min_value=int(param_info["min"]),
                        max_value=int(param_info["max"]),
                        key=f"opt_min_{param_name}",
                    )
                else:
                    min_val = st.number_input(
                        f"Min {param_name}",
                        value=float(param_info["min"]),
                        format="%.4f",
                        key=f"opt_min_{param_name}",
                    )
            with coll2:
                if param_info["type"] == "int":
                    max_val: float = st.number_input(
                        f"Max {param_name}",
                        value=int(param_info["max"]),
                        min_value=int(param_info["min"]),
                        max_value=int(param_info["max"]),
                        key=f"opt_max_{param_name}",
                    )
                else:
                    max_val = st.number_input(
                        f"Max {param_name}",
                        value=float(param_info["max"]),
                        format="%.4f",
                        key=f"opt_max_{param_name}",
                    )
            user_param_ranges[param_name] = {
                "min": min_val,
                "max": max_val,
                "type": param_info["type"],
            }
    with col1.container(border=True):
        st.subheader("Optimization Settings")
        col_opt1, col_opt2, col_opt3 = st.columns([1, 2, 2])
        with col_opt1:
            n_iterations = st.slider("Iterations", 10, 200, 50, key="opt_iterations")
        with col_opt2:
            tolerance = st.number_input(
                "Boundary Tolerance (seconds)", 0, 300, 10, key="opt_tolerance"
            )
        with col_opt3:
            cv_folds = st.number_input(
                "CV Folds to Use",
                1,
                len(dataset.leave_one_day_out_split()),
                len(dataset.leave_one_day_out_split()),
                key="opt_cv_folds",
            )
        if st.button("Start Optimization", key="opt_start_button", width="stretch"):
            try:
                with st.spinner("Running Bayesian Opt... This may take a while."):
                    if opt_method == "SepCpd":
                        segmenter_class: type[BaseSegmentationMethod] = SepCpd
                    elif opt_method == "CorrWan":
                        segmenter_class = CorrWan
                    elif opt_method == "CorrXu":
                        segmenter_class = CorrXu
                    elif opt_method == "RulsifPlsbdCpd":
                        segmenter_class = RulsifPlsbdCpd
                    elif opt_method == "RulsifPeCpd":
                        segmenter_class = RulsifPeCpd
                    elif opt_method == "W2vCsCPD":
                        segmenter_class = W2vCsCPD
                    elif opt_method == "W2vCcsCPD":
                        segmenter_class = W2vCcsCPD
                    elif opt_method == "W2vCcsdCPD":
                        segmenter_class = W2vCcsdCPD
                    elif opt_method == "FixedCount":
                        segmenter_class = FixedCount
                    elif opt_method == "FixedTime":
                        segmenter_class = FixedTime
                    else:
                        st.error(f"Unknown method: {opt_method}")
                        st.stop()
                    evaluator = CrossValidationEvaluator(
                        dataset, tolerance, int(cv_folds)
                    )
                    optimizer = HyperparameterOptimizer(evaluator)
                    result = optimizer.optimize(
                        segmenter_class,
                        user_param_ranges,
                        n_calls=int(n_iterations),
                        n_initial_points=max(3, int(n_iterations) // 10),
                    )
                    st.success(
                        f"✓ Optimization complete! Best F1 Score: {result.best_f1:.4f}"
                    )
                st.subheader("Optimization Results")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Best F1 Score", f"{result.best_f1:.4f}")
                    st.write("**Best Hyperparameters:**")
                    for param_name, param_value in result.best_params.items():
                        param_type = user_param_ranges[param_name]["type"]
                        if param_type == "int":
                            st.write(f"- {param_name}: {int(param_value)}")
                        else:
                            st.write(f"- {param_name}: {float(param_value):.6f}")
                with col_res2:
                    st.write("**Search Space:**")
                    for param_name, param_config in user_param_ranges.items():
                        st.write(
                            f"- {param_name}: [{param_config['min']}, {param_config['max']}]"
                        )
                st.subheader("Optimization History")
                history_data = []
                for iter_result in result.iteration_results:
                    history_data.append(
                        {
                            "Iteration": iter_result["iteration"] + 1,
                            "F1 Score": f"{iter_result['mean_f1']:.4f}",
                            "Params": str(
                                {
                                    k: (
                                        int(v)
                                        if user_param_ranges[k]["type"] == "int"
                                        else f"{v:.4f}"
                                    )
                                    for k, v in iter_result["params"].items()
                                }
                            )[:80],
                        }
                    )
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, width="stretch", hide_index=True)
                if result.iteration_results:
                    st.subheader("Convergence Plot")
                    iterations = [r["iteration"] + 1 for r in result.iteration_results]
                    f1_scores = [r["mean_f1"] for r in result.iteration_results]
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
                            y=f1_scores,
                            mode="lines+markers",
                            name="F1 Score",
                            line=dict(width=2),
                            marker=dict(size=8),
                            customdata=param_texts,
                            hovertemplate="<b>Iteration %{x}</b><br>F1 Score: %{y:.4f}<br><br><b>Parameters:</b><br>%{customdata}<extra></extra>",
                        )
                    )
                    fig.add_hline(
                        y=result.best_f1,
                        line_dash="dash",
                        line_color="red",
                        name="Best F1",
                        annotation_text="Best F1",
                        annotation_position="right",
                    )
                    fig.update_layout(
                        title=f"Optimization Convergence - {opt_method}",
                        xaxis_title="Iteration",
                        yaxis_title="F1 Score",
                        hovermode="closest",
                        height=500,
                        width=800,
                    )
                    st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.error(traceback.format_exc())
