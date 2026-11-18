import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.dataset.methods import SmartHomeDataset  # noqa: E402


def render():
    @st.cache_resource(show_spinner=False)
    def load_dataset(path, num_days):
        return SmartHomeDataset(path, num_days=num_days)

    st.sidebar.header("Dataset Selection")
    dataset_root = Path("backend/dataset/content")
    available_datasets = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    if not available_datasets:
        st.error("No datasets found in backend/dataset/content/")
        st.stop()

    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        available_datasets,
        index=available_datasets.index("ordonez-a")
        if "ordonez-a" in available_datasets
        else 0,
    )
    dataset_path = dataset_root / selected_dataset
    max_days = SmartHomeDataset.get_total_days(dataset_path)
    num_days = st.sidebar.number_input(
        "Number of Days to Load",
        min_value=1,
        max_value=max_days,
        value=min(10, max_days),
        step=1,
        help="Load data from the first N 24-hour periods",
    )

    try:
        dataset = load_dataset(dataset_path, num_days)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    st.sidebar.subheader("Dataset Info")
    stats = dataset.get_statistics()
    st.sidebar.metric("Duration (days)", f"{stats.total_duration_seconds / 86400:.1f}")
    st.sidebar.metric("Sensors", stats.num_sensors)
    st.sidebar.metric("Activities", stats.num_activities)
    st.sidebar.metric("Activations", stats.num_sensor_activations)
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by FlowAR")
    st.session_state["dataset"] = dataset
