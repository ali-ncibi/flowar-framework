import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.title("Clean-FlowAR")
st.subheader(
    "A comprehensive data-driven framework for activity recognition in smart environments"
)
