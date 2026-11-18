import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from frontend.components import data_loader  # noqa: E402

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("📊 Smart Home Dataset Explorer")
data_loader.render()
dataset = st.session_state["dataset"]

st.write(
    "Explore various aspects of the selected smart home dataset using the tabs below."
)
(
    overview,
    sensor_analysis,
    activity_analysis,
    relationships,
    tab5,
    tab6,
    tab7,
    tab8,
    tab9,
    tab10,
) = st.tabs(
    [
        "📈 Overview",
        "🔍 Sensor Analysis",
        "🎯 Activity Analysis",
        "🔗 Relationships",
        "⏱️ Temporal",
        "📊 Rankings",
        "ℹ️ Metadata",
        "🕳️ Event Gaps",
        "📅 Timeline",
        "📋 Dataset Details",
    ]
)
with overview:
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Basic Statistics")
        stats = dataset.get_statistics()
        time_span_text = ""
        if stats.time_span[0] and stats.time_span[1]:
            time_span_text = f"- **Time Span**: {stats.time_span[0].date()} to {stats.time_span[1].date()}\n"
        st.write(f"""
        - **Total Sensors**: {stats.num_sensors}
        - **Total Activities**: {stats.num_activities}
        - **Sensor Activations**: {stats.num_sensor_activations:,}
        - **Activity Annotations**: {stats.num_activity_occurrences:,}
        {time_span_text}
        - **Duration**: {stats.total_duration_seconds / 86400:.1f} days
        """)
    with col2:
        st.subheader("Sensor List")
        sensor_names = dataset.get_sensor_names()
        st.write(f"**Total**: {len(sensor_names)} sensors")
        st.dataframe(
            pd.DataFrame({"Sensor": sensor_names}),
            width="stretch",
            hide_index=True,
        )
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Activity List")
        activity_names = dataset.get_activity_names()
        st.write(f"**Total**: {len(activity_names)} activities")
        st.dataframe(
            pd.DataFrame({"Activity": activity_names}),
            width="stretch",
            hide_index=True,
        )
with sensor_analysis:
    st.header("Sensor Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sensor Statistics")
        sensor_stats = dataset.get_sensor_statistics()
        df_sensor_stats = pd.DataFrame(
            [
                {"Sensor": k, "Activations Count": v}
                for k, v in sorted(
                    sensor_stats.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        st.dataframe(df_sensor_stats, width="stretch", hide_index=True)
        st.caption(f"Total: {len(sensor_stats)} sensors")
    with col2:
        st.subheader("Sensor Duration Statistics")
        duration_stats = dataset.get_sensor_duration_statistics()
        df_duration = pd.DataFrame(
            [
                {"Sensor": k, "Total Duration (hours)": round(v / 3600, 2)}
                for k, v in sorted(
                    duration_stats.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        st.dataframe(df_duration, width="stretch", hide_index=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average Activation Duration")
        avg_durations = dataset.get_sensor_average_durations()
        df_avg = pd.DataFrame(
            [
                {"Sensor": k, "Avg Duration (seconds)": round(v, 2)}
                for k, v in avg_durations.items()
            ]
        )
        st.dataframe(df_avg, width="stretch", hide_index=True)
    with col4:
        st.subheader("Activity Diversity by Sensor")
        k_diversity = st.slider(
            "Top sensors by diversity", 5, 20, 10, key="sensor_diversity"
        )
        if st.button("Compute Sensor Activity Diversity"):
            with st.spinner("Computing diversity..."):
                top_diverse = dataset.get_top_sensors_by_activity_diversity(
                    k=k_diversity, limit=20
                )
                df_diversity = pd.DataFrame(
                    [
                        {"Sensor": name, "Unique Activities": count}
                        for name, count in top_diverse
                    ]
                )
                st.dataframe(df_diversity, width="stretch", hide_index=True)
        else:
            st.info("Click button to compute sensor activity diversity")
    st.divider()
    st.subheader("Sensor Duration KDE Analysis")
    selected_sensor_kde = st.selectbox(
        "Select sensor for KDE analysis", dataset.get_sensor_names(), key="kde_sensor"
    )
    if st.button("Analyze Sensor Durations"):
        with st.spinner("Computing KDE distributions..."):
            try:
                fig_act, fig_deact = dataset.get_sensor_duration_kde_plot(
                    selected_sensor_kde
                )
                col_act, col_deact = st.columns(2)
                with col_act:
                    st.plotly_chart(fig_act, width="stretch")
                with col_deact:
                    st.plotly_chart(fig_deact, width="stretch")
            except ValueError as e:
                st.error(str(e))
    else:
        st.info("Click button to analyze sensor activation/deactivation durations")
with activity_analysis:
    st.header("Activity Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Activity Occurrence Count")
        activity_stats = dataset.get_activity_statistics()
        df_activity_stats = pd.DataFrame(
            [
                {"Activity": k, "Occurrences": v}
                for k, v in sorted(
                    activity_stats.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        st.dataframe(df_activity_stats, width="stretch", hide_index=True)
    with col2:
        st.subheader("Activity Duration Statistics")
        activity_duration = dataset.get_activity_duration_statistics()
        df_activity_duration = pd.DataFrame(
            [
                {"Activity": k, "Total Duration (hours)": round(v / 3600, 2)}
                for k, v in sorted(
                    activity_duration.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        st.dataframe(df_activity_duration, width="stretch", hide_index=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Longest Activity Instances")
        k_longest = st.slider(
            "Top longest instances", 5, 50, 10, key="longest_activities"
        )
        longest = dataset.get_longest_activities(k=k_longest)
        df_longest = pd.DataFrame(
            [
                {
                    "Activity": act.name,
                    "Duration (hours)": round(act.duration / 3600, 2),
                    "Start": act.interval.start,
                    "End": act.interval.end,
                }
                for act in longest
            ]
        )
        st.dataframe(df_longest, width="stretch", hide_index=True)
    with col4:
        st.subheader("Activity Correlation with Sensors")
        num_activities = st.slider(
            "Number of activities to analyze",
            1,
            len(dataset.get_activity_names()),
            5,
            key="correlation",
        )
        if st.button("Compute Activity Correlation"):
            with st.spinner("Computing correlations..."):
                correlation = dataset.get_sensor_activity_correlation(
                    num_activities=num_activities
                )
                for activity, sensors in correlation.items():
                    st.write(f"**{activity}**: {len(sensors)} sensors")
                    st.write(", ".join(sorted(sensors)))
        else:
            st.info("Click button to compute activity-sensor correlations")
with relationships:
    st.header("Sensor-Activity Relationships")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Select a Sensor")
        selected_sensor = st.selectbox("Choose sensor", dataset.get_sensor_names())
        if st.button("Get Sensor Details"):
            with st.spinner("Loading sensor details..."):
                activities = dataset.get_activities_by_sensor(selected_sensor)
                activation_count = dataset.get_sensor_activation_count(selected_sensor)

                st.write(f"**Activations**: {activation_count}")
                st.write(f"**Associated Activities ({len(activities)})**:")
                st.write(
                    ", ".join(sorted(activities)) if activities else "No activities"
                )
        else:
            st.info("Click button to get sensor details")
    with col2:
        st.subheader("Select an Activity")
        selected_activity = st.selectbox(
            "Choose activity", dataset.get_activity_names()
        )
        if st.button("Get Activity Details"):
            with st.spinner("Loading activity details..."):
                sensors = dataset.get_sensors_by_activity(selected_activity)
                annotation_count = dataset.get_activity_annotation_count(
                    selected_activity
                )
                st.write(f"**Occurrences**: {annotation_count}")
                st.write(f"**Associated Sensors ({len(sensors)})**:")
                st.write(", ".join(sorted(sensors)) if sensors else "No sensors")
        else:
            st.info("Click button to get activity details")
    st.subheader("Sensor-Activity Overlap Matrix")
    st.caption("This may take a moment to compute...")
    col_table, col_plot = st.columns([1, 1])
    with col_table:
        if st.button("Compute Full Overlap Matrix"):
            with st.spinner("Computing matrix..."):
                try:
                    matrix = dataset.get_sensor_activity_matrix()
                    matrix_data = []
                    for sensor_name, activities_dict in matrix.items():
                        for activity_name, overlap_seconds in activities_dict.items():
                            duration_hours = (
                                round(overlap_seconds / 3600, 2)
                                if overlap_seconds
                                else 0
                            )
                            matrix_data.append(
                                {
                                    "Sensor": sensor_name,
                                    "Activity": activity_name,
                                    "Overlap Duration (hours)": duration_hours,
                                }
                            )
                    df_matrix = pd.DataFrame(matrix_data)
                    st.dataframe(df_matrix, width="stretch", hide_index=True)
                except Exception as e:
                    st.error(f"Error computing matrix: {e}")
        else:
            st.info("Click button to compute overlap table")
    with col_plot:
        if st.button("Visualize Overlap Matrix"):
            with st.spinner("Generating overlap plot..."):
                try:
                    fig = dataset.get_sensor_activity_overlap_plot()
                    st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.error(f"Error generating plot: {e}")
        else:
            st.info("Click button to visualize overlaps")
with tab5:
    st.header("Temporal Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Summary")
        num_days = st.slider("Number of days to analyze", 1, 30, 7, key="daily_summary")
        if st.button("Compute Daily Summary"):
            with st.spinner("Computing daily summary..."):
                daily = dataset.get_daily_summary(num_days=num_days)
                for date, info in daily.items():
                    st.write(f"**{date}**")
                    st.write(f"- Activities: {info['total_activities']}")
                    st.write(f"- Activations: {info['total_activations']}")
                    with st.expander("Activity Breakdown"):
                        for activity, count in info["activity_breakdown"].items():
                            st.write(f"  {activity}: {count}")
        else:
            st.info("Click button to compute daily summary")
    with col2:
        st.subheader("Time Range Query")
        time_span = dataset.get_time_span()
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Start date", value=time_span[0].date() if time_span[0] else None
            )
        with col_end:
            end_date = st.date_input(
                "End date",
                value=(time_span[0] + timedelta(days=7)).date()
                if time_span[0]
                else None,
            )
        if st.button("Query Time Range"):
            if start_date and end_date:
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                activations = dataset.get_activations_in_time_range(start_dt, end_dt)
                activities = dataset.get_activities_in_time_range(start_dt, end_dt)
                st.write(f"**Activations**: {len(activations.data)}")
                st.write(f'**Activities**: {len(activities)}"')
                if activations:
                    df_activations = pd.DataFrame(
                        [
                            {
                                "Sensor": r.sensor.name,
                                "Start": r.interval.start,
                                "End": r.interval.end,
                                "Duration (s)": round(r.duration, 2),
                            }
                            for r in activations.data[:100]
                        ]
                    )
                    st.subheader("Sensor Activations (first 100)")
                    st.dataframe(df_activations, width="stretch", hide_index=True)
                if activities:
                    df_activities = pd.DataFrame(
                        [
                            {
                                "Activity": a.name,
                                "Start": a.interval.start,
                                "End": a.interval.end,
                                "Duration (s)": round(a.duration, 2),
                            }
                            for a in activities[:100]
                        ]
                    )
                    st.subheader("Activities (first 100)")
                    st.dataframe(df_activities, width="stretch", hide_index=True)
            else:
                st.warning("Please select valid dates")
with tab6:
    st.header("Top-N Rankings")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Sensors")
        k = st.slider("Top K sensors", 5, 50, 10, key="top_k_sensors")
        metric = st.radio(
            "Rank by",
            ["Activations Count", "Total Duration", "Average Duration"],
            key="sensor_metric",
        )
        if metric == "Activations Count":
            top = dataset.get_top_sensors_by_count(k=k)
            df_top = pd.DataFrame(
                [{"Sensor": name, "Count": count} for name, count in top]
            )
        elif metric == "Total Duration":
            top = dataset.get_top_sensors_by_duration(k=k)
            df_top = pd.DataFrame(
                [
                    {"Sensor": name, "Duration (hours)": round(dur / 3600, 2)}
                    for name, dur in top
                ]
            )
        else:
            top = dataset.get_top_sensors_by_average_duration(k=k)
            df_top = pd.DataFrame(
                [
                    {"Sensor": name, "Avg Duration (s)": round(dur, 2)}
                    for name, dur in top
                ]
            )
        st.dataframe(df_top, width="stretch", hide_index=True)
    with col2:
        st.subheader("Top Activities")
        k = st.slider("Top K activities", 5, 50, 10, key="top_k_activities")

        metric = st.radio(
            "Rank by", ["Occurrence Count", "Total Duration"], key="activity_metric"
        )
        if metric == "Occurrence Count":
            top = dataset.get_top_activities_by_count(k=k)
            df_top = pd.DataFrame(
                [{"Activity": name, "Count": count} for name, count in top]
            )
        else:
            top = dataset.get_top_activities_by_duration(k=k)
            df_top = pd.DataFrame(
                [
                    {"Activity": name, "Duration (hours)": round(dur / 3600, 2)}
                    for name, dur in top
                ]
            )
        st.dataframe(df_top, width="stretch", hide_index=True)
    st.subheader("Activity Transitions")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        k = st.slider("Top K transitions", 5, 50, 10, key="top_transitions")
    with col2:
        max_gap = st.slider(
            "Max gap between activities (seconds)", 10, 300, 60, key="max_gap"
        )
    with col3:
        compute_transitions = st.button("Compute Transitions")
    if compute_transitions:
        with st.spinner("Computing activity transitions..."):
            top_transitions = dataset.get_top_activity_transitions(
                k=k, max_gap_seconds=max_gap
            )
            df_transitions = pd.DataFrame(
                [
                    {"Transition": trans, "Count": count}
                    for trans, count in top_transitions
                ]
            )
            st.dataframe(df_transitions, width="stretch", hide_index=True)
    else:
        st.info("Click button to compute activity transitions")
    st.subheader("Longest Sensor Activations")
    k = st.slider("Top K sensor activations", 5, 50, 10, key="longest_sensors")
    if st.button("Compute Longest Activations"):
        with st.spinner("Computing longest sensor activations..."):
            longest_sensors = dataset.get_longest_sensor_activations(k=k)
            df_longest_sensors = pd.DataFrame(
                [
                    {
                        "Sensor": activation.sensor.name,
                        "Duration (hours)": round(activation.duration / 3600, 2),
                        "Start": activation.interval.start,
                        "End": activation.interval.end,
                    }
                    for activation in longest_sensors.data
                ]
            )
            st.dataframe(df_longest_sensors, width="stretch", hide_index=True)
    else:
        st.info("Click button to compute longest sensor activations")
with tab7:
    st.header("Dataset Metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Browse by Area")
        areas = dataset.get_areas()
        if areas:
            selected_area = st.selectbox("Select a area", areas)
            sensors_in_area = dataset.get_sensors_by_area(selected_area)
            st.write(f"**{selected_area}** - {len(sensors_in_area)} sensors")
            for sensor in sensors_in_area:
                metadata = dataset.get_sensor_metadata(sensor)
                if metadata:
                    st.write(
                        f"- **{sensor}**: {metadata.get('type', 'N/A')} on {metadata.get('object', 'N/A')}"
                    )
                else:
                    st.write(f"- **{sensor}**")
        else:
            st.info("No area metadata available")
    with col2:
        st.subheader("Browse by Sensor Type")
        sensor_types = dataset.get_sensor_types()
        if sensor_types:
            selected_type = st.selectbox("Select a type", sensor_types)
            sensors_of_type = dataset.get_sensors_by_type(selected_type)
            st.write(f"**{selected_type}** - {len(sensors_of_type)} sensors")
            for sensor in sensors_of_type:
                metadata = dataset.get_sensor_metadata(sensor)
                if metadata:
                    st.write(
                        f"- **{sensor}**: {metadata.get('area', 'N/A')} ({metadata.get('object', 'N/A')})"
                    )
                else:
                    st.write(f"- **{sensor}**")
        else:
            st.info("No sensor type metadata available")
    st.subheader("All Sensors Metadata")
    all_metadata = dataset.get_all_sensors_metadata()
    if all_metadata:
        metadata_list = []
        for sensor_name, meta in all_metadata.items():
            if isinstance(meta, dict):
                metadata_list.append(
                    {
                        "Sensor": sensor_name,
                        "Type": meta.get("type", "N/A"),
                        "Area": meta.get("area", "N/A"),
                        "Object": meta.get("object", "N/A"),
                    }
                )
        df_metadata = pd.DataFrame(metadata_list)
        st.dataframe(df_metadata, width="stretch", hide_index=True)
    else:
        st.info("No sensor metadata available")
with tab8:
    st.header("Event Gaps Analysis")
    st.caption(
        "Find consecutive event pairs whose inter-event gap exceeds a threshold. "
        "Useful for detecting periods of inactivity in the dataset."
    )
    gap_threshold = st.slider(
        "Max Gap Threshold (seconds)",
        min_value=1,
        max_value=3600,
        value=60,
        step=5,
        help="Events separated by more than this threshold will be flagged",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sensor Event Gaps")
        if st.button("Find Sensor Gaps"):
            with st.spinner("Computing sensor event gaps..."):
                gaps = dataset.get_sensor_event_gaps(max_gap_seconds=gap_threshold)
                if gaps.empty:
                    st.info(
                        f"No sensor event gaps exceeding {gap_threshold} seconds found."
                    )
                else:
                    st.write(f"**Found {len(gaps)} gap(s)**")
                    df_gaps = gaps.copy()
                    df_gaps["gap_seconds"] = df_gaps["gap_seconds"].round(2)
                    st.dataframe(
                        df_gaps[
                            [
                                "prev_name",
                                "prev_end",
                                "next_name",
                                "next_start",
                                "gap_seconds",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                    st.download_button(
                        label="Download Sensor Gaps (CSV)",
                        data=gaps.to_csv(index=False),
                        file_name="sensor_event_gaps.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Click button to find sensor event gaps")
with tab9:
    st.header("Timeline Visualization")
    st.caption(
        "Interactive timeline showing sensor activations and activity occurrences. "
        "Select sensors and activities to display in separate tracks."
    )
    col_select1, col_select2 = st.columns(2)
    with col_select1:
        st.subheader("Select Sensors")
        all_sensors = dataset.get_sensor_names()
        selected_sensors = st.multiselect(
            "Choose sensors to display",
            all_sensors,
            default=all_sensors[: min(5, len(all_sensors))],
            key="timeline_sensors",
        )
    with col_select2:
        st.subheader("Select Activities")
        all_activities = dataset.get_activity_names()
        selected_activities = st.multiselect(
            "Choose activities to display",
            all_activities,
            default=all_activities[: min(5, len(all_activities))],
            key="timeline_activities",
        )
    if st.button("Generate Timeline"):
        if not selected_sensors and not selected_activities:
            st.warning("Please select at least one sensor or activity")
        else:
            with st.spinner("Generating timeline..."):
                try:
                    sensors_df, activities_df = dataset.get_timeline_data(
                        selected_sensors=selected_sensors if selected_sensors else None,
                        selected_activities=selected_activities
                        if selected_activities
                        else None,
                    )
                    all_events = []
                    if not sensors_df.empty:
                        all_events.extend(sensors_df["start"].tolist())
                        all_events.extend(sensors_df["end"].tolist())
                    if not activities_df.empty:
                        all_events.extend(activities_df["start"].tolist())
                        all_events.extend(activities_df["end"].tolist())

                    if not all_events:
                        st.warning("No data available for selected sensors/activities")
                    else:
                        min_time = min(all_events)
                        max_time = max(all_events)
                        fig = go.Figure()
                        track_idx: float = 0
                        sensor_names = (
                            sorted(set(sensors_df["name"].tolist()))
                            if not sensors_df.empty
                            else []
                        )
                        for sensor_name in sensor_names:
                            sensor_events = sensors_df[
                                sensors_df["name"] == sensor_name
                            ]
                            for _, event in sensor_events.iterrows():
                                start_str = event["start"].strftime("%Y-%m-%d %H:%M:%S")
                                end_str = event["end"].strftime("%Y-%m-%d %H:%M:%S")
                                hover_text = (
                                    f"<b>{sensor_name}</b><br>"
                                    f"Start: {start_str}<br>"
                                    f"End: {end_str}<br>"
                                    f"Duration: {event['duration']:.1f}s"
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=[event["start"], event["end"]],
                                        y=[track_idx, track_idx],
                                        mode="lines",
                                        line=dict(color="steelblue", width=20),
                                        name=f"Sensor: {sensor_name}",
                                        hovertext=hover_text,
                                        hoverinfo="text",
                                        showlegend=False,
                                    )
                                )
                            track_idx += 1
                        track_idx += 0.5
                        activity_names = (
                            sorted(set(activities_df["name"].tolist()))
                            if not activities_df.empty
                            else []
                        )
                        for activity_name in activity_names:
                            activity_events = activities_df[
                                activities_df["name"] == activity_name
                            ]
                            for _, event in activity_events.iterrows():
                                start_str = event["start"].strftime("%Y-%m-%d %H:%M:%S")
                                end_str = event["end"].strftime("%Y-%m-%d %H:%M:%S")
                                hover_text = (
                                    f"<b>{activity_name}</b><br>"
                                    f"Start: {start_str}<br>"
                                    f"End: {end_str}<br>"
                                    f"Duration: {event['duration']:.1f}s"
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=[event["start"], event["end"]],
                                        y=[track_idx, track_idx],
                                        mode="lines",
                                        line=dict(color="darkorange", width=20),
                                        name=f"Activity: {activity_name}",
                                        hovertext=hover_text,
                                        hoverinfo="text",
                                        showlegend=False,
                                    )
                                )
                            track_idx += 1
                        y_labels = []
                        y_positions: list[float] = []
                        track_idx = 0
                        for sensor_name in sensor_names:
                            y_labels.append(f"🔌 {sensor_name}")
                            y_positions.append(track_idx)
                            track_idx += 1

                        track_idx += 0.5
                        for activity_name in activity_names:
                            y_labels.append(f"📌 {activity_name}")
                            y_positions.append(track_idx)
                            track_idx += 1
                        fig.update_layout(
                            title="Smart Home Timeline: Sensors & Activities",
                            xaxis_title="Time",
                            yaxis_title="Tracks",
                            height=max(
                                400,
                                len(sensor_names) * 40 + len(activity_names) * 40 + 200,
                            ),
                            hovermode="closest",
                            xaxis=dict(
                                range=[min_time, max_time],
                                type="date",
                            ),
                            yaxis=dict(
                                ticktext=y_labels,
                                tickvals=y_positions,
                                showgrid=True,
                                gridwidth=1,
                                gridcolor="lightgray",
                            ),
                            showlegend=False,
                            template="plotly_white",
                        )

                        st.plotly_chart(fig, width="stretch")
                        st.subheader("Timeline Statistics")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Sensors", len(sensor_names))
                        with col_stat2:
                            st.metric("Activities", len(activity_names))
                        with col_stat3:
                            st.metric(
                                "Sensor Events",
                                len(sensors_df) if not sensors_df.empty else 0,
                            )
                        with col_stat4:
                            st.metric(
                                "Activity Events",
                                len(activities_df) if not activities_df.empty else 0,
                            )
                        if not sensors_df.empty:
                            st.subheader("Sensor Events")
                            df_display = sensors_df[
                                ["name", "start", "end", "duration"]
                            ].copy()
                            df_display["duration_seconds"] = df_display[
                                "duration"
                            ].round(2)
                            st.dataframe(
                                df_display[
                                    ["name", "start", "end", "duration_seconds"]
                                ],
                                width="stretch",
                                hide_index=True,
                            )

                        if not activities_df.empty:
                            st.subheader("Activity Events")
                            df_display = activities_df[
                                ["name", "start", "end", "duration"]
                            ].copy()
                            df_display["duration_seconds"] = df_display[
                                "duration"
                            ].round(2)
                            st.dataframe(
                                df_display[
                                    ["name", "start", "end", "duration_seconds"]
                                ],
                                width="stretch",
                                hide_index=True,
                            )

                except Exception as e:
                    st.error(f"Error generating timeline: {e}")
    else:
        st.info(
            "Select sensors and/or activities, then click the button to generate the timeline visualization"
        )
with tab10:
    st.header("📋 Dataset Details")
    st.write(
        "Comprehensive statistical details about the smart home dataset, including sensor and activity metrics."
    )

    try:
        with st.spinner("Computing dataset details..."):
            details_df = dataset.compute_dataset_details()
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Number of Sensors",
                details_df[details_df["Metric"] == "Number of Sensors"]["Value"].values[
                    0
                ],
            )
            st.metric(
                "Number of Sensor Intervals",
                details_df[details_df["Metric"] == "Number of Sensor Intervals"][
                    "Value"
                ].values[0],
            )
            st.metric(
                "Number of Activities",
                details_df[details_df["Metric"] == "Number of Activities"][
                    "Value"
                ].values[0],
            )
            st.metric(
                "Number of Activity Intervals",
                details_df[details_df["Metric"] == "Number of Activity Intervals"][
                    "Value"
                ].values[0],
            )

        with col2:
            st.metric(
                "Total Duration (days)",
                details_df[details_df["Metric"] == "Total Duration (days)"][
                    "Value"
                ].values[0],
            )
            st.metric(
                "Max Gap Sensors (hours)",
                details_df[
                    details_df["Metric"] == "Max Time Gap Between Sensors (hours)"
                ]["Value"].values[0],
            )
            st.metric(
                "Max Gap Activities (hours)",
                details_df[
                    details_df["Metric"] == "Max Time Gap Between Activities (hours)"
                ]["Value"].values[0],
            )
            st.metric(
                "Number of Null Periods",
                details_df[details_df["Metric"] == "Number of Null Periods"][
                    "Value"
                ].values[0],
            )

        st.divider()
        st.subheader("Detailed Metrics Table")
        st.dataframe(details_df, width="stretch", hide_index=True)
        st.divider()
        st.subheader("Insights")
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        with col_insight1:
            avg_activations_per_sensor = details_df[
                details_df["Metric"] == "Number of Sensor Intervals"
            ]["Value"].values[0] / max(
                1,
                details_df[details_df["Metric"] == "Number of Sensors"]["Value"].values[
                    0
                ],
            )
            st.info(
                f"**Avg. Activations per Sensor**: {avg_activations_per_sensor:.1f}"
            )

        with col_insight2:
            avg_activities_per_annotation = details_df[
                details_df["Metric"] == "Number of Activity Intervals"
            ]["Value"].values[0] / max(
                1,
                details_df[details_df["Metric"] == "Number of Activities"][
                    "Value"
                ].values[0],
            )
            st.info(
                f"**Avg. Intervals per Activity**: {avg_activities_per_annotation:.1f}"
            )

        with col_insight3:
            data_completeness = 1.0 - (
                details_df[details_df["Metric"] == "Number of Null Periods"][
                    "Value"
                ].values[0]
                / max(
                    1,
                    details_df[details_df["Metric"] == "Number of Sensor Intervals"][
                        "Value"
                    ].values[0],
                )
            )
            st.info(f"**Data Continuity**: {data_completeness * 100:.1f}%")

    except Exception as e:
        st.error(f"Error computing dataset details: {e}")
