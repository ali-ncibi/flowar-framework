import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ...models import (
    ActivityOccurrence,
    DatasetStatistics,
    Event,
    EventStream,
    FunctionalArea,
    Interval,
    IntervalType,
    Sensor,
    SensorActivation,
    SensorActivations,
    SensorMeta,
    SensorObject,
    SensorType,
    Timestamp,
)


def _as_series(value: Any) -> pd.Series:
    # pandas 3 types DataFrame[...] as a wide union; a column key always yields a Series.
    return cast(pd.Series, value)


def _as_frame(value: Any) -> pd.DataFrame:
    # A boolean mask or column list always yields a DataFrame.
    return cast(pd.DataFrame, value)


class SmartHomeDataset:
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        sensors_df: pd.DataFrame | None = None,
        activities_df: pd.DataFrame | None = None,
        metadata: dict | None = None,
        num_days: int | None = None,
    ):
        self.num_days = num_days
        self.name = Path(dataset_path).name if dataset_path is not None else "in-memory"
        if dataset_path is not None:
            self.dataset_path: Path | None = Path(dataset_path)
            self._validate_dataset_path()
            self._sensors_df: pd.DataFrame
            self._activities_df: pd.DataFrame
            self._metadata: dict = {}
            self._load_data()
        else:
            self.dataset_path = None
            self._sensors_df = (
                sensors_df.copy() if sensors_df is not None else pd.DataFrame()
            )
            self._activities_df = (
                activities_df.copy() if activities_df is not None else pd.DataFrame()
            )
            self._metadata = dict(metadata) if metadata is not None else {}
            if not self._sensors_df.empty:
                self._sensors_df["start"] = pd.to_datetime(self._sensors_df["start"])
                self._sensors_df["end"] = pd.to_datetime(self._sensors_df["end"])
            if not self._activities_df.empty:
                self._activities_df["start"] = pd.to_datetime(
                    self._activities_df["start"]
                )
                self._activities_df["end"] = pd.to_datetime(self._activities_df["end"])

    @staticmethod
    def get_total_days(dataset_path: str | Path) -> int:
        """
        Get the total number of days in a dataset without loading all data.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Total number of days in the dataset (rounded up)
        """
        dataset_path = Path(dataset_path)

        # Read CSV files to get time span
        sensors_df = pd.read_csv(dataset_path / "sensors.csv")
        activities_df = pd.read_csv(dataset_path / "activities.csv")

        sensors_df["start"] = pd.to_datetime(sensors_df["start"])
        sensors_df["end"] = pd.to_datetime(sensors_df["end"])
        activities_df["start"] = pd.to_datetime(activities_df["start"])
        activities_df["end"] = pd.to_datetime(activities_df["end"])

        # Get time span
        start_times = []
        end_times = []
        if not sensors_df.empty:
            start_times.append(sensors_df["start"].min())
            end_times.append(sensors_df["end"].max())
        if not activities_df.empty:
            start_times.append(activities_df["start"].min())
            end_times.append(activities_df["end"].max())

        if not start_times or not end_times:
            return 0

        start_time = min(start_times)
        end_time = max(end_times)

        # Calculate total days
        total_duration = end_time - start_time
        total_days = int(total_duration.total_seconds() / 86400) + 1

        return max(1, total_days)

    def _validate_dataset_path(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path cannot be None when called")
        required_files = ["sensors.csv", "activities.csv", "meta.toml"]
        for file in required_files:
            if not (self.dataset_path / file).exists():
                raise FileNotFoundError(
                    f"Missing required file: {file} in {self.dataset_path}"
                )

    def _filter_by_days(self, num_days: int) -> None:
        if self._sensors_df.empty and self._activities_df.empty:
            return
        timestamps = []
        if not self._sensors_df.empty:
            timestamps.append(self._sensors_df["start"].min())
        if not self._activities_df.empty:
            timestamps.append(self._activities_df["start"].min())
        if (
            not timestamps
        ):  # pragma: no cover  # unreachable: both frames non-empty here
            return
        start_time = min(timestamps)
        end_time = start_time + timedelta(days=num_days)
        self._sensors_df = _as_frame(
            self._sensors_df[self._sensors_df["start"] < end_time]
        )
        self._activities_df = _as_frame(
            self._activities_df[self._activities_df["start"] < end_time]
        )

    def _load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be set to load data from file")
        self._sensors_df = pd.read_csv(self.dataset_path / "sensors.csv")
        self._sensors_df["start"] = pd.to_datetime(self._sensors_df["start"])
        self._sensors_df["end"] = pd.to_datetime(self._sensors_df["end"])
        self._activities_df = pd.read_csv(self.dataset_path / "activities.csv")
        self._activities_df["start"] = pd.to_datetime(self._activities_df["start"])
        self._activities_df["end"] = pd.to_datetime(self._activities_df["end"])
        if self.num_days is not None and self.num_days > 0:
            self._filter_by_days(self.num_days)
        with open(self.dataset_path / "meta.toml", "rb") as f:
            self._metadata = tomllib.load(f)

    def _get_functional_area_for_sensor(self, sensor_name: str) -> FunctionalArea:
        """Get functional area for sensor from metadata."""
        sensor_metadata = self.get_sensor_metadata(sensor_name)
        if sensor_metadata and isinstance(sensor_metadata, dict):
            area = sensor_metadata.get("area")
            if area:
                return FunctionalArea(area)
        return FunctionalArea.UNDEFINED

    def _get_type_for_sensor(self, sensor_name: str) -> SensorType:
        """Get sensor type from metadata."""
        sensor_metadata = self.get_sensor_metadata(sensor_name)
        if sensor_metadata and isinstance(sensor_metadata, dict):
            sensor_type = sensor_metadata.get("type")
            if sensor_type:
                return SensorType(sensor_type)
        return SensorType.UNDEFINED

    def _get_object_for_sensor(self, sensor_name: str) -> SensorObject:
        """Get sensor object from metadata."""
        sensor_metadata = self.get_sensor_metadata(sensor_name)
        if sensor_metadata and isinstance(sensor_metadata, dict):
            sensor_object = sensor_metadata.get("object")
            if sensor_object:
                return SensorObject(sensor_object)
        return SensorObject.UNDEFINED

    def _sensor_activation_from_row(self, row: pd.Series) -> SensorActivation:
        name = cast(str, row["name"])
        return SensorActivation(
            interval=Interval(
                start=cast(Timestamp, row["start"]),
                end=cast(Timestamp, row["end"]),
                interval_type=IntervalType.LEFT_OPEN,
            ),
            sensor=Sensor(
                name=name,
                meta=SensorMeta(
                    sensor_type=self._get_type_for_sensor(name),
                    area=self._get_functional_area_for_sensor(name),
                    sensor_object=self._get_object_for_sensor(name),
                ),
            ),
        )

    def _activity_occurrence_from_row(self, row: pd.Series) -> ActivityOccurrence:
        return ActivityOccurrence(
            interval=Interval(
                start=cast(Timestamp, row["start"]),
                end=cast(Timestamp, row["end"]),
                interval_type=IntervalType.LEFT_OPEN,
            ),
            name=cast(str, row["name"]),
        )

    @property
    def sensor_activations(self) -> SensorActivations:
        return SensorActivations(
            data=[
                self._sensor_activation_from_row(row)
                for _, row in self._sensors_df.iterrows()
            ]
        )

    @property
    def activity_occurrences(self) -> list[ActivityOccurrence]:
        return [
            self._activity_occurrence_from_row(row)
            for _, row in self._activities_df.iterrows()
        ]

    @property
    def metadata(self) -> dict:
        return self._metadata

    def get_sensor_names(self) -> list[str]:
        return sorted(self._sensors_df["name"].unique().tolist())

    def get_sensors_by_activity(self, activity_name: str) -> set[str]:
        activity_rows = self._activities_df[
            self._activities_df["name"] == activity_name
        ]
        if activity_rows.empty:
            return set()
        active_sensors = set()
        for _, activity in activity_rows.iterrows():
            activity_interval = Interval(
                start=cast(Timestamp, activity["start"]),
                end=cast(Timestamp, activity["end"]),
                interval_type=IntervalType.LEFT_OPEN,
            )
            for _, sensor in self._sensors_df.iterrows():
                sensor_interval = Interval(
                    start=cast(Timestamp, sensor["start"]),
                    end=cast(Timestamp, sensor["end"]),
                    interval_type=IntervalType.LEFT_OPEN,
                )
                if sensor_interval.overlaps(activity_interval):
                    active_sensors.add(cast(str, sensor["name"]))
        return active_sensors

    def get_sensor_activation_count(self, sensor_name: str) -> int:
        return len(self._sensors_df[self._sensors_df["name"] == sensor_name])

    def get_sensor_activations_by_sensor(self, sensor_name: str) -> SensorActivations:
        rows = self._sensors_df[self._sensors_df["name"] == sensor_name]
        return SensorActivations(
            data=[self._sensor_activation_from_row(row) for _, row in rows.iterrows()]
        )

    def get_sensor_statistics(self) -> dict[str, int]:
        stats = self._sensors_df["name"].value_counts()
        return {
            str(name): int(count)
            for name, count in sorted(stats.items(), key=lambda item: str(item[0]))
        }

    def get_sensor_duration_statistics(self) -> dict[str, float]:
        df = self._sensors_df.copy()
        df["duration"] = df.apply(
            lambda row: (
                Interval(
                    start=cast(Timestamp, row["start"]),
                    end=cast(Timestamp, row["end"]),
                    interval_type=IntervalType.LEFT_OPEN,
                ).duration
            ),
            axis=1,
        )
        stats = _as_series(df.groupby("name")["duration"].sum())
        return {
            str(name): float(duration)
            for name, duration in sorted(stats.items(), key=lambda item: str(item[0]))
        }

    def get_activity_names(self) -> list[str]:
        return sorted(self._activities_df["name"].unique().tolist())

    def get_activities_by_sensor(self, sensor_name: str) -> set[str]:
        sensor_rows = self._sensors_df[self._sensors_df["name"] == sensor_name]
        if sensor_rows.empty:
            return set()
        activities = set()
        for _, sensor in sensor_rows.iterrows():
            sensor_interval = Interval(
                start=cast(Timestamp, sensor["start"]),
                end=cast(Timestamp, sensor["end"]),
                interval_type=IntervalType.LEFT_OPEN,
            )
            for _, activity in self._activities_df.iterrows():
                activity_interval = Interval(
                    start=cast(Timestamp, activity["start"]),
                    end=cast(Timestamp, activity["end"]),
                    interval_type=IntervalType.LEFT_OPEN,
                )
                if sensor_interval.overlaps(activity_interval):
                    activities.add(cast(str, activity["name"]))
        return activities

    def get_activity_annotation_count(self, activity_name: str) -> int:
        return len(self._activities_df[self._activities_df["name"] == activity_name])

    def get_activity_annotations_by_activity(
        self, activity_name: str
    ) -> list[ActivityOccurrence]:
        rows = self._activities_df[self._activities_df["name"] == activity_name]
        return [self._activity_occurrence_from_row(row) for _, row in rows.iterrows()]

    def get_activity_statistics(self) -> dict[str, int]:
        stats = self._activities_df["name"].value_counts()
        return {
            str(name): int(count)
            for name, count in sorted(stats.items(), key=lambda item: str(item[0]))
        }

    def get_activity_duration_statistics(self) -> dict[str, float]:
        df = self._activities_df.copy()
        df["duration"] = df.apply(
            lambda row: (
                Interval(
                    start=cast(Timestamp, row["start"]),
                    end=cast(Timestamp, row["end"]),
                    interval_type=IntervalType.LEFT_OPEN,
                ).duration
            ),
            axis=1,
        )
        stats = _as_series(df.groupby("name")["duration"].sum())
        return {
            str(name): float(duration)
            for name, duration in sorted(stats.items(), key=lambda item: str(item[0]))
        }

    def _compute_overlap_duration(
        self,
        interval1: Interval,
        interval2: Interval,
    ) -> float:
        """Helper method that uses Interval.overlap_duration()."""
        return interval1.overlap_duration(interval2)

    def get_sensor_activity_matrix(self) -> dict[str, dict[str, float]]:
        matrix: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for _, sensor_row in self._sensors_df.iterrows():
            sensor_name = cast(str, sensor_row["name"])
            sensor_interval = Interval(
                start=cast(Timestamp, sensor_row["start"]),
                end=cast(Timestamp, sensor_row["end"]),
                interval_type=IntervalType.LEFT_OPEN,
            )
            for _, activity_row in self._activities_df.iterrows():
                activity_name = cast(str, activity_row["name"])
                activity_interval = Interval(
                    start=cast(Timestamp, activity_row["start"]),
                    end=cast(Timestamp, activity_row["end"]),
                    interval_type=IntervalType.LEFT_OPEN,
                )
                if sensor_interval.overlaps(activity_interval):
                    overlap = sensor_interval.overlap_duration(activity_interval)
                    matrix[sensor_name][activity_name] += overlap
        return {k: dict(v) for k, v in matrix.items()}

    def get_sensor_activity_overlap_plot(self):
        matrix = defaultdict(lambda: defaultdict(lambda: {"duration": 0.0, "count": 0}))
        for _, sensor_row in self._sensors_df.iterrows():
            sensor_name = cast(str, sensor_row["name"])
            sensor_interval = Interval(
                start=cast(Timestamp, sensor_row["start"]),
                end=cast(Timestamp, sensor_row["end"]),
                interval_type=IntervalType.LEFT_OPEN,
            )
            for _, activity_row in self._activities_df.iterrows():
                activity_name = cast(str, activity_row["name"])
                activity_interval = Interval(
                    start=cast(Timestamp, activity_row["start"]),
                    end=cast(Timestamp, activity_row["end"]),
                    interval_type=IntervalType.LEFT_OPEN,
                )
                if sensor_interval.overlaps(activity_interval):
                    overlap = sensor_interval.overlap_duration(activity_interval)
                    if overlap > 0:
                        matrix[sensor_name][activity_name]["duration"] += overlap
                        matrix[sensor_name][activity_name]["count"] += 1
        sensors = sorted(self.get_sensor_names())
        activities = sorted(self.get_activity_names())
        overlap_values = []
        hover_texts = []
        for activity in activities:
            row_values = []
            row_hovers = []
            for sensor in sensors:
                if sensor in matrix and activity in matrix[sensor]:
                    data = matrix[sensor][activity]
                    duration_hours = data["duration"] / 3600
                    count = data["count"]
                    row_values.append(1)
                    row_hovers.append(
                        f"Activity: {activity}<br>"
                        f"Sensor: {sensor}<br>"
                        f"Total Duration: {duration_hours:.2f} hours<br>"
                        f"Overlap Count: {count}"
                    )
                else:
                    row_values.append(0)
                    row_hovers.append(
                        f"Activity: {activity}<br>Sensor: {sensor}<br>No overlap"
                    )
            overlap_values.append(row_values)
            hover_texts.append(row_hovers)
        fig = go.Figure(
            data=go.Heatmap(
                z=overlap_values,
                x=sensors,
                y=activities,
                colorscale=[[0, "white"], [1, "#1f77b4"]],
                showscale=False,
                hovertext=hover_texts,
                hoverinfo="text",
                text=[
                    ["X" if val == 1 else "" for val in row] for row in overlap_values
                ],
                textfont=dict(size=14, color="black"),
            )
        )
        fig.update_layout(
            title="Sensor-Activity Overlap Matrix",
            xaxis_title="Sensors",
            yaxis_title="Activities",
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
            height=max(400, len(activities) * 30 + 100),
            width=max(600, len(sensors) * 50 + 100),
            hovermode="closest",
        )

        return fig

    def get_time_span(self) -> tuple[datetime | None, datetime | None]:
        if self._sensors_df.empty and self._activities_df.empty:
            return None, None
        start_times = []
        end_times = []
        if not self._sensors_df.empty:
            start_times.append(self._sensors_df["start"].min())
            end_times.append(self._sensors_df["end"].max())
        if not self._activities_df.empty:
            start_times.append(self._activities_df["start"].min())
            end_times.append(self._activities_df["end"].max())
        return min(start_times), max(end_times)

    def get_sensor_event_gaps(self, max_gap_seconds: float) -> pd.DataFrame:
        if max_gap_seconds < 0:
            raise ValueError(
                "max_gap_seconds must be non-embedding_w2v_negative_sampling"
            )
        events = _as_frame(self._sensors_df[["name", "start", "end"]]).copy()
        if events.empty or len(events) < 2:
            return pd.DataFrame(
                columns=[
                    "prev_name",
                    "prev_start",
                    "prev_end",
                    "next_name",
                    "next_start",
                    "next_end",
                    "gap_seconds",
                ]
            )
        events = events.sort_values("start", kind="mergesort").reset_index(drop=True)
        prev_name = events["name"].shift(1)
        prev_start = events["start"].shift(1)
        prev_end = events["end"].shift(1)
        gap_seconds = (
            _as_series(events["start"]) - _as_series(prev_end)
        ).dt.total_seconds()
        mask = gap_seconds > max_gap_seconds
        result = pd.DataFrame(
            {
                "prev_name": prev_name[mask],
                "prev_start": prev_start[mask],
                "prev_end": prev_end[mask],
                "next_name": events.loc[mask, "name"],
                "next_start": events.loc[mask, "start"],
                "next_end": events.loc[mask, "end"],
                "gap_seconds": gap_seconds[mask],
            }
        ).reset_index(drop=True)

        return result

    def get_sensor_duration_kde_plot(self, sensor_name: str):
        sensor_activations = self.get_sensor_activations_by_sensor(sensor_name)
        if not sensor_activations or len(sensor_activations.data) < 2:
            raise ValueError(
                f"Sensor '{sensor_name}' has fewer than 2 activations. "
                "Cannot compute meaningful distributions."
            )
        activation_durations = [r.duration for r in sensor_activations.data]
        deactivation_durations = []
        for i in range(len(sensor_activations.data) - 1):
            gap_seconds = (
                sensor_activations.data[i + 1].interval.start
                - sensor_activations.data[i].interval.end
            ).total_seconds()
            if gap_seconds >= 0:
                deactivation_durations.append(gap_seconds)
        fig_act = go.Figure()
        if (
            activation_durations
        ):  # pragma: no cover  # always true: >=2 activations enforced above
            act_array = np.array(activation_durations)
            kde_act = stats.gaussian_kde(act_array)
            x_act = np.linspace(act_array.min(), act_array.max(), 200)
            fig_act.add_trace(
                go.Scatter(
                    x=x_act,
                    y=kde_act(x_act),
                    mode="lines",
                    name="Activation Duration (On)",
                    line=dict(color="blue", width=2),
                    fill="tozeroy",
                )
            )
        fig_act.update_layout(
            title=f"Activation Duration (On) - {sensor_name}",
            xaxis_title="Duration (seconds)",
            yaxis_title="Probability Density",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )
        fig_deact = go.Figure()
        if deactivation_durations:
            deact_array = np.array(deactivation_durations)
            kde_deact = stats.gaussian_kde(deact_array)
            x_deact = np.linspace(deact_array.min(), deact_array.max(), 200)
            fig_deact.add_trace(
                go.Scatter(
                    x=x_deact,
                    y=kde_deact(x_deact),
                    mode="lines",
                    name="Deactivation Duration (Off)",
                    line=dict(color="red", width=2),
                    fill="tozeroy",
                )
            )
        fig_deact.update_layout(
            title=f"Deactivation Duration (Off) - {sensor_name}",
            xaxis_title="Duration (seconds)",
            yaxis_title="Probability Density",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )

        return fig_act, fig_deact

    def get_activations_in_time_range(
        self, start: datetime, end: datetime
    ) -> SensorActivations:
        time_range_interval = Interval(
            start=start,
            end=end,
            interval_type=IntervalType.LEFT_OPEN,
        )
        activations = []
        for _, row in self._sensors_df.iterrows():
            activation = self._sensor_activation_from_row(row)
            if activation.interval.overlaps(time_range_interval):
                activations.append(activation)
        return SensorActivations(data=activations)

    def get_activities_in_time_range(
        self, start: datetime, end: datetime
    ) -> list[ActivityOccurrence]:
        time_range_interval = Interval(
            start=start,
            end=end,
            interval_type=IntervalType.LEFT_OPEN,
        )
        activities = []
        for _, row in self._activities_df.iterrows():
            occurrence = self._activity_occurrence_from_row(row)
            if occurrence.interval.overlaps(time_range_interval):
                activities.append(occurrence)
        return activities

    def get_statistics(self) -> DatasetStatistics:
        time_span = self.get_time_span()
        total_duration = (
            Interval(
                start=time_span[0],
                end=time_span[1],
                interval_type=IntervalType.LEFT_OPEN,
            ).duration
            if time_span[0] and time_span[1]
            else 0.0
        )
        return DatasetStatistics(
            num_sensors=len(self.get_sensor_names()),
            num_activities=len(self.get_activity_names()),
            num_sensor_activations=len(self._sensors_df),
            num_activity_occurrences=len(self._activities_df),
            time_span=time_span,
            total_duration_seconds=total_duration,
        )

    def print_statistics(self) -> None:
        """Print formatted dataset statistics."""
        stats = self.get_statistics()
        start_time, end_time = stats.time_span
        print("\n" + "=" * 60)
        print("SMART HOME DATASET STATISTICS")
        print("=" * 60)
        dataset_name = self.dataset_path.name if self.dataset_path else "in-memory"
        print(f"Dataset: {dataset_name}")
        print(f"Time Span: {start_time} to {end_time}")
        print(f"Total Duration: {stats.total_duration_seconds / 86400:.2f} days")
        print()
        print(f"Sensors: {stats.num_sensors}")
        print(f"Sensor Activations: {stats.num_sensor_activations}")
        print()
        print(f"Activities: {stats.num_activities}")
        print(f"Activity Annotations: {stats.num_activity_occurrences}")
        print("=" * 60)

    def get_top_sensors_by_count(self, k: int = 5) -> list[tuple[str, int]]:
        stats = self.get_sensor_statistics()
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_top_sensors_by_duration(self, k: int = 5) -> list[tuple[str, float]]:
        stats = self.get_sensor_duration_statistics()
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_top_activities_by_count(self, k: int = 5) -> list[tuple[str, int]]:
        stats = self.get_activity_statistics()
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_top_activities_by_duration(self, k: int = 5) -> list[tuple[str, float]]:
        stats = self.get_activity_duration_statistics()
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_sensor_average_durations(self) -> dict[str, float]:
        counts = self.get_sensor_statistics()
        durations = self.get_sensor_duration_statistics()
        avg_durations = {}
        for sensor, count in counts.items():
            total_duration = durations[sensor]
            avg_durations[sensor] = total_duration / count if count > 0 else 0.0
        return dict(sorted(avg_durations.items(), key=lambda x: x[1], reverse=True))

    def get_top_sensors_by_average_duration(
        self, k: int = 5
    ) -> list[tuple[str, float]]:
        avg_durations = self.get_sensor_average_durations()
        return sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_sensor_activity_diversity(self, limit: int | None = None) -> dict[str, int]:
        sensor_names = self.get_sensor_names()
        if limit:
            top_sensors = self.get_top_sensors_by_count(k=limit)
            sensor_names = [s for s, _ in top_sensors]
        diversity = {}
        for sensor in sensor_names:
            activities = self.get_activities_by_sensor(sensor)
            diversity[sensor] = len(activities)
        return dict(sorted(diversity.items(), key=lambda x: x[1], reverse=True))

    def get_top_sensors_by_activity_diversity(
        self, k: int = 5, limit: int | None = None
    ) -> list[tuple[str, int]]:
        diversity = self.get_sensor_activity_diversity(limit=limit or k * 2)
        return sorted(diversity.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_activity_transition_matrix(
        self, max_gap_seconds: float = 60.0
    ) -> dict[str, int]:
        all_activities = sorted(
            self.activity_occurrences, key=lambda a: a.interval.start
        )

        transitions = {}
        for i in range(len(all_activities) - 1):
            current = all_activities[i]
            next_activity = all_activities[i + 1]

            gap = (next_activity.interval.start - current.interval.end).total_seconds()
            if gap < max_gap_seconds:
                key = (current.name, next_activity.name)
                if key not in transitions:
                    transitions[key] = 0
                transitions[key] += 1

        sorted_transitions = sorted(
            transitions.items(), key=lambda x: x[1], reverse=True
        )
        return {f"{k[0]} -> {k[1]}": v for k, v in sorted_transitions}

    def get_top_activity_transitions(
        self, k: int = 10, max_gap_seconds: float = 60.0
    ) -> list[tuple[str, int]]:
        matrix = self.get_activity_transition_matrix(max_gap_seconds)
        return sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_longest_activities(self, k: int = 10) -> list[ActivityOccurrence]:
        all_activities = sorted(
            self.activity_occurrences, key=lambda a: a.duration, reverse=True
        )
        return all_activities[:k]

    def get_longest_sensor_activations(self, k: int = 10) -> SensorActivations:
        all_activations = sorted(
            self.sensor_activations.data, key=lambda r: r.duration, reverse=True
        )
        return SensorActivations(data=all_activations[:k])

    def get_daily_summary(self, num_days: int = 3) -> dict:
        time_span = self.get_time_span()
        if not time_span[0]:
            return {}
        summaries = {}
        current = time_span[0]
        for _day_idx in range(num_days):
            day_end = current + timedelta(days=1)
            activities = self.get_activities_in_time_range(current, day_end)
            activations = self.get_activations_in_time_range(current, day_end)
            activity_types: dict[str, int] = {}
            for activity in activities:
                activity_types[activity.name] = activity_types.get(activity.name, 0) + 1
            summaries[current.date()] = {
                "total_activities": len(activities),
                "total_activations": len(activations.data),
                "activity_breakdown": dict(
                    sorted(activity_types.items(), key=lambda x: x[1], reverse=True)
                ),
            }
            current = day_end
        return summaries

    def get_sensor_activity_correlation(
        self, num_activities: int = 3
    ) -> dict[str, set[str]]:
        activity_names = self.get_activity_names()[:num_activities]
        correlation = {}
        for activity_name in activity_names:
            sensors = self.get_sensors_by_activity(activity_name)
            correlation[activity_name] = sensors
        return correlation

    def get_sensor_metadata(self, sensor_name: str) -> dict | None:
        sensors_meta = self._metadata.get("sensors", {}).get("metadata", {})
        return sensors_meta.get(sensor_name)

    def get_sensors_by_area(self, area: str) -> list[str]:
        sensors_meta = self._metadata.get("sensors", {}).get("metadata", {})
        result = []
        for sensor_name, metadata in sensors_meta.items():
            if isinstance(metadata, dict) and metadata.get("area") == area:
                result.append(sensor_name)
        return sorted(result)

    def get_sensors_by_type(self, sensor_type: str) -> list[str]:
        sensors_meta = self._metadata.get("sensors", {}).get("metadata", {})
        result = []
        for sensor_name, metadata in sensors_meta.items():
            if isinstance(metadata, dict) and metadata.get("type") == sensor_type:
                result.append(sensor_name)
        return sorted(result)

    def get_areas(self) -> list[str]:
        sensors_meta = self._metadata.get("sensors", {}).get("metadata", {})
        areas = set()
        for metadata in sensors_meta.values():
            if isinstance(metadata, dict) and "area" in metadata:
                areas.add(metadata["area"])
        return sorted(areas)

    def get_sensor_types(self) -> list[str]:
        sensors_meta = self._metadata.get("sensors", {}).get("metadata", {})
        types = set()
        for metadata in sensors_meta.values():
            if isinstance(metadata, dict) and "type" in metadata:
                types.add(metadata["type"])
        return sorted(types)

    def get_all_sensors_metadata(self) -> dict[str, dict]:
        return self._metadata.get("sensors", {}).get("metadata", {})

    def to_event_stream(self, max_events: int | None = None) -> "EventStream":
        sensors_dict = {}
        for sensor_name in self.get_sensor_names():
            sensors_dict[sensor_name] = Sensor(
                name=sensor_name,
                meta=SensorMeta(
                    sensor_type=self._get_type_for_sensor(sensor_name),
                    area=self._get_functional_area_for_sensor(sensor_name),
                    sensor_object=self._get_object_for_sensor(sensor_name),
                ),
            )
        events = []
        for _, row in self._sensors_df.iterrows():
            sensor_name = cast(str, row["name"])
            if sensor_name not in sensors_dict:
                sensors_dict[sensor_name] = Sensor(
                    name=sensor_name,
                    meta=SensorMeta(
                        sensor_type=self._get_type_for_sensor(sensor_name),
                        area=self._get_functional_area_for_sensor(sensor_name),
                        sensor_object=self._get_object_for_sensor(sensor_name),
                    ),
                )
            sensor = sensors_dict[sensor_name]
            event_on = Event(
                sensor=sensor, timestamp=cast(Timestamp, row["start"]), value=1
            )
            events.append(event_on)
            event_off = Event(
                sensor=sensor, timestamp=cast(Timestamp, row["end"]), value=0
            )
            events.append(event_off)
            if max_events and len(events) >= max_events:
                break
        events = sorted(events, key=lambda e: (e.timestamp, e.sensor.name))
        return EventStream(events=events)

    def leave_one_day_out_split(
        self,
    ) -> list[tuple["SmartHomeDataset", "SmartHomeDataset"]]:
        time_span = self.get_time_span()
        if not time_span[0] or not time_span[1]:
            return []
        splits = []
        window_start = time_span[0]
        window_end = time_span[1]
        while window_start < window_end:
            window_end_24h = window_start + timedelta(days=1)
            sensors_in_window = (self._sensors_df["start"] >= window_start) & (
                self._sensors_df["start"] < window_end_24h
            )
            test_sensors = _as_frame(self._sensors_df[sensors_in_window]).copy()
            train_sensors = _as_frame(self._sensors_df[~sensors_in_window]).copy()
            activities_in_window = (self._activities_df["start"] >= window_start) & (
                self._activities_df["start"] < window_end_24h
            )
            test_activities = _as_frame(
                self._activities_df[activities_in_window]
            ).copy()
            train_activities = _as_frame(
                self._activities_df[~activities_in_window]
            ).copy()
            train_ds = SmartHomeDataset(
                sensors_df=train_sensors,
                activities_df=train_activities,
                metadata=self._metadata,
            )
            test_ds = SmartHomeDataset(
                sensors_df=test_sensors,
                activities_df=test_activities,
                metadata=self._metadata,
            )
            if test_sensors.empty or test_activities.empty:
                break
            splits.append((train_ds, test_ds))
            window_start = window_end_24h
        return splits

    def train_test_split(
        self, test_size: float = 0.2
    ) -> list[tuple["SmartHomeDataset", "SmartHomeDataset"]]:
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")
        time_span = self.get_time_span()
        if not time_span[0] or not time_span[1]:
            return [
                (
                    SmartHomeDataset(
                        sensors_df=self._sensors_df.copy(),
                        activities_df=self._activities_df.copy(),
                        metadata=self._metadata,
                    ),
                    SmartHomeDataset(
                        sensors_df=self._sensors_df.copy(),
                        activities_df=self._activities_df.copy(),
                        metadata=self._metadata,
                    ),
                )
            ]
        total_duration = (time_span[1] - time_span[0]).total_seconds()
        test_duration = total_duration * test_size
        split_time = time_span[1] - timedelta(seconds=test_duration)
        sensors_in_test = self._sensors_df["start"] >= split_time
        test_sensors = _as_frame(self._sensors_df[sensors_in_test]).copy()
        train_sensors = _as_frame(self._sensors_df[~sensors_in_test]).copy()
        activities_in_test = self._activities_df["start"] >= split_time
        test_activities = _as_frame(self._activities_df[activities_in_test]).copy()
        train_activities = _as_frame(self._activities_df[~activities_in_test]).copy()
        train_ds = SmartHomeDataset(
            sensors_df=train_sensors,
            activities_df=train_activities,
            metadata=self._metadata,
        )
        test_ds = SmartHomeDataset(
            sensors_df=test_sensors,
            activities_df=test_activities,
            metadata=self._metadata,
        )
        return [(train_ds, test_ds)]

    def extract_activity_labels(
        self, events: "EventStream"
    ) -> list[tuple[datetime, datetime, "FunctionalArea"]]:
        activity_labels = []
        occurrences = self.activity_occurrences
        for activity_occurrence in occurrences:
            active_areas = []
            for event in events.events:
                if activity_occurrence.interval.contains(event.timestamp):
                    active_areas.append(event.sensor.meta.area)
            if active_areas:
                area_counts = Counter(active_areas)
                area = area_counts.most_common(1)[0][0]
            else:
                area = FunctionalArea.UNDEFINED
            activity_labels.append(
                (
                    activity_occurrence.interval.start,
                    activity_occurrence.interval.end,
                    area,
                )
            )
        return activity_labels

    def get_timeline_data(
        self,
        selected_sensors: list[str] | None = None,
        selected_activities: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sensors_data = self._sensors_df.copy()
        if selected_sensors:
            sensors_data = sensors_data[sensors_data["name"].isin(selected_sensors)]
        sensors_data["duration"] = sensors_data.apply(
            lambda row: (
                Interval(
                    start=row["start"],
                    end=row["end"],
                    interval_type=IntervalType.LEFT_OPEN,
                ).duration
            ),
            axis=1,
        )
        activities_data = self._activities_df.copy()
        if selected_activities:
            activities_data = activities_data[
                activities_data["name"].isin(selected_activities)
            ]
        activities_data["duration"] = activities_data.apply(
            lambda row: (
                Interval(
                    start=row["start"],
                    end=row["end"],
                    interval_type=IntervalType.LEFT_OPEN,
                ).duration
            ),
            axis=1,
        )

        return _as_frame(sensors_data[["name", "start", "end", "duration"]]), _as_frame(
            activities_data[["name", "start", "end", "duration"]]
        )

    def compute_dataset_details(self) -> pd.DataFrame:
        details: dict[str, Any] = {}
        details["nb_sensors"] = len(self.get_sensor_names())
        details["nb_sensor_intervals"] = len(self._sensors_df)
        details["nb_activities"] = len(self.get_activity_names())
        details["nb_activity_intervals"] = len(self._activities_df)
        time_span = self.get_time_span()
        if time_span[0] and time_span[1]:
            total_duration = (
                Interval(
                    start=time_span[0],
                    end=time_span[1],
                    interval_type=IntervalType.LEFT_OPEN,
                ).duration
                / 3600
            )
        else:
            total_duration = 0.0
        details["total_duration_hours"] = round(total_duration, 2)
        max_sensor_gap = 0.0
        nb_null_periods = 0
        if len(self._sensors_df) > 1:
            sensors_sorted = self._sensors_df.sort_values("end").reset_index(drop=True)
            gaps_seconds = []
            for i in range(len(sensors_sorted) - 1):
                gap = (
                    sensors_sorted.iloc[i + 1]["start"] - sensors_sorted.iloc[i]["end"]
                ).total_seconds()
                if gap > 0:
                    gaps_seconds.append(gap)
                    nb_null_periods += 1
            if gaps_seconds:
                max_sensor_gap = max(gaps_seconds) / 3600
        details["max_time_gap_sensors_hours"] = round(max_sensor_gap, 2)
        max_activity_gap = 0.0
        if len(self._activities_df) > 1:
            activities_sorted = self._activities_df.sort_values("end").reset_index(
                drop=True
            )
            gaps_seconds = []
            for i in range(len(activities_sorted) - 1):
                gap = (
                    activities_sorted.iloc[i + 1]["start"]
                    - activities_sorted.iloc[i]["end"]
                ).total_seconds()
                if gap > 0:
                    gaps_seconds.append(gap)
            if gaps_seconds:
                max_activity_gap = max(gaps_seconds) / 3600
        details["max_time_gap_activities_hours"] = round(max_activity_gap, 2)
        details["nb_null_periods"] = nb_null_periods
        df = pd.DataFrame(
            [
                {"Metric": "Number of Sensors", "Value": details["nb_sensors"]},
                {
                    "Metric": "Number of Sensor Intervals",
                    "Value": details["nb_sensor_intervals"],
                },
                {"Metric": "Number of Activities", "Value": details["nb_activities"]},
                {
                    "Metric": "Number of Activity Intervals",
                    "Value": details["nb_activity_intervals"],
                },
                {
                    "Metric": "Total Duration (hours)",
                    "Value": details["total_duration_hours"],
                },
                {
                    "Metric": "Total Duration (days)",
                    "Value": round(details["total_duration_hours"] / 24, 2),
                },
                {
                    "Metric": "Max Time Gap Between Sensors (hours)",
                    "Value": details["max_time_gap_sensors_hours"],
                },
                {
                    "Metric": "Max Time Gap Between Activities (hours)",
                    "Value": details["max_time_gap_activities_hours"],
                },
                {
                    "Metric": "Number of Null Periods",
                    "Value": details["nb_null_periods"],
                },
            ]
        )

        return df

    def __repr__(self) -> str:
        """String representation of the dataset."""
        stats = self.get_statistics()
        path_name = self.dataset_path.name if self.dataset_path else "in-memory"
        return (
            f"SmartHomeDataset(path='{path_name}', "
            f"sensors={stats.num_sensors}, "
            f"activities={stats.num_activities})"
        )
