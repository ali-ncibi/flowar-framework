from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytest

from backend.dataset import SmartHomeDataset
from backend.models import FunctionalArea, SensorObject, SensorType

T0 = datetime(2024, 1, 1, 8, 0, 0)


class TestInitFromPath:
    def test_loads_dataset(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.name == "toy-home"
        assert ds.dataset_path == dataset_dir
        assert len(ds.get_sensor_names()) == 2
        assert sorted(ds.get_activity_names()) == ["COOKING", "SHOWERING"]

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="sensors.csv"):
            SmartHomeDataset(tmp_path)

    def test_num_days_filter(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir, num_days=1)
        assert len(ds._sensors_df) == 3  # only day-1 rows
        assert len(ds._activities_df) == 2

    def test_get_total_days(self, dataset_dir):
        assert SmartHomeDataset.get_total_days(dataset_dir) == 2

    def test_metadata_loaded(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.metadata["sensors"]["sorted_names"] == ["SENSOR_A", "SENSOR_B"]


class TestInitInMemory:
    def test_empty(self):
        ds = SmartHomeDataset()
        assert ds.name == "in-memory"
        assert ds.dataset_path is None
        assert ds.get_time_span() == (None, None)
        assert ds.get_daily_summary() == {}
        assert ds.leave_one_day_out_split() == []

    def test_with_dataframes(self, sensors_df, activities_df, metadata):
        ds = SmartHomeDataset(
            sensors_df=sensors_df, activities_df=activities_df, metadata=metadata
        )
        assert ds.name == "in-memory"
        assert len(ds._sensors_df) == 5
        assert ds.metadata == metadata

    def test_dataframe_dates_converted(self):
        df = pd.DataFrame(
            {
                "start": ["2024-01-01 08:00:00"],
                "end": ["2024-01-01 08:01:00"],
                "name": ["S"],
            }
        )
        ds = SmartHomeDataset(sensors_df=df)
        assert isinstance(ds._sensors_df["start"].iloc[0], pd.Timestamp)


class TestSensorAccessors:
    def test_sensor_activations(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        acts = ds.sensor_activations
        assert len(acts.data) == 5
        first = acts.data[0]
        assert first.sensor.name == "SENSOR_A"
        assert first.sensor.meta.sensor_type == SensorType.MOTION
        assert first.sensor.meta.area == FunctionalArea.KITCHEN
        assert first.sensor.meta.sensor_object == SensorObject.COOKTOP
        assert first.duration == 120.0

    def test_undefined_metadata(self):
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:01:00"]),
                "name": ["X"],
            }
        )
        ds = SmartHomeDataset(sensors_df=df, metadata={})
        act = ds.sensor_activations.data[0]
        assert act.sensor.meta.sensor_type == SensorType.UNDEFINED
        assert act.sensor.meta.area == FunctionalArea.UNDEFINED
        assert act.sensor.meta.sensor_object == SensorObject.UNDEFINED

    def test_get_sensor_activation_count(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_sensor_activation_count("SENSOR_A") == 3
        assert ds.get_sensor_activation_count("NOPE") == 0

    def test_get_sensor_activations_by_sensor(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        acts = ds.get_sensor_activations_by_sensor("SENSOR_B")
        assert len(acts.data) == 2
        assert all(a.sensor.name == "SENSOR_B" for a in acts.data)

    def test_get_sensor_statistics(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_sensor_statistics() == {"SENSOR_A": 3, "SENSOR_B": 2}

    def test_get_sensor_duration_statistics(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        stats = ds.get_sensor_duration_statistics()
        assert stats["SENSOR_A"] == 120.0 + 1800.0 + 300.0
        assert stats["SENSOR_B"] == 240.0

    def test_get_sensors_by_activity(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_sensors_by_activity("COOKING") == {"SENSOR_A", "SENSOR_B"}
        assert ds.get_sensors_by_activity("UNKNOWN") == set()


class TestActivityAccessors:
    def test_activity_occurrences(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        occs = ds.activity_occurrences
        assert len(occs) == 3
        assert occs[0].name == "COOKING"
        assert occs[0].duration == 600.0

    def test_get_activities_by_sensor(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_activities_by_sensor("SENSOR_A") == {"COOKING", "SHOWERING"}
        assert ds.get_activities_by_sensor("NOPE") == set()

    def test_get_activity_annotation_count(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_activity_annotation_count("COOKING") == 2
        assert ds.get_activity_annotation_count("NOPE") == 0

    def test_get_activity_annotations_by_activity(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        occs = ds.get_activity_annotations_by_activity("COOKING")
        assert len(occs) == 2
        assert all(o.name == "COOKING" for o in occs)

    def test_get_activity_statistics(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_activity_statistics() == {"COOKING": 2, "SHOWERING": 1}

    def test_get_activity_duration_statistics(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        stats = ds.get_activity_duration_statistics()
        assert stats["COOKING"] == 600.0 + 900.0
        assert stats["SHOWERING"] == 1800.0


class TestMatricesAndPlots:
    def test_overlap_duration_helper(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        from backend.models import Interval, IntervalType

        i1 = Interval(T0, T0 + timedelta(minutes=10), IntervalType.LEFT_OPEN)
        i2 = Interval(T0 + timedelta(minutes=5), T0 + timedelta(minutes=15))
        assert ds._compute_overlap_duration(i1, i2) == 300.0

    def test_get_sensor_activity_matrix(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        m = ds.get_sensor_activity_matrix()
        assert m["SENSOR_A"]["COOKING"] > 0
        assert "SHOWERING" not in m["SENSOR_B"]

    def test_get_sensor_activity_overlap_plot(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        fig = ds.get_sensor_activity_overlap_plot()
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_get_sensor_duration_kde_plot(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        fig_act, fig_deact = ds.get_sensor_duration_kde_plot("SENSOR_A")
        assert isinstance(fig_act, go.Figure)
        assert isinstance(fig_deact, go.Figure)

    def test_get_sensor_duration_kde_plot_too_few(self):
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:01:00"]),
                "name": ["ONLY"],
            }
        )
        ds = SmartHomeDataset(sensors_df=df, metadata={})
        with pytest.raises(ValueError, match="fewer than 2 activations"):
            ds.get_sensor_duration_kde_plot("ONLY")


class TestTimeSpanAndRanges:
    def test_get_time_span(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        start, end = ds.get_time_span()
        assert start == T0
        assert end == datetime(2024, 1, 2, 8, 15, 0)

    def test_get_activations_in_time_range(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        acts = ds.get_activations_in_time_range(
            datetime(2024, 1, 1, 8, 30), datetime(2024, 1, 1, 23, 59)
        )
        assert len(acts.data) == 1
        assert acts.data[0].sensor.name == "SENSOR_A"

    def test_get_activities_in_time_range(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        acts = ds.get_activities_in_time_range(
            datetime(2024, 1, 1, 8, 30), datetime(2024, 1, 1, 23, 59)
        )
        assert len(acts) == 1
        assert acts[0].name == "SHOWERING"

    def test_get_sensor_event_gaps(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        gaps = ds.get_sensor_event_gaps(max_gap_seconds=60.0)
        assert list(gaps.columns) == [
            "prev_name",
            "prev_start",
            "prev_end",
            "next_name",
            "next_start",
            "next_end",
            "gap_seconds",
        ]
        assert len(gaps) >= 1
        assert (gaps["gap_seconds"] > 60.0).all()

    def test_get_sensor_event_gaps_negative(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        with pytest.raises(ValueError, match="max_gap_seconds"):
            ds.get_sensor_event_gaps(max_gap_seconds=-1.0)

    def test_get_sensor_event_gaps_single_row(self):
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:01:00"]),
                "name": ["ONLY"],
            }
        )
        ds = SmartHomeDataset(sensors_df=df, metadata={})
        gaps = ds.get_sensor_event_gaps(max_gap_seconds=60.0)
        assert gaps.empty


class TestStatisticsAndRankings:
    def test_get_statistics(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        stats = ds.get_statistics()
        assert stats.num_sensors == 2
        assert stats.num_activities == 2
        assert stats.num_sensor_activations == 5
        assert stats.num_activity_occurrences == 3
        assert stats.time_span == (T0, datetime(2024, 1, 2, 8, 15, 0))
        assert stats.total_duration_seconds == (86400.0 + 900.0)

    def test_print_statistics(self, dataset_dir, capsys):
        ds = SmartHomeDataset(dataset_dir)
        ds.print_statistics()
        out = capsys.readouterr().out
        assert "SMART HOME DATASET STATISTICS" in out
        assert "toy-home" in out

    def test_print_statistics_in_memory(self, sensors_df, activities_df, capsys):
        SmartHomeDataset(
            sensors_df=sensors_df, activities_df=activities_df
        ).print_statistics()
        assert "in-memory" in capsys.readouterr().out

    def test_top_sensors_by_count(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_top_sensors_by_count(1) == [("SENSOR_A", 3)]
        assert len(ds.get_top_sensors_by_count(5)) == 2

    def test_top_sensors_by_duration(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        top = ds.get_top_sensors_by_duration(1)
        assert top[0][0] == "SENSOR_A"
        assert top[0][1] == 2220.0

    def test_top_activities(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_top_activities_by_count(1) == [("COOKING", 2)]
        top_dur = ds.get_top_activities_by_duration(1)
        assert top_dur[0][0] == "SHOWERING"

    def test_sensor_average_durations(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        avg = ds.get_sensor_average_durations()
        assert avg["SENSOR_A"] == pytest.approx(2220.0 / 3)
        assert ds.get_top_sensors_by_average_duration(1)[0][0] == "SENSOR_A"

    def test_sensor_activity_diversity(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        div = ds.get_sensor_activity_diversity()
        assert div == {"SENSOR_A": 2, "SENSOR_B": 1}
        limited = ds.get_sensor_activity_diversity(limit=1)
        assert list(limited) == ["SENSOR_A"]
        top = ds.get_top_sensors_by_activity_diversity(k=1)
        assert top[0] == ("SENSOR_A", 2)

    def test_activity_transitions(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        matrix = ds.get_activity_transition_matrix(max_gap_seconds=60.0)
        assert matrix == {}
        top = ds.get_top_activity_transitions(k=10, max_gap_seconds=60.0)
        assert top == []
        # with a huge gap tolerance, transitions also count
        matrix_all = ds.get_activity_transition_matrix(max_gap_seconds=10**9)
        assert matrix_all == {"COOKING -> SHOWERING": 1, "SHOWERING -> COOKING": 1}

    def test_longest(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        longest_acts = ds.get_longest_activities(k=2)
        assert len(longest_acts) == 2
        assert longest_acts[0].duration >= longest_acts[1].duration
        longest_sensors = ds.get_longest_sensor_activations(k=2)
        assert len(longest_sensors.data) == 2

    def test_daily_summary(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        summary = ds.get_daily_summary(num_days=2)
        assert len(summary) == 2
        day1 = summary[datetime(2024, 1, 1).date()]
        assert day1["total_activities"] == 3
        assert day1["activity_breakdown"] == {"COOKING": 2, "SHOWERING": 1}
        expected_acts = len(
            ds.get_activations_in_time_range(
                datetime(2024, 1, 1, 8, 0), datetime(2024, 1, 2, 8, 0)
            ).data
        )
        assert day1["total_activations"] == expected_acts

    def test_daily_summary_empty(self):
        assert SmartHomeDataset().get_daily_summary() == {}

    def test_sensor_activity_correlation(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        corr = ds.get_sensor_activity_correlation(num_activities=2)
        assert set(corr) == {"COOKING", "SHOWERING"}
        assert corr["COOKING"] == {"SENSOR_A", "SENSOR_B"}


class TestMetadataAccessors:
    def test_get_sensor_metadata(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        meta = ds.get_sensor_metadata("SENSOR_A")
        assert meta == {"type": "motion", "area": "kitchen", "object": "cooktop"}
        assert ds.get_sensor_metadata("NOPE") is None

    def test_get_sensors_by_area_and_type(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_sensors_by_area("kitchen") == ["SENSOR_A"]
        assert ds.get_sensors_by_area("garage") == []
        assert ds.get_sensors_by_type("contact") == ["SENSOR_B"]
        assert ds.get_sensors_by_type("sonar") == []

    def test_get_areas_and_types(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        assert ds.get_areas() == ["bathroom", "kitchen"]
        assert ds.get_sensor_types() == ["contact", "motion"]

    def test_get_all_sensors_metadata(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        all_meta = ds.get_all_sensors_metadata()
        assert set(all_meta) == {"SENSOR_A", "SENSOR_B"}


class TestEventStream:
    def test_to_event_stream(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        stream = ds.to_event_stream()
        assert len(stream) == 10
        assert {s.name for s in stream.sensors} == {"SENSOR_A", "SENSOR_B"}
        assert [e.index for e in stream.events] == list(range(10))

    def test_to_event_stream_max_events(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        stream = ds.to_event_stream(max_events=2)
        assert len(stream) == 2

    def test_extract_activity_labels(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        events = ds.to_event_stream()
        labels = ds.extract_activity_labels(events)
        assert len(labels) == 3
        areas = [area for _, _, area in labels]
        assert areas[0] == FunctionalArea.KITCHEN
        assert areas[1] == FunctionalArea.KITCHEN  # SHOWERING overlaps SENSOR_A only
        assert areas[2] == FunctionalArea.KITCHEN

    def test_extract_activity_labels_no_events(self, dataset_dir):
        from backend.models import EventStream

        ds = SmartHomeDataset(dataset_dir)
        labels = ds.extract_activity_labels(EventStream([]))
        assert len(labels) == 3
        assert all(area == FunctionalArea.UNDEFINED for _, _, area in labels)


class TestSplits:
    def test_train_test_split(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        splits = ds.train_test_split(test_size=0.5)
        assert len(splits) == 1
        train, test = splits[0]
        assert len(test._sensors_df) == 2  # day-2 rows
        assert len(train._sensors_df) == 3

    def test_train_test_split_invalid(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        with pytest.raises(ValueError, match="test_size"):
            ds.train_test_split(test_size=1.5)

    def test_train_test_split_empty(self):
        ds = SmartHomeDataset()
        splits = ds.train_test_split(test_size=0.2)
        assert len(splits) == 1
        train, test = splits[0]
        assert len(train._sensors_df) == 0 and len(test._sensors_df) == 0

    def test_leave_one_day_out(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        splits = ds.leave_one_day_out_split()
        assert len(splits) == 2
        for _train, test in splits:
            assert not test._sensors_df.empty
            assert not test._activities_df.empty

    def test_leave_one_day_out_empty(self):
        assert SmartHomeDataset().leave_one_day_out_split() == []


class TestTimelineAndDetails:
    def test_get_timeline_data(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        sensors_data, activities_data = ds.get_timeline_data()
        assert list(sensors_data.columns) == ["name", "start", "end", "duration"]
        assert len(sensors_data) == 5
        sensors_sel, activities_sel = ds.get_timeline_data(
            selected_sensors=["SENSOR_A"], selected_activities=["COOKING"]
        )
        assert set(sensors_sel["name"]) == {"SENSOR_A"}
        assert set(activities_sel["name"]) == {"COOKING"}

    def test_compute_dataset_details(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        df = ds.compute_dataset_details()
        metrics = dict(zip(df["Metric"], df["Value"], strict=False))
        assert metrics["Number of Sensors"] == 2
        assert metrics["Number of Sensor Intervals"] == 5
        assert metrics["Number of Activities"] == 2
        assert metrics["Total Duration (days)"] == pytest.approx(
            87120.0 / 86400, abs=0.01
        )
        assert metrics["Number of Null Periods"] >= 1

    def test_repr(self, dataset_dir, sensors_df, activities_df):
        ds = SmartHomeDataset(dataset_dir)
        r = repr(ds)
        assert "toy-home" in r and "sensors=2" in r
        assert "in-memory" in repr(
            SmartHomeDataset(sensors_df=sensors_df, activities_df=activities_df)
        )


def _write_csv_dataset(d, sensors_rows, activities_rows):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensors_rows, columns=["start", "end", "name"]).to_csv(
        d / "sensors.csv", index=False
    )
    pd.DataFrame(activities_rows, columns=["start", "end", "name"]).to_csv(
        d / "activities.csv", index=False
    )
    (d / "meta.toml").write_text("[sensors]\nsorted_names = []\n")


class TestEdgeCases:
    def test_get_total_days_empty(self, tmp_path):
        d = tmp_path / "empty-home"
        _write_csv_dataset(d, [], [])
        assert SmartHomeDataset.get_total_days(d) == 0

    def test_get_total_days_activities_only(self, tmp_path):
        d = tmp_path / "act-only"
        _write_csv_dataset(
            d,
            [],
            [("2024-01-01 08:00:00", "2024-01-03 08:00:00", "A")],
        )
        assert SmartHomeDataset.get_total_days(d) == 3

    def test_validate_dataset_path_none(self):
        ds = SmartHomeDataset()
        with pytest.raises(ValueError, match="cannot be None"):
            ds._validate_dataset_path()

    def test_load_data_none_path(self):
        ds = SmartHomeDataset()
        with pytest.raises(ValueError, match="must be set"):
            ds._load_data()

    def test_filter_by_days_empty(self):
        ds = SmartHomeDataset()
        ds._filter_by_days(5)  # no-op on empty frames
        assert ds._sensors_df.empty

    def test_metadata_missing_keys(self):
        df = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:01:00"]),
                "name": ["X"],
            }
        )
        metadata = {
            "sensors": {
                "metadata": {
                    "X": {"type": "motion"},  # no area / object
                    "Y": "not-a-dict",
                    "Z": {"area": "kitchen"},  # no type / object
                }
            }
        }
        ds = SmartHomeDataset(sensors_df=df, metadata=metadata)
        assert ds._get_functional_area_for_sensor("X") == FunctionalArea.UNDEFINED
        assert ds._get_object_for_sensor("X") == SensorObject.UNDEFINED
        assert ds._get_type_for_sensor("X") == SensorType.MOTION
        assert ds._get_functional_area_for_sensor("Y") == FunctionalArea.UNDEFINED
        assert ds._get_type_for_sensor("Z") == SensorType.UNDEFINED
        assert ds._get_functional_area_for_sensor("Z") == FunctionalArea.KITCHEN
        assert ds.get_areas() == ["kitchen"]
        assert ds.get_sensor_types() == ["motion"]

    def test_time_span_activities_only(self, activities_df):
        ds = SmartHomeDataset(activities_df=activities_df)
        start, end = ds.get_time_span()
        assert start == T0
        assert end == datetime(2024, 1, 2, 8, 15, 0)

    def test_overlap_plot_zero_overlap(self):
        # touching intervals overlap but have zero overlap duration
        sensors = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:10:00"]),
                "name": ["S"],
            }
        )
        activities = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:10:00"]),
                "end": pd.to_datetime(["2024-01-01 08:20:00"]),
                "name": ["A"],
            }
        )
        ds = SmartHomeDataset(sensors_df=sensors, activities_df=activities, metadata={})
        fig = ds.get_sensor_activity_overlap_plot()
        assert isinstance(fig, go.Figure)
        matrix = ds.get_sensor_activity_matrix()
        assert matrix == {"S": {"A": 0.0}}  # touching intervals record 0 overlap

    def test_kde_plot_overlapping_activations(self):
        # consecutive activations overlap -> negative gaps are skipped,
        # deactivation plot has no trace
        sensors = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 08:05:00"]),
                "end": pd.to_datetime(["2024-01-01 08:10:00", "2024-01-01 08:16:00"]),
                "name": ["S", "S"],
            }
        )
        ds = SmartHomeDataset(sensors_df=sensors, metadata={})
        fig_act, fig_deact = ds.get_sensor_duration_kde_plot("S")
        assert len(fig_act.data) == 1
        assert len(fig_deact.data) == 0

    def test_transition_repeated_key(self):
        activities = pd.DataFrame(
            {
                "start": pd.to_datetime(
                    [
                        "2024-01-01 08:00:00",
                        "2024-01-01 08:05:10",
                        "2024-01-01 08:10:20",
                        "2024-01-01 08:15:30",
                    ]
                ),
                "end": pd.to_datetime(
                    [
                        "2024-01-01 08:05:00",
                        "2024-01-01 08:10:10",
                        "2024-01-01 08:15:20",
                        "2024-01-01 08:20:30",
                    ]
                ),
                "name": ["A", "B", "A", "B"],
            }
        )
        ds = SmartHomeDataset(activities_df=activities, metadata={})
        matrix = ds.get_activity_transition_matrix(max_gap_seconds=60.0)
        assert matrix == {"A -> B": 2, "B -> A": 1}

    def test_to_event_stream_unknown_sensor_branch(self, dataset_dir):
        ds = SmartHomeDataset(dataset_dir)
        ds.get_sensor_names = lambda: []  # type: ignore[method-assign]
        stream = ds.to_event_stream()
        assert len(stream) == 10
        assert {s.name for s in stream.sensors} == {"SENSOR_A", "SENSOR_B"}

    def test_leave_one_day_out_breaks_on_empty_test(self):
        sensors = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-02 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:01:00", "2024-01-02 08:01:00"]),
                "name": ["S", "S"],
            }
        )
        activities = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:05:00"]),
                "name": ["A"],
            }
        )
        ds = SmartHomeDataset(sensors_df=sensors, activities_df=activities, metadata={})
        splits = ds.leave_one_day_out_split()
        assert len(splits) == 1  # day-2 window has no activities -> break

    def test_compute_dataset_details_empty(self):
        ds = SmartHomeDataset()
        ds.get_sensor_names = lambda: []  # type: ignore[method-assign]
        ds.get_activity_names = lambda: []  # type: ignore[method-assign]
        df = ds.compute_dataset_details()
        metrics = dict(zip(df["Metric"], df["Value"], strict=False))
        assert metrics["Total Duration (hours)"] == 0.0
        assert metrics["Number of Null Periods"] == 0

    def test_compute_dataset_details_no_positive_gaps(self):
        sensors = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 08:05:00"]),
                "end": pd.to_datetime(["2024-01-01 08:10:00", "2024-01-01 08:15:00"]),
                "name": ["S", "S"],
            }
        )
        activities = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 08:05:00"]),
                "end": pd.to_datetime(["2024-01-01 08:10:00", "2024-01-01 08:15:00"]),
                "name": ["A", "A"],
            }
        )
        ds = SmartHomeDataset(sensors_df=sensors, activities_df=activities, metadata={})
        df = ds.compute_dataset_details()
        metrics = dict(zip(df["Metric"], df["Value"], strict=False))
        assert metrics["Max Time Gap Between Sensors (hours)"] == 0.0
        assert metrics["Max Time Gap Between Activities (hours)"] == 0.0
        assert metrics["Number of Null Periods"] == 0

    def test_compute_dataset_details_single_rows(self):
        sensors = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:10:00"]),
                "name": ["S"],
            }
        )
        activities = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01 08:00:00"]),
                "end": pd.to_datetime(["2024-01-01 08:10:00"]),
                "name": ["A"],
            }
        )
        ds = SmartHomeDataset(sensors_df=sensors, activities_df=activities, metadata={})
        df = ds.compute_dataset_details()
        metrics = dict(zip(df["Metric"], df["Value"], strict=False))
        assert metrics["Max Time Gap Between Sensors (hours)"] == 0.0
        assert metrics["Number of Sensors"] == 1


class TestMoreEdgeCases:
    def test_time_span_sensors_only(self, sensors_df):
        ds = SmartHomeDataset(sensors_df=sensors_df)
        start, end = ds.get_time_span()
        assert start == T0
        assert end == datetime(2024, 1, 2, 8, 12, 0)

    def test_filter_by_days_sensors_only(self, sensors_df):
        ds = SmartHomeDataset(
            sensors_df=sensors_df,
            activities_df=pd.DataFrame(columns=["start", "end", "name"]),
        )
        ds._filter_by_days(1)
        assert len(ds._sensors_df) == 3

    def test_filter_by_days_activities_only(self, activities_df):
        ds = SmartHomeDataset(
            sensors_df=pd.DataFrame(columns=["start", "end", "name"]),
            activities_df=activities_df,
        )
        ds._filter_by_days(1)
        assert len(ds._activities_df) == 2
