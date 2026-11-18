from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ....models import (
    EventStream,
    FunctionalArea,
    Interval,
    IntervalType,
    Segment,
    SegmentationResult,
    Sensor,
    SensorName,
    Timestamp,
)
from ...methods.base import BaseSegmentationMethod


@dataclass
class ActiveWindow:
    j_first: int
    j_last: int
    t_first: Timestamp
    t_last: Timestamp
    sensor_last: Sensor


class CorrWan(BaseSegmentationMethod):
    def __init__(
        self,
        threshold: float = 0.5,
        time_bin_size: int = 60,
        time_th: str = "max",
    ):
        super().__init__("CorrWan (Wan et al. 2015)")
        self.threshold = threshold
        self.time_bin_size = time_bin_size
        self.time_th = time_th
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.time_bin_size <= 0:
            raise ValueError(f"time_bin_size must be > 0, got {self.time_bin_size}")
        if self.time_th not in ("mean", "max"):
            raise ValueError(f"time_th must be 'mean' or 'max', got {self.time_th}")
        self._scm: NDArray[np.float64] | None = None
        self._sensor_to_idx: dict[SensorName, int] | None = None
        self._t_max_span: dict[FunctionalArea, float] = {}
        self._t_max_int: dict[FunctionalArea, float] = {}

    def _build_activation_sequences(
        self,
        events: EventStream,
        sensors: list[Sensor],
    ) -> NDArray[np.float64]:
        t_min = events.events[0].timestamp
        time_th = events.events[-1].timestamp
        time_span = (time_th - t_min).total_seconds()
        n_bins = int(np.ceil(time_span / self.time_bin_size)) + 1
        X = np.zeros((len(sensors), n_bins), dtype=np.float64)
        sensor_states: dict[SensorName, bool] = {s.name: False for s in sensors}
        sensor_on_times: dict[SensorName, float] = {}
        for event in events.events:
            sensor_idx = self._sensor_to_idx[event.sensor.name]  # type: ignore
            bin_idx = int(
                ((event.timestamp - t_min).total_seconds()) / self.time_bin_size
            )
            if event.value == 1:
                sensor_states[event.sensor.name] = True
                sensor_on_times[event.sensor.name] = (
                    event.timestamp - t_min
                ).total_seconds()
            else:
                if (
                    sensor_states[event.sensor.name]
                    and event.sensor.name in sensor_on_times
                ):
                    start_bin = int(
                        sensor_on_times[event.sensor.name] / self.time_bin_size
                    )
                    end_bin = min(bin_idx, n_bins - 1)
                    X[sensor_idx, start_bin : end_bin + 1] = 1.0
                sensor_states[event.sensor.name] = False
        for sensor_id, is_on in sensor_states.items():
            if is_on and sensor_id in sensor_on_times:
                sensor_idx = self._sensor_to_idx[sensor_id]  # type: ignore
                start_bin = int(sensor_on_times[sensor_id] / self.time_bin_size)
                X[sensor_idx, start_bin:] = 1.0
        return X

    def _compute_correlation_matrix(
        self, events: EventStream, sensors: list[Sensor]
    ) -> NDArray[np.float64]:
        X = self._build_activation_sequences(events, sensors)
        M = len(sensors)
        scm = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(i, M):
                if i == j:
                    scm[i, j] = 1.0
                else:
                    x_i = X[i]
                    x_j = X[j]
                    mean_i = np.mean(x_i)
                    mean_j = np.mean(x_j)
                    std_i = np.std(x_i, ddof=1) if len(x_i) > 1 else 1.0
                    std_j = np.std(x_j, ddof=1) if len(x_j) > 1 else 1.0
                    if std_i > 0 and std_j > 0:
                        cov = np.mean((x_i - mean_i) * (x_j - mean_j))
                        rho = cov / (std_i * std_j)
                        scm[i, j] = abs(rho)
                        scm[j, i] = scm[i, j]
                    else:
                        scm[i, j] = 0.0
                        scm[j, i] = 0.0
        return scm

    def _compute_time_thresholds(
        self,
        events: EventStream,
        activity_labels: list[tuple[Timestamp, Timestamp, FunctionalArea]]
        | None = None,
    ) -> tuple[dict[FunctionalArea, float], dict[FunctionalArea, float]]:
        t_max_span: dict[FunctionalArea, float] = {}
        t_max_int: dict[FunctionalArea, float] = {}
        if activity_labels is not None:
            area_durations: dict[FunctionalArea, list[float]] = defaultdict(list)
            area_intervals: dict[FunctionalArea, list[float]] = defaultdict(list)
            for start_t, end_t, area in activity_labels:
                # Use Interval to compute duration consistently
                activity_interval = Interval(start_t, end_t)
                duration = activity_interval.duration
                area_durations[area].append(duration)
                # Filter events in interval using Interval.contains()
                activity_events = [
                    e
                    for e in events.events
                    if activity_interval.contains(e.timestamp)
                    and e.sensor.meta.area == area
                ]
                for i in range(1, len(activity_events)):
                    # Use Interval for inter-event gap computation
                    gap_interval = Interval(
                        activity_events[i - 1].timestamp, activity_events[i].timestamp
                    )
                    interval_gap = gap_interval.duration
                    area_intervals[area].append(interval_gap)
            for area in FunctionalArea:
                if area not in area_durations or len(area_durations[area]) == 0:
                    area_durations[area] = [60.0]
                if area not in area_intervals or len(area_intervals[area]) == 0:
                    area_intervals[area] = [60.0]

            def agg_fn(values: list[float]) -> float:
                if self.time_th == "mean":
                    return float(np.mean(values))
                return float(max(values))

            for area in FunctionalArea:
                t_max_span[area] = max(1e-6, float(agg_fn(area_durations[area])))
                t_max_int[area] = max(1e-6, float(agg_fn(area_intervals[area])))
        else:
            for area in FunctionalArea:
                t_max_span[area] = 60.0
                t_max_int[area] = 60.0
        return t_max_span, t_max_int

    def fit(
        self,
        events: EventStream,
        activity_labels: list[tuple[Timestamp, Timestamp, FunctionalArea]]
        | None = None,
        **kwargs: Any,
    ) -> "CorrWan":
        sensors = list(events.sensors)
        self._sensor_to_idx = {s.name: i for i, s in enumerate(sensors)}
        self._scm = self._compute_correlation_matrix(events, sensors)
        self._t_max_span, self._t_max_int = self._compute_time_thresholds(
            events, activity_labels
        )
        self._is_calibrated = True
        return self

    def transform(self, events: EventStream) -> SegmentationResult:
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before segment()")
        closed_segments: list[Segment] = []
        active_windows: list[ActiveWindow] = []
        windows_closed_time = 0
        windows_closed_end = 0
        max_concurrent_windows = 0
        for j, event in enumerate(events.events):
            windows_to_close: list[int] = []
            matching_window_idx: int | None = None
            for w_idx, w in enumerate(active_windows):
                sensor_idx_j = (
                    self._sensor_to_idx.get(event.sensor.name)
                    if self._sensor_to_idx is not None
                    else None
                )
                sensor_idx_last = (
                    self._sensor_to_idx.get(w.sensor_last.name)
                    if self._sensor_to_idx is not None
                    else None
                )
                if sensor_idx_j is None or sensor_idx_last is None:
                    sc_wan_pass = False
                else:
                    correlation = (
                        self._scm[sensor_idx_j, sensor_idx_last]
                        if self._scm is not None
                        else 0.0
                    )
                    sc_wan_pass = correlation >= self.threshold
                area = event.sensor.meta.area
                t_max_span_area = self._t_max_span[area]
                t_max_int_area = self._t_max_int[area]
                # Use Interval to compute normalized time span and gap
                span_interval = Interval(w.t_first, event.timestamp)
                gap_interval = Interval(w.t_last, event.timestamp)
                t_span_normalized = span_interval.duration / t_max_span_area
                t_gap_normalized = gap_interval.duration / t_max_int_area
                tc_wan_pass = t_span_normalized <= 1.0 and t_gap_normalized <= 1.0
                if not tc_wan_pass:
                    windows_to_close.append(w_idx)
                elif sc_wan_pass and tc_wan_pass:
                    # only the first matching window is extended (scm is symmetric,
                    # so a second matching window cannot occur in practice)
                    if matching_window_idx is None:  # pragma: no cover
                        matching_window_idx = w_idx
            for w_idx in reversed(windows_to_close):
                w = active_windows.pop(w_idx)
                windows_closed_time += 1
                segment_interval = Interval(w.t_first, w.t_last, IntervalType.LEFT_OPEN)
                segment = Segment(
                    interval=segment_interval,
                    event_indices=(w.j_first, w.j_last),
                    metadata={"window_closed": "time_exceeded"},
                )
                closed_segments.append(segment)
            if matching_window_idx is not None and matching_window_idx < len(
                active_windows
            ):
                w = active_windows[matching_window_idx]
                w.j_last = j
                w.t_last = event.timestamp
                w.sensor_last = event.sensor
            else:
                new_window = ActiveWindow(
                    j_first=j,
                    j_last=j,
                    t_first=event.timestamp,
                    t_last=event.timestamp,
                    sensor_last=event.sensor,
                )
                active_windows.append(new_window)
            max_concurrent_windows = max(max_concurrent_windows, len(active_windows))
        for w in active_windows:
            segment_interval = Interval(w.t_first, w.t_last, IntervalType.LEFT_OPEN)
            segment = Segment(
                interval=segment_interval,
                event_indices=(w.j_first, w.j_last),
                metadata={"window_closed": "end_of_stream"},
            )
            closed_segments.append(segment)
            windows_closed_end += 1
        closed_segments.sort(key=lambda seg: seg.interval.start)
        closed_segments = [seg for seg in closed_segments if seg.interval.duration > 0]
        return SegmentationResult(
            segments=closed_segments,
            method_name=self.method_name,
            parameters={
                "threshold": self.threshold,
                "time_bin_size": self.time_bin_size,
            },
            metadata={
                "windows_closed_time": windows_closed_time,
                "windows_closed_end": windows_closed_end,
                "max_concurrent_windows": max_concurrent_windows,
            },
        )
