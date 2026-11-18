from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from math import cos, pi, sin
from typing import Any

from numpy import array, concatenate, ndarray, zeros

from backend.models import (
    Interval,
    IntervalType,
    Segment,
    SensorActivations,
    SensorName,
)

from .....methods import BaseRepresentationMethod
from ...base import group_activations_by_sensor


class StatsVectorExtractor(BaseRepresentationMethod):
    def __init__(self, desired_features: Sequence[str]) -> None:
        self.desired_features = desired_features

    def fit(self, sensor_activations: SensorActivations, **kwargs):
        self.training_sensors = {sa.sensor for sa in sensor_activations.data}

    def transform(
        self,
        segments: list[Segment],
        sensor_activations: SensorActivations,
    ) -> tuple[ndarray, ndarray]:
        masks = array([])  # No mask for vector features
        return self._build_segment_features(
            segments,
            sensor_activations,
            self.desired_features,
        ), masks

    # @staticmethod
    # def _build_textual_features(
    #     sensor_activations: SensorActivations,
    #     segments: list[Segment],
    #     textualizer: WindowTextualizer,
    #     vectorizer: TfidfVectorizer
    #     | Word2VecVectorizer
    #     | SentenceTransformerVectorizer,
    # ) -> tuple[ndarray, list[int]]:
    #     events = sensor_activations.to_event_stream()
    #     descriptions_by_segment: list[list[str]] = []
    #     for segment in segments:
    #         segment_events = segment.to_event_window(events)
    #         if segment_events:
    #             descriptions = textualizer.describe_window(segment_events)
    #         else:
    #             descriptions = ["No events in segment."]
    #         descriptions_by_segment.append(descriptions)
    #     descriptions_flat: list[str] = []
    #     lengths: list[int] = []
    #     for descriptions in descriptions_by_segment:
    #         lengths.append(len(descriptions))
    #         descriptions_flat.extend(descriptions)
    #     vectors = vectorizer.fit_transform(descriptions_flat)
    #     return vectors, lengths

    def _build_segment_features(
        self,
        segments: list[Segment],
        sensor_activations: SensorActivations,
        desired_features: Sequence[str],
    ) -> ndarray:
        n_segments = len(segments)
        if n_segments == 0 or len(desired_features) == 0:
            return array([]).reshape((n_segments, 0))
        desired = set(desired_features)
        activations_by_name = group_activations_by_sensor(sensor_activations)
        sensor_names: list[SensorName] = (
            sorted({s.name for s in self.training_sensors})
            if self.training_sensors
            else sorted(activations_by_name.keys())
        )
        n_sensors = len(sensor_names)
        need_sensor_features = bool(
            desired
            & {
                "sensor_state_changed",
                "sensor_last_fired",
                "sensor_is_active",
                "sensor_shifted_activation",
                "sensor_event_count",
                "sensor_dwell_time",
            }
        )
        state_changed: ndarray | None = None
        is_active: ndarray | None = None
        last_fired: ndarray | None = None
        shifted_activation: ndarray | None = None
        event_count: ndarray | None = None
        dwell_time: ndarray | None = None
        if need_sensor_features and n_sensors > 0:
            starts_sorted: list[list[Any]] = []
            ends_sorted: list[list[Any]] = []
            by_end_start: list[list[tuple[Any, Any]]] = []
            intervals_by_sensor: list[list[Interval]] = []
            state_change_events: list[tuple[Any, int]] = []
            for si, name in enumerate(sensor_names):
                lst = activations_by_name.get(name, SensorActivations(data=[]))
                s = [r.interval.start for r in lst.data]
                e = [r.interval.end for r in lst.data]
                starts_sorted.append(s)
                ends_sorted.append(sorted(e))
                by_end_start.append(
                    sorted(
                        ((r.interval.end, r.interval.start) for r in lst.data),
                        key=lambda t: t[0],
                    )
                )
                intervals_by_sensor.append([r.interval for r in lst.data])
                for r in lst.data:
                    state_change_events.append((r.interval.start, si))
                    state_change_events.append((r.interval.end, si))
            state_change_events.sort(key=lambda t: t[0])
            state_changed = zeros((n_segments, n_sensors), dtype=int)
            is_active = zeros((n_segments, n_sensors), dtype=int)
            last_fired = zeros((n_segments, n_sensors), dtype=int)
            event_count = zeros((n_segments, n_sensors), dtype=int)
            dwell_time = zeros((n_segments, n_sensors), dtype=float)
            ev_i = 0
            last_si: int | None = None
            for seg_i, seg in enumerate(segments):
                seg_dur_s = seg.interval.duration
                inv_seg_dur = 0.0 if seg_dur_s <= 0 else (1.0 / seg_dur_s)
                while (
                    ev_i < len(state_change_events)
                    and state_change_events[ev_i][0] <= seg.interval.end
                ):
                    last_si = state_change_events[ev_i][1]
                    ev_i += 1
                if last_si is not None:
                    last_fired[seg_i, last_si] = 1
                for si in range(n_sensors):
                    s_list = starts_sorted[si]
                    e_list = ends_sorted[si]
                    has_start_in = False
                    if s_list:
                        j = bisect_left(s_list, seg.interval.start)
                        has_start_in = j < len(s_list) and s_list[j] <= seg.interval.end
                    has_end_in = False
                    if e_list:
                        k = bisect_left(e_list, seg.interval.start)
                        has_end_in = k < len(e_list) and e_list[k] <= seg.interval.end
                    state_changed[seg_i, si] = 1 if (has_start_in or has_end_in) else 0
                    bes = by_end_start[si]
                    if bes:
                        idx = bisect_right(
                            bes, (seg.interval.start, seg.interval.start)
                        )
                        if idx < len(bes):
                            end_t, start_t = bes[idx]
                            sensor_activation_interval = Interval(
                                start_t, end_t, IntervalType.LEFT_OPEN
                            )
                            if sensor_activation_interval.overlaps(seg.interval):
                                is_active[seg_i, si] = 1
                    if (
                        event_count is not None
                    ):  # pragma: no cover  # always set when sensor features requested
                        if s_list:
                            left = bisect_left(s_list, seg.interval.start)
                            right = bisect_right(s_list, seg.interval.end)
                            cnt_starts = right - left
                        else:
                            cnt_starts = 0
                        if e_list:
                            left = bisect_left(e_list, seg.interval.start)
                            right = bisect_right(e_list, seg.interval.end)
                            cnt_ends = right - left
                        else:
                            cnt_ends = 0
                        event_count[seg_i, si] = cnt_starts + cnt_ends
                    if dwell_time is not None and seg_dur_s > 0:
                        intervals = intervals_by_sensor[si]
                        if intervals:
                            first_ge = (
                                bisect_left(s_list, seg.interval.start) if s_list else 0
                            )
                            scan_from = max(0, first_ge - 1)
                            overlap_s = 0.0
                            for sensor_interval in intervals[scan_from:]:
                                overlap_s += sensor_interval.overlap_duration(
                                    seg.interval
                                )
                            dwell_time[seg_i, si] = overlap_s * inv_seg_dur
                    elif (
                        dwell_time is not None
                    ):  # pragma: no cover  # always set when sensor features requested
                        dwell_time[seg_i, si] = 0.0
            if "sensor_shifted_activation" in desired:
                shifted_activation = zeros((n_segments, n_sensors), dtype=int)
                if n_segments > 1:
                    shifted_activation[1:, :] = is_active[:-1, :]
        trig_time: ndarray | None = None
        duration_feat: ndarray | None = None
        if "segment_trigonometric_time_encoding" in desired:
            T = 24 * 60 * 60
            trig_time = zeros((n_segments, 4), dtype=float)
            for i, seg in enumerate(segments):
                end = seg.interval.end
                end_t = end.hour * 3600 + end.minute * 60 + end.second
                end_angle = 2 * pi * end_t / T
                trig_time[i, 0] = cos(end_angle)
                trig_time[i, 1] = sin(end_angle)
                start = seg.interval.start
                start_t = start.hour * 3600 + start.minute * 60 + start.second
                start_angle = 2 * pi * start_t / T
                trig_time[i, 2] = cos(start_angle)
                trig_time[i, 3] = sin(start_angle)
        if "segment_duration" in desired:
            duration_feat = zeros((n_segments,), dtype=float)
            for i, seg in enumerate(segments):
                d = seg.interval.duration
                duration_feat[i] = cos(2 * pi * d / 86400)
        parts: list[ndarray] = []
        for name in desired_features:
            if name == "sensor_state_changed":
                parts.append(
                    state_changed
                    if state_changed is not None
                    else zeros((n_segments, n_sensors), dtype=int)
                )
            elif name == "sensor_last_fired":
                parts.append(
                    last_fired
                    if last_fired is not None
                    else zeros((n_segments, n_sensors), dtype=int)
                )
            elif name == "sensor_is_active":
                parts.append(
                    is_active
                    if is_active is not None
                    else zeros((n_segments, n_sensors), dtype=int)
                )
            elif name == "sensor_shifted_activation":
                parts.append(
                    shifted_activation
                    if shifted_activation is not None
                    else zeros((n_segments, n_sensors), dtype=int)
                )
            elif name == "sensor_event_count":
                parts.append(
                    event_count
                    if event_count is not None
                    else zeros((n_segments, n_sensors), dtype=int)
                )
            elif name == "sensor_dwell_time":
                parts.append(
                    dwell_time
                    if dwell_time is not None
                    else zeros((n_segments, n_sensors), dtype=float)
                )
            elif name == "segment_trigonometric_time_encoding":
                parts.append(
                    trig_time
                    if trig_time is not None
                    else zeros((n_segments, 4), dtype=float)
                )
            elif name == "segment_duration":
                col = (
                    duration_feat
                    if duration_feat is not None
                    else zeros((n_segments,), dtype=float)
                )
                parts.append(col.reshape(n_segments, 1))

            else:
                raise ValueError(
                    f"Unknown feature '{name}'. Supported: "
                    "sensor_state_changed, sensor_last_fired, sensor_is_active, sensor_shifted_activation, "
                    "sensor_event_count, sensor_dwell_time, "
                    "segment_trigonometric_time_encoding, segment_duration, "
                )
        if len(parts) == 1:
            return parts[0]
        output = concatenate(parts, axis=1)
        return output
