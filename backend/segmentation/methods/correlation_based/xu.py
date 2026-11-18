from collections import defaultdict
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


class CorrXu(BaseSegmentationMethod):
    def __init__(self, context_size: int = 10):
        super().__init__("CorrXu (Xu et al. 2022)")
        if context_size < 2:
            raise ValueError("context_size must be >= 2")
        self.context_size = context_size
        self._scm: NDArray[np.float64] | None = None
        self._sensor_to_idx: dict[SensorName, int] | None = None
        self._sct: dict[SensorName, float] | None = None
        self._mti: dict[tuple[SensorName, SensorName], float] | None = None
        self._mts: dict[FunctionalArea, float] | None = None

    def _compute_scm(
        self,
        stream: EventStream,
        sensor_to_idx: dict[SensorName, int],
    ) -> NDArray[np.float64]:
        M = len(sensor_to_idx)
        scm = np.zeros((M, M), dtype=np.float64)
        windows = len(stream) - self.context_size + 1
        if windows <= 0:
            raise ValueError("Not enough data to compute SCM")
        for w in range(windows):
            sensors_in_window = {
                stream.events[i].sensor.name for i in range(w, w + self.context_size)
            }
            for s1 in sensors_in_window:
                i = sensor_to_idx[s1]
                for s2 in sensors_in_window:
                    j = sensor_to_idx[s2]
                    scm[i, j] += 1.0
        scm /= windows
        return scm

    def _compute_sct(
        self,
        scm: NDArray[np.float64],
        sensors: list[Sensor],
        critical_neighbors: dict[SensorName, SensorName] | None,
    ) -> dict[SensorName, float]:
        if self._sensor_to_idx is None:
            raise RuntimeError("Sensor to index mapping is not initialized")
        sct = {}
        for sensor in sensors:
            idx = self._sensor_to_idx[sensor.name]
            if critical_neighbors and sensor.name in critical_neighbors:
                neighbor = critical_neighbors[sensor.name]
                sct[sensor.name] = scm[idx, self._sensor_to_idx[neighbor]]
            else:
                values = np.delete(scm[idx], idx)
                sct[sensor.name] = float(np.median(values)) if len(values) else 0.0
        return sct

    def _compute_mti(
        self, stream: EventStream
    ) -> dict[tuple[SensorName, SensorName], float]:
        gaps = defaultdict(list)
        for i in range(1, len(stream)):
            e_prev = stream.events[i - 1]
            e_curr = stream.events[i]
            # Use Interval to compute inter-event gap consistently
            gap_interval = Interval(e_prev.timestamp, e_curr.timestamp)
            dt = gap_interval.duration
            gaps[(e_prev.sensor.name, e_curr.sensor.name)].append(dt)
        mti = {}
        for pair, values in gaps.items():
            arr = np.asarray(values)
            mu = arr.mean()
            sigma = arr.std(ddof=1) if len(arr) > 1 else 0.0
            mti[pair] = float(mu + 2 * sigma)
        return mti

    def _compute_mts(
        self,
        activity_labels: list[tuple[Timestamp, Timestamp, FunctionalArea]] | None,
    ) -> dict[FunctionalArea, float]:
        durations = defaultdict(list)
        if activity_labels:
            for start, end, area in activity_labels:
                # Use Interval to compute activity duration consistently
                activity_interval = Interval(start, end)
                duration = activity_interval.duration
                durations[area].append(duration)
        mts = {}
        for area in FunctionalArea:
            if durations[area]:
                arr = np.asarray(durations[area])
                mu = arr.mean()
                sigma = arr.std(ddof=1) if len(arr) > 1 else 0.0
                mts[area] = float(mu + 2 * sigma)
            else:
                mts[area] = 3600.0
        return mts

    def fit(
        self,
        events: EventStream,
        critical_neighbors: dict[SensorName, SensorName] | None = None,
        activity_labels: list[tuple[Timestamp, Timestamp, FunctionalArea]]
        | None = None,
        **kwargs: Any,
    ) -> "CorrXu":
        sensors = list(events.sensors)
        self._sensor_to_idx = {s.name: i for i, s in enumerate(sensors)}
        self._scm = self._compute_scm(events, self._sensor_to_idx)
        self._sct = self._compute_sct(self._scm, sensors, critical_neighbors)
        self._mti = self._compute_mti(events)
        self._mts = self._compute_mts(activity_labels)
        self._is_calibrated = True
        return self

    def transform(self, events: EventStream) -> SegmentationResult:
        if not self._is_calibrated:
            raise RuntimeError("CorrXu must be calibrated before segmentation")
        if (
            self._scm is None
            or self._sensor_to_idx is None
            or self._sct is None
            or self._mti is None
            or self._mts is None
        ):
            raise RuntimeError("CorrXu internal state is not properly initialized")
        segments: list[Segment] = []
        for i in range(1, len(events)):
            Ei = events.events[i]
            Si = Ei.sensor.name
            area = Ei.sensor.meta.area
            start = i - 1
            while start >= 0:
                Ej = events.events[start]
                Sj = Ej.sensor.name
                if (
                    self._scm[self._sensor_to_idx[Sj], self._sensor_to_idx[Si]]
                    < self._sct[Si]
                ):
                    break
                if start + 1 <= i - 1:
                    Ek = events.events[start + 1]
                    pair = (Ej.sensor.name, Ek.sensor.name)
                    # Use Interval for inter-event gap comparison
                    gap_interval = Interval(Ej.timestamp, Ek.timestamp)
                    gap_duration = gap_interval.duration
                    if pair in self._mti and gap_duration > self._mti[pair]:
                        break
                # Use Interval for activity duration comparison
                activity_interval = Interval(Ej.timestamp, Ei.timestamp)
                activity_duration = activity_interval.duration
                if activity_duration > self._mts[area]:
                    break
                start -= 1
            start += 1
            seg = Segment(
                interval=Interval(
                    events.events[start].timestamp,
                    Ei.timestamp,
                    IntervalType.LEFT_OPEN,
                ),
                event_indices=(start, i),
                metadata={"window_size": i - start + 1},
            )
            segments.append(seg)
        return SegmentationResult(
            segments=segments,
            method_name=self.method_name,
            parameters={"context_size": self.context_size},
            metadata={"num_events": len(events)},
        )
