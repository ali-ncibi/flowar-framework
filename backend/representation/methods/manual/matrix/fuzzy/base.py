from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from numpy import array, ndarray

from backend.models import Interval, Segment, Sensor, SensorActivations

from .....methods import BaseRepresentationMethod
from ...base import group_activations_by_sensor


@dataclass(frozen=True)
class FuzzyTemporalWindow:
    direction: str
    l1: timedelta
    l2: timedelta
    l3: timedelta
    l4: timedelta

    def apply_trapezoidal_fn(
        self,
        t_x: datetime,
        t_l1: datetime,
        t_l2: datetime,
        t_l3: datetime,
        t_l4: datetime,
    ):
        if t_x <= t_l1:
            return 0.0
        elif t_x < t_l2:
            return (t_x - t_l1).total_seconds() / (t_l2 - t_l1).total_seconds()
        elif t_x <= t_l3:
            return 1.0
        elif t_x < t_l4:
            return (t_l4 - t_x).total_seconds() / (t_l4 - t_l3).total_seconds()
        else:
            return 0.0

    def compute_degree(
        self,
        current_time: datetime,
        sensor_x_activations: SensorActivations,
    ) -> float:
        if self.direction == "past":
            t_l1 = current_time - self.l1
            t_l2 = current_time - self.l2
            t_l3 = current_time - self.l3
            t_l4 = current_time - self.l4
        else:
            t_l1 = current_time + self.l4
            t_l2 = current_time + self.l3
            t_l3 = current_time + self.l2
            t_l4 = current_time + self.l1
        x_ov_activations = [
            (a, a.interval.overlap_duration(Interval(t_l1, t_l4)))
            for a in sensor_x_activations.data
            if a.interval.overlaps(Interval(t_l1, t_l4))
        ]
        if (
            len(x_ov_activations) == 1
            and x_ov_activations[0][1] >= (t_l4 - t_l1).total_seconds()
        ):
            return 1.0
        t_xs = sorted(
            set(
                [
                    a.interval.start
                    for a in sensor_x_activations.data
                    if Interval(t_l1, t_l4).contains(a.interval.start)
                ]
                + [
                    a.interval.end
                    for a in sensor_x_activations.data
                    if Interval(t_l1, t_l4).contains(a.interval.end)
                ]
            )
        )
        degrees = [
            self.apply_trapezoidal_fn(t_x, t_l1, t_l2, t_l3, t_l4) for t_x in t_xs
        ]
        degree = max(degrees) if degrees else 0.0
        return degree


def _make_ftws(
    minutes: int | None,
    nb_ftws: int | None,
    ftw_params: Sequence[Sequence[int]] | None,
    direction: str,
) -> list[FuzzyTemporalWindow]:
    assert minutes is not None or nb_ftws is not None or ftw_params is not None
    params: Sequence[Sequence[int]] = []
    if minutes is not None and minutes > 0:
        fibs = fib_sequence_from_f0(max(1, _get_closest_fib_index(minutes)) + 1)

        params = [
            [fibs[i], fibs[i + 1], fibs[i + 2], fibs[i + 3]]
            for i in range(len(fibs) - 3)
        ]

    if nb_ftws is not None and nb_ftws > 0:
        fibs = fib_sequence_from_f0(nb_ftws + 2)
        params = [
            [fibs[i], fibs[i + 1], fibs[i + 2], fibs[i + 3]]
            for i in range(len(fibs) - 3)
        ]
    if ftw_params is not None and len(ftw_params) > 0:
        params = ftw_params
    out = []
    for p in params:
        out.append(
            FuzzyTemporalWindow(
                direction=direction,
                l1=timedelta(minutes=p[-1]),
                l2=timedelta(minutes=p[-2]),
                l3=timedelta(minutes=p[-3]),
                l4=timedelta(minutes=p[-4]),
            )
        )
    if direction == "past":
        out.reverse()
    return out


def fib_sequence_from_f0(n: int) -> Sequence[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fibs = [0, 0, 1]
    for _ in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def _get_closest_fib_index(value: int) -> int:
    if value <= 0:
        return 0
    elif value == 1:
        return 1
    fibs = [0, 1]
    n = 2
    while True:
        next_fib = fibs[-1] + fibs[-2]
        if next_fib > value:
            return n - 1
        fibs.append(next_fib)
        n += 1


class FuzzyMatrixExtractor(BaseRepresentationMethod):
    def __init__(
        self,
        minutes: int | None = 30,
        nb_ftws: int | None = None,
        ftw_params: Sequence[Sequence[int]] | None = None,
    ):
        self.minutes = minutes
        self.nb_ftws = nb_ftws
        self.ftw_params = ftw_params
        self.sensors: set[Sensor] | None = None

    def set_sensors(self, sensors: set[Sensor]) -> None:
        self.sensors = sensors

    def fit(self, sensor_activations: SensorActivations, **kwargs) -> None:
        if self.sensors is None:
            self.sensors = set(sa.sensor for sa in sensor_activations.data)

    def transform(
        self,
        segments: list[Segment],
        sensor_activations: SensorActivations,
    ) -> tuple[ndarray, ndarray]:
        masks = array([])  # No mask for fuzzy matrix features
        return self.compute_fuzzy_features(
            segments,
            sensor_activations,
            self.minutes,
            self.nb_ftws,
            self.ftw_params,
        ), masks

    def compute_fuzzy_features(
        self,
        segments: list[Segment],
        sensor_activations: SensorActivations,
        minutes: int | None,
        nb_ftws: int | None,
        ftw_params: Sequence[Sequence[int]] | None,
    ) -> ndarray:
        assert self.sensors is not None, (
            "Sensors must be set before computing fuzzy features."
        )
        if not segments:
            return array([])
        if not sensor_activations:  # pragma: no cover  # dataclass is always truthy
            return array([])
        sensor_to_activations = group_activations_by_sensor(sensor_activations)
        sensors = sorted({s.name for s in self.sensors})
        ftws = _make_ftws(
            minutes,
            nb_ftws,
            ftw_params,
            "past",
        )
        return array(
            [
                [
                    [
                        ftw.compute_degree(
                            segments[i].interval.end, sensor_to_activations[sensor]
                        )
                        for sensor in sensors
                    ]
                    for ftw in ftws
                ]
                for i in range(len(segments))
            ]
        )
