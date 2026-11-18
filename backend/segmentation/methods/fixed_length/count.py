import math
from typing import Any

from ....models import EventStream, Interval, IntervalType, Segment, SegmentationResult
from ...methods.base import BaseSegmentationMethod


class FixedCount(BaseSegmentationMethod):
    def __init__(
        self,
        nb_events: int = 10,
        overlap_percentage: float = 0.0,
    ):
        super().__init__(method_name="FixedCount")
        if nb_events < 1:
            raise ValueError(f"nb_events must be >= 1, got {nb_events}")
        if not 0 <= overlap_percentage <= 100:
            raise ValueError(
                f"overlap_percentage must be in [0, 100], got {overlap_percentage}"
            )
        self.nb_events = nb_events
        self.overlap_percentage = overlap_percentage

    def fit(self, events: EventStream, **kwargs: Any) -> "FixedCount":
        self._is_calibrated = True
        return self

    def transform(self, events: EventStream) -> SegmentationResult:
        if not events.events:
            return SegmentationResult(
                segments=[],
                method_name=self.method_name,
                parameters={
                    "nb_events": self.nb_events,
                    "overlap_percentage": self.overlap_percentage,
                },
            )
        segments = []
        overlap_events = int(self.nb_events * self.overlap_percentage / 100)
        step = self.nb_events - overlap_events
        step = max(1, step)
        n = len(events.events)
        # Last start index producing a segment that reaches the final event;
        # iterating exactly up to it reproduces the previous break-based loop.
        j_star = max(0, math.ceil((n - 1 - self.nb_events) / step) * step)
        j_first = 0
        while j_first <= j_star:
            j_last = min(j_first + self.nb_events, n - 1)
            segment_events = events.events[j_first : j_last + 1]
            interval = Interval(
                start=segment_events[0].timestamp,
                end=segment_events[-1].timestamp,
                interval_type=IntervalType.LEFT_OPEN,
            )
            segment = Segment(
                interval=interval,
                event_indices=(j_first, j_last),
                metadata={
                    "num_events_requested": self.nb_events,
                    "num_events_actual": len(segment_events),
                },
            )
            segments.append(segment)
            j_first += step
        return SegmentationResult(
            segments=segments,
            method_name=self.method_name,
            parameters={
                "nb_events": self.nb_events,
                "overlap_percentage": self.overlap_percentage,
            },
        )
