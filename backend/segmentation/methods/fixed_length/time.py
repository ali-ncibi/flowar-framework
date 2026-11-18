import datetime
from typing import Any

from ....models import EventStream, Interval, IntervalType, Segment, SegmentationResult
from ...methods.base import BaseSegmentationMethod


class FixedTime(BaseSegmentationMethod):
    def __init__(
        self,
        length_in_seconds: float = 60.0,
        overlap_percentage: float = 0.0,
    ):
        super().__init__(method_name="FixedTime")
        if length_in_seconds <= 0:
            raise ValueError(f"length_in_seconds must be > 0, got {length_in_seconds}")
        if not 0 <= overlap_percentage <= 100:
            raise ValueError(
                f"overlap_percentage must be in [0, 100], got {overlap_percentage}"
            )
        self.length_in_seconds = length_in_seconds
        self.overlap_percentage = overlap_percentage

    def fit(self, events: EventStream, **kwargs: Any) -> "FixedTime":
        self._is_calibrated = True
        return self

    def transform(self, events: EventStream) -> SegmentationResult:
        if not events.events:
            return SegmentationResult(
                segments=[],
                method_name=self.method_name,
                parameters={
                    "length_in_seconds": self.length_in_seconds,
                    "overlap_percentage": self.overlap_percentage,
                },
            )
        segments = []
        overlap_seconds = self.length_in_seconds * self.overlap_percentage / 100
        step_seconds = self.length_in_seconds - overlap_seconds
        if step_seconds <= 0:
            step_seconds = self.length_in_seconds
        stream_start = events.events[0].timestamp
        stream_end = events.events[-1].timestamp
        segment_start = stream_start
        while segment_start < stream_end:
            segment_end = segment_start + datetime.timedelta(
                seconds=self.length_in_seconds
            )
            j_first = -999
            j_last = -999
            for j, event in enumerate(events.events):
                if event.timestamp >= segment_start and event.timestamp < segment_end:
                    if j_first == -999:
                        j_first = j
                    j_last = j
            if j_first == -999 or j_last == -999:
                segment_events = []
            else:
                segment_events = events.events[j_first : j_last + 1]
            interval = Interval(
                start=segment_start,
                end=segment_end,
                interval_type=IntervalType.LEFT_OPEN,
            )
            segment = Segment(
                interval=interval,
                event_indices=(j_first, j_last),
                metadata={
                    "duration_requested_seconds": self.length_in_seconds,
                    "duration_actual_seconds": interval.duration,
                    "num_events": len(segment_events),
                },
            )
            segments.append(segment)
            segment_start += datetime.timedelta(seconds=step_seconds)
        return SegmentationResult(
            segments=segments,
            method_name=self.method_name,
            parameters={
                "length_in_seconds": self.length_in_seconds,
                "overlap_percentage": self.overlap_percentage,
            },
        )
