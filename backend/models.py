from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

type Timestamp = datetime

type SensorName = str

type FunctionalAreaId = int

type ActivityName = str

type ActivityState = bool


class IntervalType(Enum):
    CLOSED = "closed"
    OPEN = "open"
    LEFT_OPEN = "left_open"
    RIGHT_OPEN = "right_open"


@dataclass(frozen=True)
class Interval:
    start: Timestamp
    end: Timestamp
    interval_type: IntervalType = IntervalType.LEFT_OPEN

    def __post_init__(self) -> None:
        """Validate interval constraints."""
        if self.end < self.start:
            raise ValueError(
                f"Interval end ({self.end}) must be >= start ({self.start})"
            )

    @property
    def duration(self) -> float:
        delta = self.end - self.start
        return delta.total_seconds()

    def contains(self, t: Timestamp) -> bool:
        if self.interval_type == IntervalType.OPEN:
            return self.start < t < self.end
        elif self.interval_type == IntervalType.LEFT_OPEN:
            return self.start <= t <= self.end
        else:
            return self.start <= t < self.end

    def overlaps(self, other: Interval) -> bool:
        if self.end < other.start or other.end < self.start:
            return False
        if self.end == other.start:
            right_included = self.interval_type in (
                IntervalType.LEFT_OPEN,
                IntervalType.LEFT_OPEN,
            )
            left_included = other.interval_type in (
                IntervalType.LEFT_OPEN,
                IntervalType.RIGHT_OPEN,
            )
            return right_included and left_included

        if self.start == other.end:
            left_included = self.interval_type in (
                IntervalType.LEFT_OPEN,
                IntervalType.RIGHT_OPEN,
            )
            right_included = other.interval_type in (
                IntervalType.LEFT_OPEN,
                IntervalType.LEFT_OPEN,
            )
            return left_included and right_included
        return True

    def contains_interval(self, other: Interval) -> bool:
        start_contained = self.start < other.start or (
            self.start == other.start
            and self.interval_type in (IntervalType.LEFT_OPEN, IntervalType.RIGHT_OPEN)
        )
        end_contained = self.end > other.end or (
            self.end == other.end
            and self.interval_type in (IntervalType.LEFT_OPEN, IntervalType.LEFT_OPEN)
        )

        return start_contained and end_contained

    def intersection(self, other: Interval) -> Interval | None:
        if not self.overlaps(other):
            return None
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        if new_start == self.start and new_start == other.start:
            left_open = self.interval_type in (
                IntervalType.OPEN,
                IntervalType.LEFT_OPEN,
            ) or other.interval_type in (IntervalType.OPEN, IntervalType.LEFT_OPEN)
        elif new_start == self.start:
            left_open = self.interval_type in (
                IntervalType.OPEN,
                IntervalType.LEFT_OPEN,
            )
        else:
            left_open = other.interval_type in (
                IntervalType.OPEN,
                IntervalType.LEFT_OPEN,
            )
        if new_end == self.end and new_end == other.end:
            right_open = self.interval_type in (
                IntervalType.OPEN,
                IntervalType.RIGHT_OPEN,
            ) or other.interval_type in (IntervalType.OPEN, IntervalType.RIGHT_OPEN)
        elif new_end == self.end:
            right_open = self.interval_type in (
                IntervalType.OPEN,
                IntervalType.RIGHT_OPEN,
            )
        else:
            right_open = other.interval_type in (
                IntervalType.OPEN,
                IntervalType.RIGHT_OPEN,
            )
        if left_open and right_open:
            new_type = IntervalType.OPEN
        elif left_open:
            new_type = IntervalType.LEFT_OPEN
        elif right_open:
            new_type = IntervalType.RIGHT_OPEN
        else:
            new_type = IntervalType.LEFT_OPEN
        return Interval(new_start, new_end, new_type)

    def overlap_duration(self, other: Interval) -> float:
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        return intersection.duration

    def __repr__(self) -> str:
        left = (
            "("
            if self.interval_type in (IntervalType.OPEN, IntervalType.LEFT_OPEN)
            else "["
        )
        right = (
            ")"
            if self.interval_type in (IntervalType.OPEN, IntervalType.RIGHT_OPEN)
            else "]"
        )
        return f"{left}{self.start:.2f}, {self.end:.2f}{right}"


class FunctionalArea(Enum):
    BALCONY = "balcony"
    BATHROOM = "bathroom"
    BEDROOM = "bedroom"
    DININGROOM = "diningroom"
    ENTRANCE = "entrance"
    GARAGE = "garage"
    HALLWAY = "hallway"
    KITCHEN = "kitchen"
    LIVING = "living"
    LIVINGROOM = "livingroom"
    TOILET = "toilet"
    WORKSPACE = "workspace"
    UNDEFINED = "undefined"


class SensorType(Enum):
    CONTACT = "contact"
    DISTANCE = "distance"
    ELECTRIC = "electric"
    FLOAT = "float"
    FLUSH = "flush"
    MOTION = "motion"
    PRESSURE = "pressure"
    SONAR = "sonar"
    TEMPERATURE = "temperature"
    UNDEFINED = "undefined"


class SensorObject(Enum):
    ARMCHAIR = "armchair"
    BATHROOM = "bathroom"
    BATHTUB = "bathtub"
    BED = "bed"
    BEDROOM = "bedroom"
    CABINET = "cabinet"
    CHAIR = "chair"
    CLOSET = "closet"
    COOKTOP = "cooktop"
    COUCH = "couch"
    CUPBOARD = "cupboard"
    DESK = "desk"
    DISHWASHER = "dishwasher"
    DOOR = "door"
    DOORWAY = "doorway"
    DRAWER = "drawer"
    DRESSER = "dresser"
    FREEZER = "freezer"
    FRIDGE = "fridge"
    HALLWAY = "hallway"
    KITCHEN = "kitchen"
    MICROWAVE = "microwave"
    SHOWER = "shower"
    SHOWER_CABINET = "shower_cabinet"
    SINK = "sink"
    SOFA = "sofa"
    STOVE = "stove"
    TABLE = "table"
    TAP = "tap"
    TOASTER = "toaster"
    TOILET = "toilet"
    TV = "tv"
    WARDROBE = "wardrobe"
    WASHING_MACHINE = "washing_machine"
    WC = "wc"
    WINDOW = "window"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class Sensor:
    name: SensorName
    meta: SensorMeta

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class SensorMeta:
    sensor_type: SensorType
    area: FunctionalArea
    sensor_object: SensorObject


@dataclass(frozen=True)
class Event:
    timestamp: Timestamp
    sensor: Sensor
    value: int
    index: int | None = None

    def __post_init__(self) -> None:
        if self.value not in (0, 1):
            raise ValueError(f"Event value must be 0 or 1, got {self.value}")

    def __lt__(self, other: Event) -> bool:
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.sensor.name < other.sensor.name


@dataclass
class EventWindow:
    j_first: int
    j_last: int
    events: list[Event] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.j_last < self.j_first:
            raise ValueError(
                f"EventWindow j_last ({self.j_last}) must be >= j_first ({self.j_first})"
            )

    @property
    def size(self) -> int:
        return self.j_last - self.j_first + 1

    def contains_index(self, j: int) -> bool:
        return self.j_first <= j <= self.j_last


@dataclass
class EventStream:
    events: list[Event]

    def __post_init__(self) -> None:
        if not self.events:
            return
        for i in range(1, len(self.events)):
            if self.events[i].timestamp < self.events[i - 1].timestamp:
                raise ValueError(
                    f"Events must be time-ordered: event {i} at t={self.events[i].timestamp} "
                    f"comes before event {i - 1} at t={self.events[i - 1].timestamp}"
                )
        for j, event in enumerate(self.events):
            object.__setattr__(event, "index", j)

    @property
    def sensors(self) -> set[Sensor]:
        return {event.sensor for event in self.events}

    @property
    def interval(self) -> Interval | None:
        if not self.events:
            return None
        return Interval(
            self.events[0].timestamp, self.events[-1].timestamp, IntervalType.LEFT_OPEN
        )

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, key: int | slice | EventWindow) -> Event | list[Event]:
        if isinstance(key, EventWindow):
            return self.events[key.j_first : key.j_last + 1]
        return self.events[key]

    def find_events_in_interval(
        self, interval: Interval, include_boundaries: bool = True
    ) -> list[int]:
        if not self.events:
            return []
        start_idx = bisect.bisect_left(
            [e.timestamp for e in self.events], interval.start
        )
        end_idx = bisect.bisect_right([e.timestamp for e in self.events], interval.end)
        indices = []
        for j in range(start_idx, end_idx):
            if include_boundaries:
                if interval.contains(self.events[j].timestamp):
                    indices.append(j)
            else:
                indices.append(j)
        return indices

    def filter_by_sensor(self, sensor: Sensor) -> EventStream:
        filtered = [e for e in self.events if e.sensor == sensor]
        return EventStream(filtered)

    def filter_by_area(self, area: FunctionalArea) -> EventStream:
        filtered = [e for e in self.events if e.sensor.meta.area == area]
        return EventStream(filtered)


class SensorActivation(BaseModel):
    interval: Interval = Field(..., description="Sensor activation time interval")
    # name: SensorName = Field(..., description="Sensor identifier in the smart home")
    sensor: Sensor = Field(..., description="Sensor associated with the activation")

    @property
    def duration(self) -> float:
        return self.interval.duration


@dataclass
class SensorActivations:
    data: list[SensorActivation]

    def to_event_stream(self) -> EventStream:
        events = []
        for activation in self.data:
            # Create activation start event (value=1)
            events.append(
                Event(
                    timestamp=activation.interval.start,
                    sensor=activation.sensor,
                    value=1,
                )
            )
            # Create activation end event (value=0)
            events.append(
                Event(
                    timestamp=activation.interval.end,
                    sensor=activation.sensor,
                    value=0,
                )
            )
        events.sort()
        return EventStream(events)


class ActivityOccurrence(BaseModel):
    interval: Interval = Field(..., description="Activity time interval")
    name: ActivityName = Field(..., description="Activity name")

    @property
    def duration(self) -> float:
        return self.interval.duration


@dataclass
class DatasetStatistics:
    num_sensors: int
    num_activities: int
    num_sensor_activations: int
    num_activity_occurrences: int
    time_span: tuple[datetime | None, datetime | None]
    total_duration_seconds: float


@dataclass(frozen=True)
class Segment:
    interval: Interval
    event_indices: tuple[int, int]
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate segment constraints."""
        if self.confidence is not None:
            if not 0 <= self.confidence <= 1:
                raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        j_first, j_last = self.event_indices
        if j_last < j_first:
            raise ValueError(
                f"Invalid event indices: j_last ({j_last}) < j_first ({j_first})"
            )

    @property
    def num_events(self) -> int:
        return self.event_indices[1] - self.event_indices[0] + 1

    @property
    def duration(self) -> float:
        return self.interval.duration

    def to_event_window(self, events: EventStream) -> EventWindow:
        j_first, j_last = self.event_indices
        return EventWindow(
            j_first=j_first,
            j_last=j_last,
            events=events.events[j_first : j_last + 1],
        )


@dataclass
class SegmentationResult:
    segments: list[Segment]
    method_name: str
    parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return sum(seg.duration for seg in self.segments)

    def get_boundaries(self) -> list[Timestamp]:
        boundaries = set()
        for seg in self.segments:
            boundaries.add(seg.interval.start)
            boundaries.add(seg.interval.end)
        return sorted(boundaries)


type SegmentLabel = Mapping[ActivityName, ActivityState]
