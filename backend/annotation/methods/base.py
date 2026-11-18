from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import cast

from numpy import ndarray
from pandas import DataFrame

from backend.models import ActivityName, ActivityOccurrence, Segment, SegmentLabel

type AnnotationMethod = "BaseAnnotationMethod"


class BaseAnnotationMethod(ABC):
    @abstractmethod
    def transform(
        self,
        segments: list[Segment],
        activity_occurrences: Sequence[ActivityOccurrence],
    ) -> tuple[ndarray, Mapping[int, ActivityName]]:
        pass


def convert_to_df(segment_labels: Sequence[SegmentLabel]) -> DataFrame:
    df = DataFrame(segment_labels).astype(int)
    colnames = sorted(df.columns)
    # A list key always selects columns, i.e. returns a DataFrame
    return cast("DataFrame", df[colnames])


def convert_to_array(segment_labels: DataFrame) -> ndarray:
    return segment_labels.astype("int").to_numpy().argmax(axis=1)
