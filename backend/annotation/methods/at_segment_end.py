from collections.abc import Mapping, Sequence

from numpy import ndarray

from backend.models import ActivityName, ActivityOccurrence, Segment

from .base import BaseAnnotationMethod, convert_to_array, convert_to_df


class AtSegmentEndAnnotator(BaseAnnotationMethod):
    def transform(
        self,
        segments: list[Segment],
        activity_occurrences: Sequence[ActivityOccurrence],
    ) -> tuple[ndarray, Mapping[int, ActivityName]]:
        activity_names = set(activity.name for activity in activity_occurrences)
        segment_labels = []
        for segment in segments:
            occs_intersecting_with = list(
                filter(
                    lambda activity_occurrence: activity_occurrence.interval.contains(
                        segment.interval.end
                    ),
                    activity_occurrences,
                )
            )
            activities_at_end = set(occ.name for occ in occs_intersecting_with)
            segment_label = {
                activity_name: (activity_name in activities_at_end)
                for activity_name in activity_names
            }
            segment_labels.append(segment_label)
        idx_to_activity = {
            idx: activity_name
            for idx, activity_name in enumerate(sorted(activity_names))
        }
        return convert_to_array(convert_to_df(segment_labels)), idx_to_activity
