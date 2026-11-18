from collections.abc import Mapping, Sequence

from numpy import ndarray

from backend.models import ActivityName, ActivityOccurrence, Segment

from .base import BaseAnnotationMethod, convert_to_array, convert_to_df


class MostOverlappingAnnotator(BaseAnnotationMethod):
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
                    lambda activity_occurrence: segment.interval.overlaps(
                        other=activity_occurrence.interval
                    ),
                    activity_occurrences,
                )
            )
            if len(occs_intersecting_with) == 0:
                activity_during = "IDLE"
            else:
                ov_durations = [
                    segment.interval.overlap_duration(other=occ.interval)
                    for occ in occs_intersecting_with
                ]
                max_ov_index = ov_durations.index(max(ov_durations))
                activity_during = occs_intersecting_with[max_ov_index].name
            segment_label = {
                activity_name: (activity_name == activity_during)
                for activity_name in activity_names
            }
            segment_labels.append(segment_label)
        idx_to_activity = {
            idx: activity_name
            for idx, activity_name in enumerate(sorted(activity_names))
        }
        return convert_to_array(convert_to_df(segment_labels)), idx_to_activity
