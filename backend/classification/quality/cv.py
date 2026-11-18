from collections.abc import Mapping
from typing import Any

from backend.annotation.methods import BaseAnnotationMethod
from backend.classification.methods import BaseClassificationMethod
from backend.dataset import SmartHomeDataset
from backend.models import ActivityName
from backend.representation.methods import BaseRepresentationMethod
from backend.segmentation.methods import BaseSegmentationMethod
from backend.segmentation.quality import compute_f1_score, get_true_boundaries

from .metrics import compute_balanced_accuracy


def training_pipeline(
    ds: SmartHomeDataset,
    segmenter: BaseSegmentationMethod,
    extractor: BaseRepresentationMethod,
    annotator: BaseAnnotationMethod,
    classifier: BaseClassificationMethod,
) -> tuple[
    BaseSegmentationMethod,
    BaseRepresentationMethod,
    BaseAnnotationMethod,
    BaseClassificationMethod,
    Mapping[int, ActivityName],
]:
    events = ds.to_event_stream()
    segmenter.fit(events, activity_labels=ds.extract_activity_labels(events))
    seg_res = segmenter.transform(events)
    segments = seg_res.segments
    extractor.fit(ds.sensor_activations, segments=segments)
    features, masks = extractor.transform(segments, ds.sensor_activations)
    labels, labels_dict = annotator.transform(segments, ds.activity_occurrences)
    masks = masks if masks is not None and len(masks) > 0 else None
    classifier.fit(features, labels, masks=masks)
    return segmenter, extractor, annotator, classifier, labels_dict


def test_pipeline(
    test_ds: SmartHomeDataset,
    segmenter: BaseSegmentationMethod,
    extractor: BaseRepresentationMethod,
    annotator: BaseAnnotationMethod,
    classifier: BaseClassificationMethod,
    labels_dict: Mapping[int, ActivityName] | None = None,
) -> tuple[float, float]:
    events = test_ds.to_event_stream()
    seg_res = segmenter.transform(events)
    tolerance_seconds = 30.0
    true_bounds = get_true_boundaries(test_ds.activity_occurrences)
    pred_bounds = seg_res.get_boundaries()
    seg_f1 = compute_f1_score(true_bounds, pred_bounds, tolerance_seconds)
    features, masks = extractor.transform(seg_res.segments, test_ds.sensor_activations)
    masks = masks if masks is not None and len(masks) > 0 else None
    labels_pred = classifier.predict(features, masks=masks)
    labels_true, _ = annotator.transform(seg_res.segments, test_ds.activity_occurrences)
    clf_acc = compute_balanced_accuracy(labels_true, labels_pred)
    return seg_f1, clf_acc


class CrossValidationEvaluator:
    def __init__(
        self,
        dataset: SmartHomeDataset,
        n_folds: int | None = None,
    ):
        self.dataset = dataset
        self.splits = dataset.leave_one_day_out_split()
        if n_folds and n_folds < len(self.splits):
            self.splits = self.splits[:n_folds]
        if not self.splits:
            raise ValueError("Dataset must have at least one day of data for CV splits")

    def evaluate_hyperparams(
        self,
        classifier_class: type,
        params: dict[str, Any],
        segmenter: BaseSegmentationMethod,
        extractor: BaseRepresentationMethod,
        annotator: BaseAnnotationMethod,
    ) -> tuple[float, list[float]]:
        fold_scores = []
        for fold_idx, (train_ds, test_ds) in enumerate(self.splits):
            try:
                classifier = classifier_class(**params)
                # Clone the methods for each fold to avoid data leakage.
                # The annotator is NOT cloned because it does not learn from data
                # (it only maps segments to activity labels based on ground truth).
                segmenter_fold = segmenter.__deepcopy__({})
                extractor_fold = extractor.__deepcopy__({})
                segmenter_fold, extractor_fold, annotator, classifier, labels_dict = (
                    training_pipeline(
                        train_ds,
                        segmenter_fold,
                        extractor_fold,
                        annotator,
                        classifier,
                    )
                )
                _, score = test_pipeline(
                    test_ds,
                    segmenter_fold,
                    extractor_fold,
                    annotator,
                    classifier,
                    labels_dict,
                )
                fold_scores.append(score)
            except Exception as e:
                print(f"Error in fold {fold_idx}: {str(e)}. Skipping this fold.")
                continue
        if not fold_scores:
            return 0.0, []
        mean_score = sum(fold_scores) / len(fold_scores)
        return mean_score, fold_scores
