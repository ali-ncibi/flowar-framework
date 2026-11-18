import numpy as np
import pytest

from backend.annotation.methods import MostOverlappingAnnotator
from backend.classification.methods import (
    BaseClassificationMethod,
    DecisionTreeClassifier,
    KNeighborsClassifier,
)
from backend.classification.quality import (
    CrossValidationEvaluator,
    compute_balanced_accuracy,
    compute_confusion_matrix,
)
from backend.classification.quality.cv import test_pipeline as run_test_pipeline
from backend.classification.quality.cv import training_pipeline
from backend.dataset import SmartHomeDataset
from backend.representation.methods.manual.vector.stats import StatsVectorExtractor
from backend.segmentation.methods.fixed_length import FixedCount


class TestMetrics:
    def test_balanced_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        score = compute_balanced_accuracy(y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_balanced_accuracy_empty(self):
        assert compute_balanced_accuracy(np.array([]), np.array([])) == 0.0

    def test_confusion_matrix(self):
        cm = compute_confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 0]))
        assert cm.shape == (2, 2)

    def test_confusion_matrix_empty(self):
        cm = compute_confusion_matrix(np.array([]), np.array([]))
        assert cm.size == 0


def make_dataset(dataset_dir) -> SmartHomeDataset:
    return SmartHomeDataset(dataset_dir)


def make_pipeline_methods():
    return (
        FixedCount(nb_events=2),
        StatsVectorExtractor(["sensor_event_count", "segment_duration"]),
        MostOverlappingAnnotator(),
        KNeighborsClassifier(n_neighbors=1),
    )


class RecordingClassifier(BaseClassificationMethod):
    """Classifier stub that records the masks it receives."""

    def __init__(self):
        self.fit_masks = None
        self.predict_masks = None
        self.fit_X_ndim = 0

    def fit(self, X, y, **kwargs):
        self.fit_masks = kwargs.get("masks")
        self.fit_X_ndim = np.asarray(X).ndim
        return self

    def predict(self, X, **kwargs):
        self.predict_masks = kwargs.get("masks")
        return np.zeros(len(X), dtype=int)


class MaskedExtractor(StatsVectorExtractor):
    """Extractor stub that returns non-empty masks."""

    def transform(self, segments, sensor_activations):
        features, _ = super().transform(segments, sensor_activations)
        masks = np.ones((len(segments), 8), dtype=np.float32)
        return features, masks


class TestPipelines:
    def test_training_pipeline(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        segmenter, extractor, annotator, classifier = make_pipeline_methods()
        out = training_pipeline(ds, segmenter, extractor, annotator, classifier)
        seg2, ext2, ann2, clf2, labels_dict = out
        assert seg2 is segmenter
        assert ext2 is extractor
        assert ann2 is annotator
        assert clf2 is classifier
        assert isinstance(labels_dict, dict)
        assert set(labels_dict.values()) <= {"COOKING", "SHOWERING"}

    def test_training_pipeline_nonempty_masks(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        segmenter = FixedCount(nb_events=2)
        extractor = MaskedExtractor(["segment_duration"])
        annotator = MostOverlappingAnnotator()
        classifier = RecordingClassifier()
        training_pipeline(ds, segmenter, extractor, annotator, classifier)
        assert classifier.fit_masks is not None
        assert len(classifier.fit_masks) > 0

    def test_test_pipeline(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        segmenter, extractor, annotator, classifier = make_pipeline_methods()
        training_pipeline(ds, segmenter, extractor, annotator, classifier)
        seg_f1, clf_acc = run_test_pipeline(
            ds, segmenter, extractor, annotator, classifier
        )
        assert 0.0 <= seg_f1 <= 1.0
        assert 0.0 <= clf_acc <= 1.0

    def test_test_pipeline_with_labels_dict_and_masks(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        segmenter = FixedCount(nb_events=2)
        extractor = MaskedExtractor(["segment_duration"])
        annotator = MostOverlappingAnnotator()
        classifier = RecordingClassifier()
        _, _, _, _, labels_dict = training_pipeline(
            ds, segmenter, extractor, annotator, classifier
        )
        seg_f1, clf_acc = run_test_pipeline(
            ds, segmenter, extractor, annotator, classifier, labels_dict
        )
        assert classifier.predict_masks is not None
        assert len(classifier.predict_masks) > 0
        assert 0.0 <= seg_f1 <= 1.0
        assert 0.0 <= clf_acc <= 1.0


class TestCrossValidationEvaluator:
    def test_splits_two_days(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        evaluator = CrossValidationEvaluator(ds)
        assert len(evaluator.splits) == 2

    def test_n_folds_limits_splits(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        evaluator = CrossValidationEvaluator(ds, n_folds=1)
        assert len(evaluator.splits) == 1

    def test_n_folds_larger_than_splits_keeps_all(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        evaluator = CrossValidationEvaluator(ds, n_folds=99)
        assert len(evaluator.splits) == 2

    def test_empty_splits_raise(self, dataset_dir, monkeypatch):
        ds = make_dataset(dataset_dir)
        monkeypatch.setattr(ds, "leave_one_day_out_split", lambda: [])
        with pytest.raises(ValueError, match="at least one day"):
            CrossValidationEvaluator(ds)

    def test_evaluate_hyperparams_success(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        evaluator = CrossValidationEvaluator(ds, n_folds=1)
        segmenter, extractor, _annotator = (
            FixedCount(nb_events=2),
            StatsVectorExtractor(["sensor_event_count"]),
            MostOverlappingAnnotator(),
        )
        mean_score, fold_scores = evaluator.evaluate_hyperparams(
            KNeighborsClassifier,
            {"n_neighbors": 1},
            segmenter,
            extractor,
            MostOverlappingAnnotator(),
        )
        assert len(fold_scores) == 1
        assert mean_score == fold_scores[0]
        assert 0.0 <= mean_score <= 1.0

    def test_evaluate_hyperparams_all_folds_fail(self, dataset_dir):
        ds = make_dataset(dataset_dir)
        evaluator = CrossValidationEvaluator(ds, n_folds=1)
        # max_depth=-1 makes sklearn raise inside the fold, which is skipped.
        mean_score, fold_scores = evaluator.evaluate_hyperparams(
            DecisionTreeClassifier,
            {"max_depth": -1},
            FixedCount(nb_events=2),
            StatsVectorExtractor(["sensor_event_count"]),
            MostOverlappingAnnotator(),
        )
        assert mean_score == 0.0
        assert fold_scores == []
