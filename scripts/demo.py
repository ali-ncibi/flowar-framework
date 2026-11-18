from collections.abc import Mapping
from pathlib import Path

from backend.annotation.methods import BaseAnnotationMethod, MostOverlappingAnnotator
from backend.classification.methods import (
    BaseClassificationMethod,
    DecisionTreeClassifier,
    DeepSequenceGridClassifier,
    KNeighborsClassifier,
)
from backend.dataset import SmartHomeDataset
from backend.models import ActivityName
from backend.representation.methods import (
    BaseRepresentationMethod,
    FuzzyMatrixExtractor,
    StatsVectorExtractor,
    TextualMatrixExtractor,
    TextualVectorExtractor,
)
from backend.segmentation.methods import (
    BaseSegmentationMethod,
    CorrWan,
    CorrXu,
    FixedCount,
    SepCpd,
    W2vCcsCPD,
)
from backend.segmentation.quality import compute_f1_score, get_true_boundaries


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
    classifier.fit(
        features,
        labels,
        masks=masks if masks is not None and len(masks) > 0 else None,
    )
    return segmenter, extractor, annotator, classifier, labels_dict


def evaluation_pipeline(
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
    print(f"(Segmentation) F1 Score (Tol. {tolerance_seconds} s):", f"{seg_f1:.4f}")
    features, masks = extractor.transform(seg_res.segments, test_ds.sensor_activations)
    labels_pred = classifier.predict(
        features, masks=masks if masks is not None and len(masks) > 0 else None
    )
    labels_true, _ = annotator.transform(seg_res.segments, test_ds.activity_occurrences)
    clf_acc = (labels_pred == labels_true).mean()
    print(f"(Classification) Accuracy: {clf_acc:.4f}")
    return seg_f1, clf_acc


# Dataset
path = (
    Path(__file__).parent.parent.resolve()
    / "backend"
    / "dataset"
    / "content"
    / "ordonez-a"
)
num_days = None  # Set to an integer to limit the number of loaded days
dataset = SmartHomeDataset(dataset_path=path, num_days=num_days)
print("Dataset Path:", path)
print("Dataset Name:", dataset.name)
print("Number of Loaded Days:", dataset.num_days)
print("Time-Span:", dataset.get_time_span())
print("Available Sensors:", len(dataset.get_sensor_names()))
print("Available Activities:", len(dataset.get_activity_names()))
print("N° of Activity Occurrences:", len(dataset.activity_occurrences))
print("N° of Sensor Activations:", len(dataset.sensor_activations.data))
print("*" * 40)


# Data Splitting
splits = dataset.train_test_split(test_size=0.7)
splits = dataset.leave_one_day_out_split()
print("Number of splits:", len(splits))
print("*" * 40)

# CV Loop
total_seg_f1 = 0.0
total_clf_acc = 0.0
for i, (train_ds, test_ds) in enumerate(splits):
    # Initialize Methods
    segmenter: BaseSegmentationMethod = CorrWan()
    segmenter = CorrXu()
    segmenter = SepCpd()
    segmenter = W2vCcsCPD()
    segmenter = FixedCount(nb_events=4)
    extractor: BaseRepresentationMethod = FuzzyMatrixExtractor()
    extractor = StatsVectorExtractor(
        desired_features=[
            "sensor_is_active",
            "sensor_shifted_activation",
        ]
    )
    extractor = TextualMatrixExtractor(
        ngram_range=(1, 2),
        max_features=5000,
        analyzer="word",
    )
    extractor = TextualVectorExtractor(
        ngram_range=(1, 2),
        max_features=5000,
        analyzer="word",
    )
    annotator: BaseAnnotationMethod = MostOverlappingAnnotator()
    classifier: BaseClassificationMethod = KNeighborsClassifier()
    classifier = DeepSequenceGridClassifier()
    classifier = DecisionTreeClassifier()

    print(f"Split {i + 1}/{len(splits)}")
    print("Train Span:", train_ds.get_time_span())
    print("N° of Train Activity Occurrences:", len(train_ds.activity_occurrences))
    print("N° of Train Sensor Activations:", len(train_ds.sensor_activations.data))
    print("Test Span:", test_ds.get_time_span())
    print("N° of Test Activity Occurrences:", len(test_ds.activity_occurrences))
    print("N° of Test Sensor Activations:", len(test_ds.sensor_activations.data))
    segmenter, extractor, annotator, classifier, labels_dict = training_pipeline(
        train_ds,
        segmenter,
        extractor,
        annotator,
        classifier,
    )
    seg_f1, clf_acc = evaluation_pipeline(
        test_ds,
        segmenter,
        extractor,
        annotator,
        classifier,
        labels_dict,
    )
    total_seg_f1 += seg_f1
    total_clf_acc += clf_acc
    print("-" * 40)

print("Cross-Validation Results:")
avg_seg_f1 = total_seg_f1 / len(splits)
avg_clf_acc = total_clf_acc / len(splits)
print("Average Segmentation F1 Score:", f"{avg_seg_f1:.4f}")
print("Average Classification Accuracy:", f"{avg_clf_acc:.4f}")
