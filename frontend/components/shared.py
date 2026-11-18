from typing import Any

import streamlit as st

from backend.annotation.methods import (
    AnyOverlappingAnnotator,
    AtSegmentEndAnnotator,
    BaseAnnotationMethod,
    MostOverlappingAnnotator,
)
from backend.classification.methods import (
    BaseClassificationMethod,
    DecisionTreeClassifier,
    DeepSequenceGridClassifier,
    KNeighborsClassifier,
)
from backend.classification.quality import compute_balanced_accuracy
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
    FixedTime,
    RulsifPeCpd,
    RulsifPlsbdCpd,
    SepCpd,
    W2vCcsCPD,
    W2vCcsdCPD,
    W2vCsCPD,
)

SEGMENTATION_METHODS = [
    "SepCpd",
    "RulsifPlsbdCpd",
    "RulsifPeCpd",
    "CorrWan",
    "CorrXu",
    "W2vCsCPD",
    "W2vCcsCPD",
    "W2vCcsdCPD",
    "FixedCount",
    "FixedTime",
]

VECTOR_STAT_FEATURES = [
    "sensor_state_changed",
    "sensor_last_fired",
    "sensor_is_active",
    "sensor_shifted_activation",
    "sensor_event_count",
    "sensor_dwell_time",
    "segment_trigonometric_time_encoding",
    "segment_duration",
]


def fixed_count_params(params):
    st.write("**FixedCount** (Fixed number of events per segment)")
    params["nb_events"] = st.number_input(
        "Number of Events per Segment", min_value=1, max_value=100, value=10
    )
    params["overlap_percentage"] = st.number_input(
        "Overlap Percentage",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )
    return params


def fixed_time_params(params):
    st.write("**FixedTime** (Fixed time length per segment)")
    params["length_in_seconds"] = st.number_input(
        "Segment Duration (seconds)",
        min_value=1.0,
        max_value=3600.0,
        value=60.0,
        step=1.0,
    )
    params["overlap_percentage"] = st.number_input(
        "Overlap Percentage",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
    )
    return params


def sep_cpd_params(params):
    st.write("**SepCpd** - Distributional Change Point Detection")
    params["local_context_size"] = st.number_input(
        "local_context_size (feature window length)",
        min_value=1,
        max_value=20,
        value=5,
    )
    params["number_of_view_samples"] = st.number_input(
        "number_of_view_samples (samples per view)",
        min_value=1,
        max_value=50,
        value=10,
    )
    params["number_of_bases"] = st.number_input(
        "number_of_bases (basis functions)", min_value=1, max_value=20, value=5
    )
    params["kernel_width"] = st.number_input(
        "kernel_width (kernel bandwidth)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    params["lambda_reg"] = st.number_input(
        "lambda_reg (regularization)",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
    )
    params["threshold"] = st.number_input(
        "threshold (detection threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    return params


def corr_wan_params(params):
    st.write("**CorrWan** - Pearson correlation based")
    params["threshold"] = st.number_input(
        "threshold (correlation threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    params["time_bin_size"] = st.number_input(
        "time_bin_size (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
    )
    return params


def corr_xu_params(params):
    st.write("**CorrXu** - Mutual information based")
    params["context_size"] = st.number_input(
        "context_size (MI window length)", min_value=2, max_value=30, value=10
    )
    return params


def rulsif_cpd_params(params):
    st.write("**RulsifPlsbdCpd** (Pearson-like scaled Bregman divergence CPD)")
    params["local_context_size"] = st.number_input(
        "local_context_size (feature window length)",
        min_value=1,
        max_value=20,
        value=5,
    )
    params["number_of_view_samples"] = st.number_input(
        "number_of_view_samples (samples per view)",
        min_value=1,
        max_value=50,
        value=10,
    )
    params["number_of_bases"] = st.number_input(
        "number_of_bases (basis functions)", min_value=1, max_value=20, value=5
    )
    params["kernel_width"] = st.number_input(
        "kernel_width (kernel bandwidth)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    params["lambda_reg"] = st.number_input(
        "lambda_reg (regularization)",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
    )
    params["threshold"] = st.number_input(
        "threshold (detection threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    params["alpha_relative"] = st.number_input(
        "alpha_relative (mixture parameter)",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    return params


def w2v_cpd_common_params(params):
    st.write("**Common W2V CPD Parameters**")
    params["embedding_w2v_dimension"] = st.number_input(
        "embedding_w2v_dimension (embedding dimensionality)",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )
    params["embedding_w2v_context_size"] = st.number_input(
        "embedding_w2v_context_size (context window size)",
        min_value=2,
        max_value=10,
        value=5,
    )
    params["embedding_w2v_training_epochs"] = st.number_input(
        "embedding_w2v_training_epochs (training iterations)",
        min_value=1,
        max_value=50,
        value=10,
    )
    params["embedding_w2v_min_token_count"] = st.number_input(
        "embedding_w2v_min_token_count (minimum word frequency)",
        min_value=1,
        max_value=10,
        value=1,
    )
    params["embedding_w2v_skipgram_or_cbow"] = st.selectbox(
        "embedding_w2v_skipgram_or_cbow (training architecture)",
        options=[1, 0],
        format_func=lambda x: "Skip-gram" if x == 1 else "CBOW",
        index=0,
        help="Skip-gram (1) or Continuous Bag of Words (0)",
    )
    params["embedding_w2v_negative_sampling"] = st.number_input(
        "embedding_w2v_negative_sampling (negative samples)",
        min_value=1,
        max_value=25,
        value=5,
        help="Number of negative samples for negative sampling",
    )
    return params


def w2v_cs_params(params):
    st.write("**W2vCsCPD** (Word2Vec Cosine Similarity CPD)")
    params["threshold"] = st.number_input(
        "threshold (cosine similarity threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    params = w2v_cpd_common_params(params)
    return params


def w2v_ccs_params(params):
    st.write("**W2vCcsCPD** (Word2Vec Context-aware Cosine Similarity)")
    params["threshold"] = st.number_input(
        "threshold (CCS threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    params["context_size"] = st.number_input(
        "context_size (context radius)", min_value=1, max_value=10, value=3
    )
    params = w2v_cpd_common_params(params)
    return params


def w2v_ccsd_params(params):
    st.write("**W2vCcsdCPD** (Word2Vec Context-aware Cosine Similarity with Drift)")
    params["context_size"] = st.number_input(
        "context_size (context radius)", min_value=1, max_value=10, value=3
    )
    params["delta"] = st.number_input(
        "delta (trend threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
    )
    params = w2v_cpd_common_params(params)
    return params


def setup_segmentation_params(method_name: str) -> dict:
    params: dict[str, Any] = {}
    if method_name in ["SepCpd", "RulsifPlsbdCpd", "RulsifPeCpd"]:
        selected_features = st.multiselect(
            f"{method_name} Features",
            options=SepCpd.get_all_features(),
            default=SepCpd.get_default_features(),
            help="At least one feature must be selected. Multiple features will be concatenated.",
        )
        if not selected_features:
            st.warning("Please select at least one feature")
            st.stop()
        params["features"] = selected_features
    if method_name == "SepCpd":
        params = sep_cpd_params(params)
    elif method_name == "CorrWan":
        params = corr_wan_params(params)
    elif method_name == "CorrXu":
        params = corr_xu_params(params)
    elif method_name == "RulsifPlsbdCpd":
        params = rulsif_cpd_params(params)
    elif method_name == "RulsifPeCpd":
        params = rulsif_cpd_params(params)
    elif method_name == "W2vCsCPD":
        params = w2v_cs_params(params)
    elif method_name == "W2vCcsCPD":
        params = w2v_ccs_params(params)
    elif method_name == "W2vCcsdCPD":
        params = w2v_ccsd_params(params)
    elif method_name == "FixedCount":
        params = fixed_count_params(params)
    elif method_name == "FixedTime":
        params = fixed_time_params(params)
    return params


def matrix_fuzzy_params(params):
    st.write("**Fuzzy Matrix Representation Configuration**")
    st.caption("Fuzzy logic-based sensor interactions")
    params["minutes"] = st.number_input("Minutes", min_value=1, max_value=120, value=30)
    return params


def matrix_textual_params(params):
    st.write("**Textual Matrix Representation Configuration**")
    st.caption("Textual similarity-based sensor interactions")
    params["textual_representation"] = st.selectbox(
        "Textual representation",
        options=[
            "tdost_plain_basic_tfidf",
            "tdost_plain_basic_word2vec",
            "tdost_plain_basic_sentencebert",
            "tdost_plain_temporal_tfidf",
            "tdost_plain_temporal_word2vec",
            "tdost_plain_temporal_sentencebert",
            "civitarese_tfidf",
            "civitarese_word2vec",
            "civitarese_sentencebert",
            "dhekane_tfidf",
            "dhekane_word2vec",
            "dhekane_sentencebert",
        ],
        index=0,
    )
    return params


def setup_matrix_representation_params(
    matrix_method: str,
) -> dict:
    params: dict[str, Any] = {}
    if matrix_method == "Fuzzy":
        params = matrix_fuzzy_params(params)
    elif matrix_method == "Textual":
        params = matrix_textual_params(params)
    return params


def text_vector_representation_params() -> dict:
    params: dict[str, Any] = {}
    params["max_len"] = st.number_input(
        "Maximum Text Length", min_value=10, max_value=1000, value=200
    )
    params["name"] = st.selectbox(
        "Text Vector Name",
        options=[
            "tdost_plain_basic_tfidf",
            "tdost_plain_basic_word2vec",
            "tdost_plain_basic_sentencebert",
        ],
        index=0,
    )
    if "tfidf" in params["name"]:
        params["max_features"] = st.number_input(
            "Max Features (TF-IDF)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
        )
        params["ngram_range"] = (
            st.number_input(
                "N-gram Range Min (TF-IDF)", min_value=1, max_value=5, value=1
            ),
            st.number_input(
                "N-gram Range Max (TF-IDF)", min_value=1, max_value=5, value=1
            ),
        )
    elif "word2vec" in params["name"]:
        params["aggregation"] = st.selectbox(
            "Word2Vec Aggregation Method",
            options=["mean", "sum"],
            index=0,
        )
        params["use_tfidf"] = st.checkbox(
            "Use TF-IDF Weighting",
            value=True,
            help="Use TF-IDF weighting for Word2Vec embeddings",
        )
        if params["use_tfidf"]:
            params["idf_power"] = st.number_input(
                "TF-IDF IDF Power",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
            )
            params["tfidf_params"] = {
                "max_features": st.number_input(
                    "TF-IDF Max Features (for weighting)",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                ),
                "ngram_range": (
                    st.number_input(
                        "TF-IDF N-gram Range Min (for weighting)",
                        min_value=1,
                        max_value=5,
                        value=1,
                    ),
                    st.number_input(
                        "TF-IDF N-gram Range Max (for weighting)",
                        min_value=1,
                        max_value=5,
                        value=1,
                    ),
                ),
            }
        params["l2_normalize"] = st.checkbox(
            "L2 Normalize Embeddings",
            value=False,
            help="L2 normalize Word2Vec embeddings",
        )
    return params


def stats_vector_representation_params() -> dict:
    params: dict[str, Any] = {}
    params["desired_features"] = st.multiselect(
        "Statistical Features",
        options=VECTOR_STAT_FEATURES,
        default=VECTOR_STAT_FEATURES,
    )
    return params


def setup_representation_params(rep_method_name: str) -> dict:
    params: dict[str, Any] = {}
    if rep_method_name == "StatsVectorExtractor":
        params = stats_vector_representation_params()
    elif rep_method_name == "TextualVectorExtractor":
        params = text_vector_representation_params()
    elif rep_method_name == "FuzzyMatrixExtractor":
        params = matrix_fuzzy_params(params)
    elif rep_method_name == "TextualMatrixExtractor":
        params = matrix_textual_params(params)
    return params


def handle_segmentation_execution(dataset, method_name: str, params: dict):
    with st.spinner("Calibrating on dataset..."):
        events = dataset.to_event_stream()
        activity_labels = dataset.extract_activity_labels(events)
        if method_name == "SepCpd":
            segmenter: BaseSegmentationMethod = SepCpd(random_state=42, **params)
        elif method_name == "CorrWan":
            segmenter = CorrWan(**params)
        elif method_name == "CorrXu":
            segmenter = CorrXu(**params)
        elif method_name == "RulsifPlsbdCpd":
            segmenter = RulsifPlsbdCpd(random_state=42, **params)
        elif method_name == "RulsifPeCpd":
            segmenter = RulsifPeCpd(random_state=42, **params)
        elif method_name == "W2vCsCPD":
            segmenter = W2vCsCPD(random_state=42, **params)
        elif method_name == "W2vCcsCPD":
            segmenter = W2vCcsCPD(random_state=42, **params)
        elif method_name == "W2vCcsdCPD":
            segmenter = W2vCcsdCPD(random_state=42, **params)
        elif method_name == "FixedCount":
            segmenter = FixedCount(**params)
        elif method_name == "FixedTime":
            segmenter = FixedTime(**params)
        else:
            st.error(f"Unknown segmentation method: {method_name}")
            st.stop()
        segmenter.fit(events, activity_labels=activity_labels)
        st.success("✓ Calibration complete")
    with st.spinner("Performing segmentation..."):
        res = segmenter.transform(events)

    return res


def handle_annotation_execution(dataset, segments, annotation_method_name: str):
    if annotation_method_name == "MostOverlappingAnnotator":
        annotator: BaseAnnotationMethod = MostOverlappingAnnotator()
    elif annotation_method_name == "AnyOverlappingAnnotator":
        annotator = AnyOverlappingAnnotator()
    elif annotation_method_name == "AtSegmentEndAnnotator":
        annotator = AtSegmentEndAnnotator()
    else:
        st.error(f"Unknown annotation method: {annotation_method_name}")
        st.stop()
    segment_labels, activity_mapping = annotator.transform(
        segments=segments,
        activity_occurrences=dataset.activity_occurrences,
    )
    return segment_labels, activity_mapping


def handle_representation_execution(
    dataset,
    segments,
    representation_type: str,
    rep_method_name: str,
    rep_method_params: dict,
):
    with st.spinner("Extracting segment representations..."):
        extractor: BaseRepresentationMethod | None = None
        extracted: Any = None
        if representation_type == "Vector":
            if rep_method_name == "StatsVectorExtractor":
                extractor = StatsVectorExtractor(**rep_method_params)
            elif rep_method_name == "TextualVectorExtractor":
                extractor = TextualVectorExtractor(**rep_method_params)
            else:
                st.error(f"Unknown vector representation method: {rep_method_name}")
                st.stop()
        else:
            if rep_method_name == "FuzzyMatrixExtractor":
                extractor = FuzzyMatrixExtractor(**rep_method_params)
            elif rep_method_name == "TextualMatrixExtractor":
                extractor = TextualMatrixExtractor(**rep_method_params)
            else:
                st.error(f"Unknown matrix representation method: {rep_method_name}")
                st.stop()
        assert extractor is not None
        extractor.fit(dataset.sensor_activations, segments=segments)
        extracted = extractor.transform(
            segments=segments, sensor_activations=dataset.sensor_activations
        )
    extracted, mask = extractor.transform(segments, dataset.sensor_activations)
    return extracted, mask


def setup_classification_params(classification_method_name: str) -> dict:
    params: dict[str, Any] = {}
    if classification_method_name == "KNeighborsClassifier":
        params["n_neighbors"] = st.number_input(
            "Number of Neighbors (K)",
            min_value=1,
            max_value=20,
            value=5,
        )
        params["weights"] = st.selectbox(
            "Weights",
            options=["uniform", "distance"],
            index=0,
        )
        params["leaf_size"] = st.number_input(
            "Leaf Size",
            min_value=10,
            max_value=100,
            value=30,
        )
        params["p"] = st.selectbox(
            "Power Parameter for Minkowski Metric",
            options=[1, 2],
            format_func=lambda x: "Manhattan (1)" if x == 1 else "Euclidean (2)",
            index=1,
        )
        params["metric"] = st.selectbox(
            "Distance Metric",
            options=["minkowski", "euclidean", "manhattan"],
            index=0,
        )
    elif classification_method_name == "DecisionTreeClassifier":
        params["max_depth"] = st.number_input(
            "Max Depth",
            min_value=1,
            max_value=50,
            value=10,
        )
        params["criterion"] = st.selectbox(
            "Criterion",
            options=["gini", "entropy", "log_loss"],
            index=0,
        )
        params["splitter"] = st.selectbox(
            "Splitter",
            options=["best", "random"],
            index=0,
        )
        params["min_samples_split"] = st.number_input(
            "Min Samples Split",
            min_value=2,
            max_value=20,
            value=2,
        )
        params["min_samples_leaf"] = st.number_input(
            "Min Samples Leaf",
            min_value=1,
            max_value=20,
            value=1,
        )
    elif classification_method_name == "DeepSequenceGridClassifier":
        params["backbone"] = st.selectbox(
            "Backbone Architecture",
            options=["rnn", "lstm", "bilstm", "cnn1d", "cnn2d"],
            index=1,
        )
        params["use_attention"] = st.checkbox(
            "Use Attention Pooling", value=True, help="Enable attention mechanism"
        )
        params["hidden_size"] = st.number_input(
            "Hidden Size", min_value=16, max_value=512, value=128, step=16
        )
        params["num_layers"] = st.number_input(
            "Number of Layers", min_value=1, max_value=10, value=2
        )
        params["batch_size"] = st.number_input(
            "Batch Size", min_value=8, max_value=256, value=32, step=8
        )
        params["learning_rate_init"] = st.number_input(
            "Initial Learning Rate",
            min_value=1e-5,
            max_value=1e-1,
            value=1e-3,
            format="%.5f",
        )
    return params


def handle_classification_execution(
    features,
    labels,
    masks,
    classification_method_name: str,
    classification_method_params: dict,
):
    with st.spinner("Training and evaluating classifier..."):
        classifier: BaseClassificationMethod | None = None
        if classification_method_name == "KNeighborsClassifier":
            classifier = KNeighborsClassifier(**classification_method_params)
        elif classification_method_name == "DecisionTreeClassifier":
            classifier = DecisionTreeClassifier(**classification_method_params)
        elif classification_method_name == "DeepSequenceGridClassifier":
            classifier = DeepSequenceGridClassifier(**classification_method_params)
        else:
            st.error(f"Unknown classification method: {classification_method_name}")
            st.stop()
        masks = masks if masks is not None and len(masks) == features.shape[0] else None
        classifier.fit(features, labels, masks=masks)
        predictions = classifier.predict(features, masks=masks)
        accuracy = compute_balanced_accuracy(labels, predictions)
        st.success(f"✓ Classification complete - Accuracy: {accuracy:.2f}")
    return {
        "classifier": classifier,
        "predictions": predictions,
        "accuracy": accuracy,
    }
