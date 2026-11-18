from numpy import array, ndarray

from backend.models import Segment, SensorActivations

from .....methods import BaseRepresentationMethod
from ....textual.textualization import TDOSTPlainBasic, TDOSTPlainTemporal
from ....textual.vectorization import (
    SentenceTransformerVectorizer,
    TfidfVectorizer,
    Word2VecVectorizer,
)


class TextualVectorExtractor(BaseRepresentationMethod):
    def __init__(
        self,
        max_len: int = 30,
        name: str = "tdost_plain_basic_tfidf",
        **kwargs,
    ):
        self.max_len = max_len
        self.name = name
        if self.name == "tdost_plain_basic_tfidf":
            self.textualizer: TDOSTPlainBasic | TDOSTPlainTemporal = TDOSTPlainBasic()
            self.vectorizer: (
                TfidfVectorizer | Word2VecVectorizer | SentenceTransformerVectorizer
            ) = TfidfVectorizer(**kwargs)
        elif self.name == "tdost_plain_basic_word2vec":
            self.textualizer = TDOSTPlainBasic()
            self.vectorizer = Word2VecVectorizer(**kwargs)
        elif self.name == "tdost_plain_basic_sentence_transformer":
            self.textualizer = TDOSTPlainBasic()
            self.vectorizer = SentenceTransformerVectorizer(**kwargs)
        elif self.name == "tdost_plain_temporal_tfidf":
            self.textualizer = TDOSTPlainTemporal()
            self.vectorizer = TfidfVectorizer(**kwargs)
        elif self.name == "tdost_plain_temporal_word2vec":
            self.textualizer = TDOSTPlainTemporal()
            self.vectorizer = Word2VecVectorizer(**kwargs)
        elif self.name == "tdost_plain_temporal_sentence_transformer":
            self.textualizer = TDOSTPlainTemporal()
            self.vectorizer = SentenceTransformerVectorizer(**kwargs)
        else:
            raise ValueError(f"Unknown extractor name: {self.name}")

    def fit(self, sensor_activations: SensorActivations, **kwargs):
        segments: list[Segment] | None = kwargs.get("segments", None)
        assert segments is not None, "Segments must be provided for fitting."
        descriptions = self._build_segment_descriptions(
            sensor_activations,
            segments,
            self.textualizer,
        )
        descriptions_flat: list[str] = []
        for descs in descriptions:
            descriptions_flat.extend(descs)
        self.vectorizer.fit(descriptions_flat)

    def transform(
        self,
        segments: list[Segment],
        sensor_activations: SensorActivations,
    ) -> tuple[ndarray, ndarray]:
        vectors, masks = self._build_textual_features(
            sensor_activations,
            segments,
            self.textualizer,
            self.vectorizer,
        )
        return vectors, masks

    def _build_segment_descriptions(
        self,
        sensor_activations: SensorActivations,
        segments: list[Segment],
        textualizer: TDOSTPlainBasic | TDOSTPlainTemporal,
    ) -> list[list[str]]:
        events = sensor_activations.to_event_stream()
        descriptions_by_segment: list[list[str]] = []
        for segment in segments:
            segment_events = segment.to_event_window(events)
            if segment_events:
                descriptions = textualizer.describe_window(segment_events)
            else:  # pragma: no cover  # EventWindow is always truthy
                descriptions = ["No events in segment."]
            descriptions_by_segment.append(descriptions)

        return descriptions_by_segment

    def _build_textual_features(
        self,
        sensor_activations: SensorActivations,
        segments: list[Segment],
        textualizer: TDOSTPlainBasic | TDOSTPlainTemporal,
        vectorizer: TfidfVectorizer
        | Word2VecVectorizer
        | SentenceTransformerVectorizer,
    ) -> tuple[ndarray, ndarray]:
        descriptions_flat: list[str] = []
        for descriptions in self._build_segment_descriptions(
            sensor_activations,
            segments,
            textualizer,
        ):
            descriptions_flat.extend(descriptions)
        vectors = vectorizer.transform(descriptions_flat)
        masks = array([])
        return vectors, masks
