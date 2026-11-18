import numpy as np
from numpy import ndarray

from backend.models import Segment, SensorActivations

from .....methods import BaseRepresentationMethod
from ....textual.textualization import TDOSTSeqBasic, TDOSTSeqTemporal
from ....textual.vectorization import (
    SentenceTransformerVectorizer,
    TfidfVectorizer,
    Word2VecVectorizer,
)


class TextualMatrixExtractor(BaseRepresentationMethod):
    def __init__(
        self,
        max_len: int = 30,
        name: str = "tdost_seq_basic_tfidf",
        **kwargs,
    ):
        self.max_len = max_len
        self.name = name
        if self.name == "tdost_seq_basic_tfidf":
            self.textualizer: TDOSTSeqBasic | TDOSTSeqTemporal = TDOSTSeqBasic()
            self.vectorizer: (
                TfidfVectorizer | Word2VecVectorizer | SentenceTransformerVectorizer
            ) = TfidfVectorizer(**kwargs)
        elif self.name == "tdost_seq_basic_word2vec":
            self.textualizer = TDOSTSeqBasic()
            self.vectorizer = Word2VecVectorizer(**kwargs)
        elif self.name == "tdost_seq_basic_sentence_transformer":
            self.textualizer = TDOSTSeqBasic()
            self.vectorizer = SentenceTransformerVectorizer(**kwargs)
        elif self.name == "tdost_seq_temporal_tfidf":
            self.textualizer = TDOSTSeqTemporal()
            self.vectorizer = TfidfVectorizer(**kwargs)
        elif self.name == "tdost_seq_temporal_word2vec":
            self.textualizer = TDOSTSeqTemporal()
            self.vectorizer = Word2VecVectorizer(**kwargs)
        elif self.name == "tdost_seq_temporal_sentence_transformer":
            self.textualizer = TDOSTSeqTemporal()
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
        textualizer: TDOSTSeqBasic | TDOSTSeqTemporal,
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
        textualizer: TDOSTSeqBasic | TDOSTSeqTemporal,
        vectorizer: TfidfVectorizer
        | Word2VecVectorizer
        | SentenceTransformerVectorizer,
    ) -> tuple[ndarray, ndarray]:
        descriptions_flat: list[str] = []
        lengths: list[int] = []
        for descriptions in self._build_segment_descriptions(
            sensor_activations,
            segments,
            textualizer,
        ):
            lengths.append(len(descriptions))
            descriptions_flat.extend(descriptions)
        vectors = vectorizer.transform(descriptions_flat)

        def pad_truncate(vectors: ndarray, max_len: int) -> tuple[ndarray, ndarray]:
            if vectors.shape[0] > max_len:
                mask = np.zeros((max_len,), dtype=bool)
                mask[:] = True
                return vectors[:max_len], mask
            else:
                pad_width = max_len - vectors.shape[0]
                padded_vectors = np.pad(
                    vectors,
                    ((0, pad_width), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                mask = np.zeros((max_len,), dtype=bool)
                mask[: vectors.shape[0]] = True
                return padded_vectors, mask

        src_vectors = []
        idx = 0
        for length in lengths:
            src_vectors.append(vectors[idx : idx + length])
            idx += length
        padded_vectors = [pad_truncate(v, max_len=self.max_len) for v in src_vectors]
        final_vectors = np.array([v[0] for v in padded_vectors])
        final_masks = np.array([v[1] for v in padded_vectors])
        return final_vectors, final_masks
