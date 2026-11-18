from typing import Any

import numpy as np
from gensim.models import Word2Vec
from numpy.typing import NDArray

from ......models import Event, EventStream, SegmentationResult
from .....methods.base import BaseSegmentationMethod


class W2vCosineCPD(BaseSegmentationMethod):
    def __init__(
        self,
        method_name: str = "W2vCosineCPD",
        embedding_w2v_dimension: int = 50,
        embedding_w2v_context_size: int = 5,
        embedding_w2v_training_epochs: int = 10,
        embedding_w2v_min_token_count: int = 1,
        embedding_w2v_skipgram_or_cbow: int = 1,
        embedding_w2v_negative_sampling: int = 5,
        random_state: int = 42,
    ):
        super().__init__(method_name)
        self.embedding_w2v_dimension = embedding_w2v_dimension
        self.embedding_w2v_context_size = embedding_w2v_context_size
        self.embedding_w2v_training_epochs = embedding_w2v_training_epochs
        self.embedding_w2v_min_token_count = embedding_w2v_min_token_count
        self.embedding_w2v_skipgram_or_cbow = embedding_w2v_skipgram_or_cbow
        self.embedding_w2v_negative_sampling = embedding_w2v_negative_sampling
        self.random_state = random_state
        self._action_embeddings: dict[str, NDArray[np.float64]] | None = None
        self._embedding_w2v_dim: int | None = None
        self._action_alphabet: set[str] | None = None

    def _event_to_action(self, event: Event) -> str:
        return f"{event.sensor.name}|{'ON' if event.value else 'OFF'}"

    def _cosine_similarity(
        self, v1: NDArray[np.float64], v2: NDArray[np.float64]
    ) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _train_word2vec(
        self,
        corpus: list[list[str]],
    ) -> Word2Vec:
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.embedding_w2v_dimension,
            window=self.embedding_w2v_context_size,
            min_count=self.embedding_w2v_min_token_count,
            sg=self.embedding_w2v_skipgram_or_cbow,
            epochs=self.embedding_w2v_training_epochs,
            negative=self.embedding_w2v_negative_sampling,
            workers=1,
            seed=self.random_state,
        )
        return model

    def _extract_embeddings(
        self,
        model: Word2Vec,
        action_alphabet: set[str],
    ) -> dict[str, NDArray[np.float64]]:
        action_embeddings: dict[str, NDArray[np.float64]] = {}
        for action in action_alphabet:
            if action in model.wv:
                action_embeddings[action] = model.wv[action].astype(
                    np.float64, copy=True
                )
            else:
                action_embeddings[action] = (
                    np.zeros(self.embedding_w2v_dimension)
                ).astype(np.float64)
        for action, vec in action_embeddings.items():
            norm = np.linalg.norm(vec)
            if norm > 0:
                action_embeddings[action] = vec / norm
        return action_embeddings

    def fit(
        self,
        events: EventStream,
        **kwargs: Any,
    ) -> "W2vCosineCPD":
        actions = [self._event_to_action(e) for e in events.events]
        action_alphabet = set(actions)
        model = self._train_word2vec([actions])
        action_embeddings = self._extract_embeddings(model, action_alphabet)
        self._action_embeddings = action_embeddings
        self._embedding_w2v_dim = self.embedding_w2v_dimension
        self._action_alphabet = action_alphabet
        self._is_calibrated = True
        return self

    def transform(self, events: EventStream) -> SegmentationResult:
        raise NotImplementedError("Subclasses must implement segment()")
