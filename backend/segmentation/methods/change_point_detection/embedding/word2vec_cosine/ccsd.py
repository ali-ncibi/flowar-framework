import numpy as np
from numpy.typing import NDArray

from ......models import (
    EventStream,
    Interval,
    IntervalType,
    Segment,
    SegmentationResult,
)
from .base import W2vCosineCPD


class W2vCcsdCPD(W2vCosineCPD):
    def __init__(
        self,
        delta: float = 0.1,
        context_size: int = 3,
        embedding_w2v_dimension: int = 50,
        embedding_w2v_context_size: int = 5,
        embedding_w2v_training_epochs: int = 10,
        embedding_w2v_min_token_count: int = 1,
        embedding_w2v_skipgram_or_cbow: int = 1,
        embedding_w2v_negative_sampling: int = 5,
        random_state: int = 42,
    ):
        super().__init__(
            method_name="W2vCcsdCPD (Bermejo et al., 2021)",
            embedding_w2v_dimension=embedding_w2v_dimension,
            embedding_w2v_context_size=embedding_w2v_context_size,
            embedding_w2v_training_epochs=embedding_w2v_training_epochs,
            embedding_w2v_min_token_count=embedding_w2v_min_token_count,
            embedding_w2v_skipgram_or_cbow=embedding_w2v_skipgram_or_cbow,
            embedding_w2v_negative_sampling=embedding_w2v_negative_sampling,
            random_state=random_state,
        )
        self.context_size = context_size
        self.delta = delta

    def _handle_ccsd_variant(self, embeddings: list[NDArray[np.float64]]) -> list[int]:
        """Detect boundaries using contextual cosine similarity with delta detection."""
        boundaries: list[int] = []
        if len(embeddings) > self.context_size + 1:
            prev_ccs = 0.0
            for j in range(self.context_size + 1, len(embeddings) - self.context_size):
                ccs = 0.0
                for ell in range(1, self.context_size + 1):
                    cs_forward = self._cosine_similarity(
                        embeddings[j], embeddings[j + ell]
                    )
                    cs_backward = self._cosine_similarity(
                        embeddings[j], embeddings[j - ell]
                    )
                    ccs += cs_forward + (1.0 - cs_backward)
                ccs /= 2.0 * self.context_size
                if ccs >= prev_ccs + self.delta:
                    boundaries.append(j)
                prev_ccs = ccs
        return boundaries

    def transform(self, events: EventStream) -> SegmentationResult:
        """Segment using the CCSD variant."""
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before segment()")
        actions = [self._event_to_action(e) for e in events.events]
        if self._action_embeddings is None or self._embedding_w2v_dim is None:
            raise RuntimeError("Embeddings are not available after calibration.")
        embeddings: list[NDArray[np.float64]] = []
        for action in actions:
            embeddings.append(
                self._action_embeddings.get(action, np.zeros(self._embedding_w2v_dim))
            )
        boundaries = self._handle_ccsd_variant(embeddings)
        boundaries_sorted = sorted(set(boundaries))
        cut_points = [0] + boundaries_sorted + [len(events.events) - 1]
        segments: list[Segment] = []
        for i in range(len(cut_points) - 1):
            start_idx = cut_points[i]
            end_idx = cut_points[i + 1]
            start_t = events.events[start_idx].timestamp
            end_t = events.events[end_idx].timestamp
            segments.append(
                Segment(
                    interval=Interval(start_t, end_t, IntervalType.LEFT_OPEN),
                    event_indices=(start_idx, end_idx),
                    metadata={"variant": "CCSD"},
                )
            )
        segments = [seg for seg in segments if seg.interval.duration > 0.0]
        return SegmentationResult(
            segments=segments,
            method_name=self.method_name,
            parameters={
                "delta": self.delta,
                "context_size": self.context_size,
            },
            metadata={"num_boundaries": len(boundaries_sorted)},
        )
