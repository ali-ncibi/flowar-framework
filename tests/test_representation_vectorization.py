import numpy as np
import pytest
from gensim.models import KeyedVectors

from backend.representation.methods.textual import vectorization
from backend.representation.methods.textual.vectorization import (
    SentenceTransformerVectorizer,
    SentenceVectorizer,
    SentenceVectorizerFactory,
    SentenceVectorizerType,
    TfidfVectorizer,
    Word2VecVectorizer,
    default_tokenizer,
)


class TestDefaultTokenizer:
    def test_tokenize(self):
        assert default_tokenizer("Hello World") == ["hello", "world"]
        assert default_tokenizer("A-B") == ["a", "b"]


class TestSentenceVectorizerBase:
    def test_abstract(self):
        with pytest.raises(TypeError):
            SentenceVectorizer()

    def test_transform_and_super(self):
        class Dummy(SentenceVectorizer):
            def fit(self, corpus=None):
                return super().fit(corpus)

            def encode_sentence(self, text):
                return super().encode_sentence(text)

        d = Dummy()
        assert d.fit(["a"]) is None  # covers ellipsis body of abstract fit
        assert (
            d.encode_sentence("x") is None
        )  # covers ellipsis body of abstract encode_sentence
        # transform uses encode_sentence
        d2 = Dummy()
        d2.encode_sentence = lambda text: np.array([1.0, 2.0])  # type: ignore[method-assign]
        out = d2.transform(["a", "b"])
        assert out.shape == (2, 2)

    def test_type_enum(self):
        assert SentenceVectorizerType.TFIDF.value == "tfidf"
        assert SentenceVectorizerType.SBERT.value == "sbert"
        assert SentenceVectorizerType.WORD2VEC.value == "word2vec"


class TestTfidfVectorizer:
    def test_fit_none_raises(self):
        with pytest.raises(ValueError, match="requires fit"):
            TfidfVectorizer().fit(None)

    def test_encode_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            TfidfVectorizer().encode_sentence("hello world")

    def test_fit_transform(self):
        v = TfidfVectorizer(max_features=100)
        assert v.fit(["hello world", "hello there world"]) is v
        out = v.transform(["hello world", "goodbye"])
        assert out.shape == (2, out.shape[1])
        assert out.dtype == np.float32

    def test_ngram_char(self):
        v = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=50)
        v.fit(["abc def", "abc"])
        out = v.transform(["abc"])
        assert out.shape[0] == 1


class TestWord2VecVectorizer:
    @pytest.fixture
    def kv(self):
        k = KeyedVectors(vector_size=8)
        rng = np.random.RandomState(0)
        k.add_vectors(["hello", "world", "foo"], rng.rand(3, 8).astype(np.float32))
        return k

    def _make(self, kv, **kwargs):
        v = Word2VecVectorizer(**kwargs)
        v._load_pretrained = lambda: kv  # type: ignore[method-assign]
        return v

    def test_ensure_fitted_raises(self, kv):
        v = self._make(kv)
        with pytest.raises(RuntimeError, match="not fitted"):
            v.encode_sentence("hello")

    def test_load_pretrained_calls_gensim(self, kv, monkeypatch):
        monkeypatch.setattr(vectorization.api, "load", lambda name: kv)
        v = Word2VecVectorizer(model_name="fake-model")
        assert v._load_pretrained() is kv

    def test_fit_and_encode_mean(self, kv):
        v = self._make(kv, aggregation="mean", use_tfidf=True)
        assert v.fit(["hello world", "hello foo"]) is v
        out = v.transform(["hello world", "unknown-token"])
        assert out.shape == (2, 8)
        assert out.dtype == np.float32

    def test_fit_no_tfidf_no_corpus(self, kv):
        v = self._make(kv, use_tfidf=False)
        v.fit()  # corpus None -> idf None -> weights 1.0
        out = v.encode_sentence("hello foo")
        assert out.shape == (8,)

    def test_encode_sum_aggregation(self, kv):
        v = self._make(kv, aggregation="sum")
        v.fit(["hello world"])
        out = v.encode_sentence("hello world")
        assert out.shape == (8,)

    def test_encode_empty_tokens(self, kv):
        v = self._make(kv)
        v.fit(["hello"])
        out = v.encode_sentence("!!!")  # tokenizer yields nothing
        assert out.shape == (8,)
        assert np.all(out == 0)

    def test_encode_empty_tokens_l2(self, kv):
        v = self._make(kv, l2_normalize=True)
        v.fit(["hello"])
        out = v.encode_sentence("!!!")
        assert np.all(out == 0)  # l2 of zero vector stays zero

    def test_encode_all_unknown(self, kv):
        v = self._make(kv)
        v.fit(["hello"])
        out = v.encode_sentence("zzz qqq")
        assert np.all(out == 0)

    def test_l2_normalize(self, kv):
        v = self._make(kv, l2_normalize=True)
        v.fit(["hello world"])
        out = v.encode_sentence("hello world")
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_idf_power(self, kv):
        v = self._make(kv, idf_power=2.0, tfidf_params={"max_features": 10})
        v.fit(["hello world", "hello foo bar"])
        assert v._idf is not None
        out = v.encode_sentence("hello world")
        assert out.shape == (8,)


class _FakeST:
    def __init__(self, name, device=None):
        self.name = name
        self.device = device
        self.max_seq_length = None

    def eval(self):
        return self

    def encode(
        self,
        text,
        convert_to_numpy=True,
        normalize_embeddings=False,
        batch_size=1,
        show_progress_bar=False,
        device=None,
    ):
        if isinstance(text, list):
            return np.array([[0.5] * 16 for _ in text], dtype=np.float32)
        vec = np.array([0.5] * 16, dtype=np.float32)
        if normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return vec


class TestSentenceTransformerVectorizer:
    def test_encode_not_fitted(self):
        v = SentenceTransformerVectorizer(model_name="fake")
        with pytest.raises(RuntimeError, match="not fitted"):
            v.encode_sentence("hi")

    def test_fit_encode(self, monkeypatch):
        monkeypatch.setattr(vectorization, "SentenceTransformer", _FakeST)
        v = SentenceTransformerVectorizer(
            model_name="fake", max_length=128, l2_normalize=True
        )
        assert v.fit(["a"]) is v
        assert v._model.max_seq_length == 128
        out = v.encode_sentence("hello")
        assert out.shape == (16,)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_fit_no_max_length(self, monkeypatch):
        monkeypatch.setattr(vectorization, "SentenceTransformer", _FakeST)
        v = SentenceTransformerVectorizer(model_name="fake")
        v.fit()
        assert v._model.max_seq_length is None
        assert v._dim == 16


class TestFactory:
    def test_create_tfidf(self):
        assert isinstance(SentenceVectorizerFactory.create("tfidf"), TfidfVectorizer)

    def test_create_word2vec(self):
        assert isinstance(
            SentenceVectorizerFactory.create("Word2Vec"), Word2VecVectorizer
        )

    def test_create_sbert(self):
        v = SentenceVectorizerFactory.create(" sbert ", model_name="fake")
        assert isinstance(v, SentenceTransformerVectorizer)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown vectorizer"):
            SentenceVectorizerFactory.create("nope")
