import numpy as np
import pytest
import torch

import backend.classification.methods.deep.adapter as adapter_module
from backend.classification.methods.deep.adapter import DeepSequenceGridClassifier
from backend.classification.methods.deep.base import SequenceGridClassifierModule
from backend.classification.methods.deep.layers import (
    AttentionPool1D,
    AttentionPool2D,
    CNN1DBackbone,
    CNN2DBackbone,
    RNNBackbone,
)


@pytest.fixture(autouse=True)
def cpu_trainer(monkeypatch):
    """Force Lightning Trainer onto CPU and avoid spawning dataloader workers."""
    real_trainer = adapter_module.Trainer

    def factory(*args, **kwargs):
        kwargs["accelerator"] = "cpu"
        kwargs["devices"] = 1
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "Trainer", factory)
    monkeypatch.setattr(adapter_module, "cpu_count", lambda: 0)
    return factory


def make_module(backbone="lstm", task="multiclass", use_attention=False, **kw):
    defaults = dict(
        backbone=backbone,
        input_size=3,
        hidden_size=8,
        num_layers=1,
        num_outputs=2,
        task=task,
        use_attention=use_attention,
        use_masking=True,
    )
    defaults.update(kw)
    return SequenceGridClassifierModule(**defaults)


class TestLayers:
    def test_attention_pool_1d_no_mask(self):
        attn = AttentionPool1D(4, dropout=0.1)
        x = torch.randn(2, 5, 4)
        pooled, weights = attn(x)
        assert pooled.shape == (2, 4)
        assert weights.shape == (2, 5)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_attention_pool_1d_with_mask(self):
        attn = AttentionPool1D(4)
        x = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5)
        mask[1, 3:] = 0.0
        pooled, weights = attn(x, mask=mask)
        assert pooled.shape == (2, 4)
        assert weights[1, 3:].abs().max() < 1e-6

    def test_attention_pool_2d(self):
        attn = AttentionPool2D(8, dropout=0.1)
        x = torch.randn(2, 8, 3, 4)
        pooled, weights = attn(x)
        assert pooled.shape == (2, 8)
        assert weights.shape == (2, 1, 3, 4)

    @pytest.mark.parametrize("kind", ["rnn", "lstm"])
    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_rnn_backbone_no_mask(self, kind, bidirectional):
        bb = RNNBackbone(
            kind,
            input_size=3,
            hidden_size=8,
            num_layers=2,
            dropout=0.1,
            bidirectional=bidirectional,
        )
        expected = 8 * (2 if bidirectional else 1)
        assert bb.output_size == expected
        x = torch.randn(2, 6, 3)
        out, last = bb(x)
        assert out.shape == (2, 6, expected)
        assert last.shape == (2, expected)

    def test_rnn_backbone_with_masking(self):
        bb = RNNBackbone(
            "lstm",
            input_size=3,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
        )
        x = torch.randn(3, 6, 3)
        mask = torch.ones(3, 6)
        mask[0, 2:] = 0.0
        mask[1, 4:] = 0.0
        mask[2, :] = 0.0  # all-zero row exercises the clamp(min=1)
        out, last = bb(x, mask=mask, use_masking=True)
        assert out.shape[0] == 3
        assert last.shape == (3, 8)

    def test_rnn_backbone_mask_but_masking_disabled(self):
        bb = RNNBackbone(
            "rnn",
            input_size=3,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
        )
        x = torch.randn(2, 6, 3)
        mask = torch.ones(2, 6)
        out, last = bb(x, mask=mask, use_masking=False)
        assert last.shape == (2, 16)

    def test_cnn1d_backbone(self):
        bb = CNN1DBackbone(
            in_channels=3, channels=8, num_blocks=2, kernel_size=3, dropout=0.1
        )
        x = torch.randn(2, 10, 3)
        seq, pooled = bb(x)
        assert seq.shape == (2, 10, 8)
        assert pooled.shape == (2, 8)
        assert bb.out_channels == 8

    @pytest.mark.parametrize("pool", [True, False])
    def test_cnn2d_backbone(self, pool):
        bb = CNN2DBackbone(
            in_channels=1,
            channels=8,
            num_blocks=2,
            kernel_size=3,
            dropout=0.1,
            pool=pool,
        )
        x = torch.randn(2, 1, 8, 6)
        feat, pooled = bb(x)
        assert feat.shape[1] == 8
        assert pooled.shape == (2, 8)


class TestModule:
    @pytest.mark.parametrize("backbone", ["rnn", "lstm", "bilstm"])
    @pytest.mark.parametrize("use_attention", [False, True])
    def test_rnn_family_forward(self, backbone, use_attention):
        m = make_module(
            backbone=backbone, use_attention=use_attention, bidirectional=True
        )
        x = torch.randn(2, 6, 3)
        mask = torch.ones(2, 6)
        mask[1, 4:] = 0.0
        logits = m(x, mask)
        assert logits.shape == (2, 2)

    def test_rnn_forward_without_attention_ignores_mask_for_pooling(self):
        m = make_module(backbone="lstm", use_attention=False)
        x = torch.randn(2, 6, 3)
        logits = m(x, torch.ones(2, 6))
        assert logits.shape == (2, 2)

    @pytest.mark.parametrize("use_attention", [False, True])
    def test_cnn1d_forward(self, use_attention):
        m = make_module(
            backbone="cnn1d",
            use_attention=use_attention,
            cnn_channels=8,
            cnn_blocks=1,
            cnn_kernel_size=3,
        )
        x = torch.randn(2, 6, 3)
        logits = m(x, torch.ones(2, 6))
        assert logits.shape == (2, 2)

    def test_cnn2d_forward_3d_input_gets_unsqueezed(self):
        m = make_module(backbone="cnn2d", cnn_channels=8, cnn_blocks=1)
        x = torch.randn(2, 8, 6)  # 3D -> unsqueeze(1)
        logits = m(x)
        assert logits.shape == (2, 2)

    def test_cnn2d_forward_4d_input(self):
        m = make_module(backbone="cnn2d", cnn_channels=8, cnn_blocks=1)
        x = torch.randn(2, 1, 8, 6)
        logits = m(x)
        assert logits.shape == (2, 2)

    def test_cnn2d_attention(self):
        m = make_module(
            backbone="cnn2d", use_attention=True, cnn_channels=8, cnn_blocks=1
        )
        x = torch.randn(2, 8, 6)
        logits = m(x)
        assert logits.shape == (2, 2)

    def test_cnn2d_attention_module_deleted(self):
        m = make_module(
            backbone="cnn2d", use_attention=True, cnn_channels=8, cnn_blocks=1
        )
        del m.attn2d
        x = torch.randn(2, 8, 6)
        logits = m(x)
        assert logits.shape == (2, 2)

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            make_module(backbone="transformer")

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            make_module(task="binary")

    def test_forward_invalid_backbone_config_raises(self):
        m = make_module(backbone="lstm")
        m.hparams["backbone"] = "bogus"
        with pytest.raises(RuntimeError, match="Invalid backbone configuration"):
            m(torch.randn(2, 6, 3))

    @pytest.mark.parametrize("solver", ["adam", "adamw", "sgd"])
    def test_configure_optimizers(self, solver):
        m = make_module(solver=solver)
        opt = m.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)

    def test_configure_optimizers_invalid(self):
        m = make_module(solver="rmsprop")
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            m.configure_optimizers()

    def test_multilabel_criterion(self):
        m = make_module(task="multilabel", num_outputs=3, pos_weight=torch.ones(3))
        assert isinstance(m.criterion, torch.nn.BCEWithLogitsLoss)

    def test_multiclass_criterion_with_weights(self):
        m = make_module(class_weights=torch.tensor([1.0, 2.0]))
        assert isinstance(m.criterion, torch.nn.CrossEntropyLoss)


def tiny_classification_data(n=16, p=6, q=2, classes=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p, q).astype(np.float32)
    y = rng.randint(0, classes, size=n)
    return X, y


def tiny_multilabel_data(n=16, p=6, q=2, k=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p, q).astype(np.float32)
    y = rng.randint(0, 2, size=(n, k)).astype(np.float32)
    return X, y


def make_clf(**overrides):
    params = dict(
        backbone="lstm",
        hidden_size=8,
        num_layers=1,
        max_epochs=1,
        batch_size=8,
        validation_fraction=0.25,
        dropout=0.0,
        random_state=0,
    )
    params.update(overrides)
    return DeepSequenceGridClassifier(**params)


class TestAdapterValidation:
    @pytest.mark.parametrize(
        "attr,value,match",
        [
            ("task", "binary", "task must be"),
            ("backbone", "xxx", "Invalid backbone"),
            ("hidden_size", 0, "positive"),
            ("num_layers", 0, "positive"),
            ("dropout", 1.5, "dropout must be"),
            ("validation_fraction", 2.0, "validation_fraction must be"),
            ("batch_size", 0, "batch_size and max_epochs"),
            ("max_epochs", 0, "batch_size and max_epochs"),
            ("n_iter_no_change", 0, "n_iter_no_change must be"),
        ],
    )
    def test_validate_params_errors(self, attr, value, match):
        clf = make_clf(**{attr: value})
        with pytest.raises(ValueError, match=match):
            clf._validate_params()

    def test_validate_multilabel_threshold(self):
        clf = make_clf(task="multilabel", threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be"):
            clf._validate_params()

    def test_validate_params_ok(self):
        make_clf()._validate_params()

    def test_setup_random_state_none(self):
        clf = make_clf(random_state=None)
        clf._setup_random_state()  # should be a no-op

    def test_class_weights_none(self):
        clf = make_clf()
        clf.n_classes_ = 2
        assert clf._compute_class_weights_multiclass(np.array([0, 1])) is None

    def test_class_weights_balanced(self):
        clf = make_clf(class_weight="balanced")
        clf.n_classes_ = 2
        w = clf._compute_class_weights_multiclass(np.array([0, 0, 1]))
        assert isinstance(w, torch.Tensor)
        assert w.shape == (2,)

    def test_class_weights_dict(self):
        clf = make_clf(class_weight={0: 3.0})
        clf.n_classes_ = 2
        w = clf._compute_class_weights_multiclass(np.array([0, 1]))
        assert w[0].item() == 3.0
        assert w[1].item() == 1.0

    def test_class_weights_invalid(self):
        clf = make_clf(class_weight="weird")
        clf.n_classes_ = 2
        with pytest.raises(ValueError, match="class_weight must be"):
            clf._compute_class_weights_multiclass(np.array([0, 1]))

    def test_pos_weight_none(self):
        clf = make_clf()
        assert clf._compute_pos_weight_multilabel(np.ones((4, 2))) is None

    def test_pos_weight_balanced(self):
        clf = make_clf(pos_weight="balanced")
        w = clf._compute_pos_weight_multilabel(np.array([[1, 0], [0, 0], [1, 0]]))
        assert isinstance(w, torch.Tensor)
        assert w.shape == (2,)

    def test_pos_weight_array(self):
        clf = make_clf(pos_weight=np.array([1.0, 2.0]))
        w = clf._compute_pos_weight_multilabel(np.ones((4, 2)))
        assert w[1].item() == 2.0

    def test_pos_weight_array_wrong_ndim(self):
        clf = make_clf(pos_weight=np.ones((2, 2)))
        with pytest.raises(ValueError, match="shape"):
            clf._compute_pos_weight_multilabel(np.ones((4, 2)))

    def test_pos_weight_invalid(self):
        clf = make_clf(pos_weight="weird")
        with pytest.raises(ValueError, match="pos_weight must be"):
            clf._compute_pos_weight_multilabel(np.ones((4, 2)))


class TestAdapterFitPredict:
    def test_fit_predict_multiclass_with_masks(self):
        X, y = tiny_classification_data()
        masks = (np.arange(16 * 6).reshape(16, 6) % 5 < 4).astype(np.float32)
        clf = make_clf(class_weight="balanced")
        result = clf.fit(X, y, masks=masks)
        assert result is clf
        assert clf.n_classes_ == 2
        assert clf.input_size_ == 2
        preds = clf.predict(X)
        assert preds.shape == (16,)
        probs = clf.predict_proba(X)
        assert probs.shape == (16, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        preds_masked = clf.predict(X, masks=masks)
        assert preds_masked.shape == (16,)
        probs_masked = clf.predict_proba(X, masks=masks)
        assert probs_masked.shape == (16, 2)
        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_fit_multiclass_2d_X_and_no_val_split_no_early_stopping(self):
        rng = np.random.RandomState(1)
        X = rng.randn(16, 6).astype(np.float32)  # 2D -> reshaped to (16, 6, 1)
        y = rng.randint(0, 2, size=16)
        clf = make_clf(
            backbone="cnn1d",
            cnn_channels=4,
            cnn_blocks=1,
            class_weight={0: 2.0},
            validation_fraction=0.0,
            early_stopping=False,
        )
        clf.fit(X, y)
        assert clf.input_size_ == 1
        preds = clf.predict(X)  # 2D predict path
        assert preds.shape == (16,)
        assert clf.score(X, y) >= 0.0

    def test_fit_predict_multilabel_verbose(self):
        X, y = tiny_multilabel_data()
        clf = make_clf(
            task="multilabel",
            backbone="bilstm",
            pos_weight="balanced",
            threshold=0.5,
            verbose=True,
            validation_fraction=0.0,
        )
        clf.fit(X, y)
        assert clf.n_labels_ == 3
        preds = clf.predict(X)
        assert preds.shape == (16, 3)
        assert set(np.unique(preds)) <= {0, 1}
        probs = clf.predict_proba(X)
        assert probs.shape == (16, 3)
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_fit_multilabel_y_not_2d_raises(self):
        X, _ = tiny_multilabel_data()
        clf = make_clf(task="multilabel")
        # 3D y passes check_array(allow_nd=True) but is rejected by the ndim check
        with pytest.raises(ValueError, match="2D array"):
            clf.fit(X, np.zeros((16, 3, 1), dtype=np.float32))

    def test_fit_X_wrong_ndim_raises(self):
        X, y = tiny_classification_data()
        clf = make_clf()
        with pytest.raises(ValueError, match="2D .* or 3D"):
            clf.fit(X.reshape(16, 6, 2, 1), y)

    def test_fit_masks_shape_mismatch_raises(self):
        X, y = tiny_classification_data()
        clf = make_clf()
        with pytest.raises(ValueError, match="masks shape"):
            clf.fit(X, y, masks=np.ones((16, 5), dtype=np.float32))

    def test_predict_before_fit_raises(self):
        clf = make_clf()
        X, _ = tiny_classification_data()
        with pytest.raises(ValueError, match="fitted"):
            clf.predict(X)

    def test_predict_X_wrong_ndim_raises(self):
        X, y = tiny_classification_data()
        clf = make_clf()
        clf.fit(X, y)
        with pytest.raises(ValueError, match="2D or 3D"):
            clf._predict_logits(X.reshape(16, 6, 2, 1))

    def test_predict_wrong_feature_count_raises(self):
        X, y = tiny_classification_data()
        clf = make_clf()
        clf.fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.predict(np.zeros((4, 6, 5), dtype=np.float32))

    def test_predict_masks_shape_mismatch_raises(self):
        X, y = tiny_classification_data()
        clf = make_clf()
        clf.fit(X, y)
        with pytest.raises(ValueError, match="masks shape"):
            clf.predict(X, masks=np.ones((16, 3), dtype=np.float32))

    def test_data_loaders_with_masks_and_val_split(self):
        X, y = tiny_classification_data()
        masks = np.ones((16, 6), dtype=np.float32)
        clf = make_clf()
        train_loader, val_loader = clf._create_data_loaders(X, y, masks)
        assert val_loader is not None
        batch = next(iter(train_loader))
        assert len(batch) == 3

    def test_data_loaders_no_masks_val_split(self):
        X, y = tiny_classification_data()
        clf = make_clf()
        train_loader, val_loader = clf._create_data_loaders(X, y, None)
        assert val_loader is not None
        batch = next(iter(train_loader))
        assert len(batch) == 2

    def test_data_loaders_no_val_with_masks(self):
        X, y = tiny_classification_data()
        masks = np.ones((16, 6), dtype=np.float32)
        clf = make_clf(validation_fraction=0.0)
        train_loader, val_loader = clf._create_data_loaders(X, y, masks)
        assert val_loader is None
        assert len(next(iter(train_loader))) == 3

    def test_data_loaders_no_val_no_masks(self):
        X, y = tiny_classification_data()
        clf = make_clf(validation_fraction=0.0)
        train_loader, val_loader = clf._create_data_loaders(X, y, None)
        assert val_loader is None
        assert len(next(iter(train_loader))) == 2

    def test_data_loaders_val_split_without_random_state(self):
        X, y = tiny_classification_data()
        clf = make_clf(random_state=None)
        train_loader, val_loader = clf._create_data_loaders(X, y, None)
        assert val_loader is not None
        assert len(next(iter(train_loader))) == 2

    def test_test_step_via_trainer(self):
        X, y = tiny_classification_data()
        clf = make_clf(validation_fraction=0.0)
        clf.fit(X, y)
        train_loader, _ = clf._create_data_loaders(X, y, None)
        results = clf.trainer_.test(clf.model_, dataloaders=train_loader, verbose=False)
        assert isinstance(results, list)


class TestAdapterSklearnInterface:
    def test_get_params(self):
        clf = make_clf()
        params = clf.get_params()
        assert params["backbone"] == "lstm"
        assert params["max_epochs"] == 1
        assert set(params) == {
            "backbone",
            "use_attention",
            "task",
            "threshold",
            "hidden_size",
            "num_layers",
            "bidirectional",
            "cnn_channels",
            "cnn_kernel_size",
            "cnn_blocks",
            "cnn2d_pool",
            "dropout",
            "learning_rate_init",
            "weight_decay",
            "solver",
            "max_epochs",
            "batch_size",
            "validation_fraction",
            "early_stopping",
            "n_iter_no_change",
            "class_weight",
            "pos_weight",
            "random_state",
            "verbose",
            "use_masking",
        }

    def test_set_params(self):
        clf = make_clf()
        result = clf.set_params(hidden_size=16, solver="sgd")
        assert result is clf
        assert clf.hidden_size == 16
        assert clf.solver == "sgd"

    def test_set_params_invalid(self):
        clf = make_clf()
        with pytest.raises(ValueError, match="Invalid parameter"):
            clf.set_params(nonexistent_param=1)
