"""
Parameter ranges for classification methods.

Defines search spaces for Bayesian hyperparameter optimization.
"""

from typing import Any

PARAM_RANGES: dict[str, dict[str, dict[str, Any]]] = {
    # Scikit-learn classifiers - vector features only
    "DecisionTreeClassifier": {
        "max_depth": {"min": 2, "max": 30, "type": "int", "default": 10},
        "min_samples_split": {"min": 2, "max": 20, "type": "int", "default": 2},
        "min_samples_leaf": {"min": 1, "max": 10, "type": "int", "default": 1},
        "criterion": {
            "values": ["gini", "entropy"],
            "type": "categorical",
            "default": "gini",
        },
    },
    "RandomForestClassifier": {
        "n_estimators": {"min": 10, "max": 300, "type": "int", "default": 100},
        "max_depth": {"min": 3, "max": 30, "type": "int", "default": 10},
        "min_samples_split": {"min": 2, "max": 20, "type": "int", "default": 2},
        "min_samples_leaf": {"min": 1, "max": 10, "type": "int", "default": 1},
    },
    "SVC": {
        "C": {"min": 0.001, "max": 100.0, "type": "float", "default": 1.0},
        "kernel": {
            "values": ["linear", "rbf", "poly"],
            "type": "categorical",
            "default": "rbf",
        },
        "gamma": {"min": 0.0001, "max": 1.0, "type": "float", "default": "scale"},
    },
    "KNeighborsClassifier": {
        "n_neighbors": {"min": 1, "max": 50, "type": "int", "default": 5},
        "weights": {
            "values": ["uniform", "distance"],
            "type": "categorical",
            "default": "uniform",
        },
        "leaf_size": {"min": 10, "max": 50, "type": "int", "default": 30},
    },
    "LogisticRegression": {
        "C": {"min": 0.001, "max": 100.0, "type": "float", "default": 1.0},
        "penalty": {"values": ["l2"], "type": "categorical", "default": "l2"},
        "max_iter": {"min": 100, "max": 1000, "type": "int", "default": 100},
    },
    "GaussianNB": {
        "var_smoothing": {
            "min": 1e-10,
            "max": 1e-6,
            "type": "float",
            "default": 1e-9,
        },
    },
    # Deep learning classifier - matrix features only
    "DeepSequenceGridClassifier": {
        "backbone": {
            "values": ["rnn", "lstm", "bilstm", "cnn1d", "cnn2d"],
            "type": "categorical",
            "default": "lstm",
        },
        "hidden_size": {"min": 32, "max": 256, "type": "int", "default": 64},
        "num_layers": {"min": 1, "max": 4, "type": "int", "default": 2},
        "dropout": {"min": 0.0, "max": 0.5, "type": "float", "default": 0.2},
        "learning_rate": {"min": 1e-5, "max": 1e-2, "type": "float", "default": 1e-3},
        "weight_decay": {"min": 0.0, "max": 1e-3, "type": "float", "default": 0.0},
        "use_attention": {
            "values": [True, False],
            "type": "categorical",
            "default": False,
        },
    },
}


def get_default_ranges(method_name: str) -> dict[str, Any]:
    """Get parameter ranges for a given method."""
    return PARAM_RANGES.get(method_name, {})


def get_vector_feature_classifiers() -> list[str]:
    """Get classifiers that work with vector features (sklearn)."""
    return [
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "SVC",
        "KNeighborsClassifier",
        "LogisticRegression",
        "GaussianNB",
    ]


def get_matrix_feature_classifiers() -> list[str]:
    """Get classifiers that work with matrix features (deep learning)."""
    return ["DeepSequenceGridClassifier"]


def get_all_classifiers() -> list[str]:
    """Get all available classifiers."""
    return get_vector_feature_classifiers() + get_matrix_feature_classifiers()
