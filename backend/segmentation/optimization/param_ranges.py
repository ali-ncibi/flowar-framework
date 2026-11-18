def get_default_ranges(method_name: str) -> dict:
    return PARAM_RANGES.get(method_name, {})


PARAM_RANGES = {
    "SepCpd": {
        "local_context_size": {"min": 1, "max": 20, "type": "int", "default": 5},
        "number_of_view_samples": {"min": 1, "max": 50, "type": "int", "default": 10},
        "number_of_bases": {"min": 1, "max": 20, "type": "int", "default": 5},
        "kernel_width": {"min": 0.1, "max": 5.0, "type": "float", "default": 1.0},
        "lambda_reg": {"min": 0.001, "max": 1.0, "type": "float", "default": 0.01},
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.3},
    },
    "RulsifPeCpd": {
        "local_context_size": {"min": 1, "max": 20, "type": "int", "default": 5},
        "number_of_view_samples": {"min": 1, "max": 50, "type": "int", "default": 10},
        "number_of_bases": {"min": 1, "max": 20, "type": "int", "default": 5},
        "kernel_width": {"min": 0.1, "max": 5.0, "type": "float", "default": 1.0},
        "lambda_reg": {"min": 0.001, "max": 1.0, "type": "float", "default": 0.01},
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.3},
        "alpha_relative": {"min": 0.01, "max": 1.0, "type": "float", "default": 0.5},
    },
    "RulsifPlsbdCpd": {
        "local_context_size": {"min": 1, "max": 20, "type": "int", "default": 5},
        "number_of_view_samples": {"min": 1, "max": 50, "type": "int", "default": 10},
        "number_of_bases": {"min": 1, "max": 20, "type": "int", "default": 5},
        "kernel_width": {"min": 0.1, "max": 5.0, "type": "float", "default": 1.0},
        "lambda_reg": {"min": 0.001, "max": 1.0, "type": "float", "default": 0.01},
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.3},
        "alpha_relative": {"min": 0.01, "max": 1.0, "type": "float", "default": 0.5},
    },
    "CorrWan": {
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.5},
        "time_bin_size": {"min": 10, "max": 300, "type": "int", "default": 60},
    },
    "CorrXu": {
        "context_size": {"min": 2, "max": 30, "type": "int", "default": 10},
    },
    "W2vCsCPD": {
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.5},
        "embedding_w2v_dimension": {
            "min": 10,
            "max": 200,
            "type": "int",
            "default": 50,
        },
        "embedding_w2v_context_size": {
            "min": 2,
            "max": 10,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_training_epochs": {
            "min": 1,
            "max": 50,
            "type": "int",
            "default": 10,
        },
        "embedding_w2v_min_token_count": {
            "min": 1,
            "max": 10,
            "type": "int",
            "default": 1,
        },
        "embedding_w2v_negative_sampling": {
            "min": 1,
            "max": 25,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_skipgram_or_cbow": {
            "min": 0,
            "max": 1,
            "type": "int",
            "default": 1,
        },
    },
    "W2vCcsCPD": {
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.3},
        "context_size": {"min": 1, "max": 10, "type": "int", "default": 3},
        "embedding_w2v_dimension": {
            "min": 10,
            "max": 200,
            "type": "int",
            "default": 50,
        },
        "embedding_w2v_context_size": {
            "min": 2,
            "max": 10,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_training_epochs": {
            "min": 1,
            "max": 50,
            "type": "int",
            "default": 10,
        },
        "embedding_w2v_min_token_count": {
            "min": 1,
            "max": 10,
            "type": "int",
            "default": 1,
        },
        "embedding_w2v_negative_sampling": {
            "min": 1,
            "max": 25,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_skipgram_or_cbow": {
            "min": 0,
            "max": 1,
            "type": "int",
            "default": 1,
        },
    },
    "W2vCcsdCPD": {
        "context_size": {"min": 1, "max": 10, "type": "int", "default": 3},
        "delta": {"min": 0.01, "max": 1.0, "type": "float", "default": 0.1},
        "embedding_w2v_dimension": {
            "min": 10,
            "max": 200,
            "type": "int",
            "default": 50,
        },
        "embedding_w2v_context_size": {
            "min": 2,
            "max": 10,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_training_epochs": {
            "min": 1,
            "max": 50,
            "type": "int",
            "default": 10,
        },
        "embedding_w2v_min_token_count": {
            "min": 1,
            "max": 10,
            "type": "int",
            "default": 1,
        },
        "embedding_w2v_negative_sampling": {
            "min": 1,
            "max": 25,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_skipgram_or_cbow": {
            "min": 0,
            "max": 1,
            "type": "int",
            "default": 1,
        },
    },
    "EMB-CPD": {
        "embedding_w2v_dimension": {
            "min": 10,
            "max": 200,
            "type": "int",
            "default": 50,
        },
        "embedding_w2v_context_size": {
            "min": 2,
            "max": 10,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_training_epochs": {
            "min": 1,
            "max": 50,
            "type": "int",
            "default": 10,
        },
        "threshold": {"min": 0.0, "max": 1.0, "type": "float", "default": 0.5},
        "context_size": {"min": 1, "max": 10, "type": "int", "default": 3},
        "delta": {"min": 0.01, "max": 1.0, "type": "float", "default": 0.1},
        "embedding_w2v_min_token_count": {
            "min": 1,
            "max": 10,
            "type": "int",
            "default": 1,
        },
        "embedding_w2v_negative_sampling": {
            "min": 1,
            "max": 25,
            "type": "int",
            "default": 5,
        },
        "embedding_w2v_skipgram_or_cbow": {
            "min": 0,
            "max": 1,
            "type": "int",
            "default": 1,
        },
    },
    "FixedCount": {
        "nb_events": {"min": 1, "max": 100, "type": "int", "default": 10},
    },
    "FixedTime": {
        "length_in_seconds": {
            "min": 1.0,
            "max": 3600.0,
            "type": "float",
            "default": 60.0,
        },
    },
}
