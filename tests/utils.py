from ablm_eval import (
    InferenceConfig,
    PerPositionConfig,
    MutationPredConfig,
    ClassificationConfig,
    RoutingConfig,
    NaturalnessConfig,
)

__all__ = ["mini_models"]

# load test models
mini_models = {
    "BALM-dense": "./test_models/mini-BALM-dense/",
    "BALM-MoE": "./test_models/mini-BALM-MoE/",
    "ESM": "./test_models/mini-ESM/",
}
