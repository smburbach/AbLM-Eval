import numpy as np
import torch
from ablm_eval.utils.metrics import (
    ComputeMetricsForMaskedLM,
    ComputeMetricsForSequenceClassification,
)


# mlm metrics
# @pytest.fixture(scope="module")
# def setup_mlm_metrics():
#     pass


# accuracy
# perplexity
# moe losses returned


# binary classification metrics
# @pytest.fixture(scope="module")
# def setup_binary_metrics():
#     logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
#     labels = torch.tensor([0, 1])
#     eval_preds = (logits.numpy(), labels.numpy())
#     metrics = ComputeMetricsForSequenceClassification()
#     metrics._process_eval_preds(eval_preds)
#     return metrics


# def test_accuracy_binary(binary_metrics):
#     assert binary_metrics.accuracy() == 1.0
# precision
# recall
# auc
# aupr
# mcc

# 3-class classification metrics
# @pytest.fixture(scope="module")
# def setup_multiclass_metrics():
#     logits = torch.tensor(
#         [
#             [1.0, 2.0, 0.5],  # predicted 1
#             [2.0, 0.5, 1.0],  # predicted 0
#             [0.5, 1.0, 2.0],  # predicted 2
#         ]
#     )
#     labels = torch.tensor([1, 0, 2])
#     eval_preds = (logits.numpy(), labels.numpy())
#     metrics = ComputeMetricsForSequenceClassification()
#     metrics._process_eval_preds(eval_preds)
#     return metrics


# def test_accuracy_multiclass(multiclass_metrics):
#     assert multiclass_metrics.accuracy() == 1.0
# precision
# recall
# mcc