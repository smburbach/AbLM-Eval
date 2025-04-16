import evaluate
import numpy as np

# import sklearn as skl
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import EvalPrediction

__all__ = [
    "ComputeMetricsForMaskedLM",
    # "ComputeMetricsForSequenceClassification",
]


class ComputeMetricsBase:
    """
    Base class for computing classification metrics.
    """

    def __init__(self, positive_label: int = 1):
        self.positive_label = positive_label
        self.labels = None
        self.predictions = None
        self.probabilities = None

    def accuracy(self) -> float:
        return accuracy_score(y_true=self.labels, y_pred=self.predictions)

    def precision(self) -> float:
        return precision_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )

    def recall(self) -> float:
        return recall_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )

    def f1(self) -> float:
        return f1_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )

    def auroc(self) -> float:
        return roc_auc_score(y_true=self.labels, y_score=self.probabilities)

    def aupr(self) -> float:
        return average_precision_score(
            y_true=self.labels,
            y_score=self.probabilities,
            pos_label=self.positive_label,
        )

    def mcc(self) -> float:
        return matthews_corrcoef(y_true=self.labels, y_pred=self.predictions)


class ComputeMetricsForMaskedLM(ComputeMetricsBase):

    def __init__(self, positive_label: int = 1, return_moe_losses=False):
        super().__init__(positive_label=positive_label)
        self.return_moe_losses = return_moe_losses
        self.lm_loss = None
        self.z_loss = None
        self.aux_loss = None

    def __call__(self, eval_preds: EvalPrediction):
        # process eval_preds
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            if self.return_moe_losses: # extract moe losses if present
                # top k
                if len(logits) == 4:
                    _, z_loss, aux_loss, lm_loss = logits
                    self.aux_loss = aux_loss.mean()
                # expert choice
                elif len(self.logits) == 3:
                    _, z_loss, lm_loss = logits
                
                self.lm_loss = lm_loss.mean()
                self.z_loss = z_loss.mean()
            
            logits = logits[0]  # logits is the first element of the tuple

        # get as tensors
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        predictions = torch.argmax(logits, dim=-1)

        # mask out padding
        mask = (labels != -100)
        self.logits = logits[mask]
        self.labels = labels[mask]
        self.predictions = predictions[mask]

        return self.compute_metrics()

    def compute_metrics(self):
        return {
            k: v for k, v in {
                "accuracy": self.accuracy(),
                "perplexity": self.perplexity(),
                "mlm_loss": self.lm_loss,
                "z_loss": self.z_loss,
                "aux_loss": self.aux_loss,
            }.items() if v is not None
        }

    def perplexity(self):
        logits_flat = self.logits.view(-1, self.logits.size(-1))
        labels_flat = self.labels.view(-1)
        ce_loss = F.cross_entropy(
            logits_flat, labels_flat, ignore_index=-100, reduction="mean"
        )
        perplexity = torch.exp(ce_loss)
        return perplexity.item()


class ComputeMetricsForSequenceClassification(ComputeMetricsBase):

    def __call__(self, eval_preds):
        # process eval_preds
        self.logits, self.labels = eval_preds
        if isinstance(self.logits, tuple):
            self.logits = self.logits[0]  # logits is the first element of the tuple

        # compute probabilities and predictions
        self.predictions = torch.argmax(self.logits, dim=-1)
        self.probabilities = torch.softmax(self.logits, dim=-1).detach().cpu().numpy()

        # compute metrics
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "auroc": self.auroc(),
            "aupr": self.aupr(),
            "f1": self.f1(),
            "mcc": self.mcc(),
        }
