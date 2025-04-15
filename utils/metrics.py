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

    def __init__(self, positive_label: int = 1):
        super().__init__(positive_label=positive_label)
        self.lm_loss = None
        self.z_loss = None
        self.aux_loss = None

    def __call__(self, eval_preds):
        # process eval_preds
        self.logits, self.labels = eval_preds
        if isinstance(output, tuple):
            # top k
            if len(self.logits) == 4:
                logits, z_loss, aux_loss, lm_loss = self.logits
                self.lm_loss = lm_loss.mean()
                self.z_loss = z_loss.mean()
                self.aux_loss = aux_loss.mean()
            # expert choice
            elif len(self.logits) == 3:
                logits, z_loss, lm_loss = self.logits
                self.lm_loss = lm_loss.mean()
                self.z_loss = z_loss.mean()
            else:
                raise Exception()

        # conver to tensors
        self.logits = torch.Tensor(self.logits)
        self.labels = torch.Tensor(self.labels)

        # preds & probs
        self.predictions = torch.argmax(logits, dim=-1)
        self.probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        # mask out padding
        mask = self.labels != -100
        self.labels = self.labels[mask]
        self.predictions = self.predictions[mask.numpy()]

        return self.compute_metrics()

    def compute_metrics(self):
        result = {
            "accuracy": self.accuracy(),
            "perplexity": self.perplexity(),
            "mlm_loss": self.lm_loss,
            "z_loss": self.z_loss,
            "aux_loss": self.aux_loss,
        }
        result = {
            key: value for key, value in result.items() if value is not None
        }  # drop None values
        return result

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
        self.probabilities = (
            torch.softmax(torch.from_numpy(self.logits), dim=1).detach().numpy()[:, -1]
        )
        self.predictions = np.argmax(self.logits, axis=1)

        # build outputs
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "auroc": self.auroc(),
            "aupr": self.aupr(),
            "f1": self.f1(),
            "mcc": self.mcc(),
        }
