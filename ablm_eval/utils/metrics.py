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
    "ComputeMetricsForSequenceClassification",
]


class ComputeMetricsBase:
    """
    Base class for computing metrics.
    """

    def __init__(self):
        self.lm_loss = None
        self.z_loss = None
        self.aux_loss = None

    def _process_eval_preds(
        self, eval_preds: EvalPrediction, return_moe_losses: bool = False
    ) -> None:
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            if return_moe_losses:  # extract moe losses if present
                # top k
                if len(logits) == 4:
                    _, z_loss, aux_loss, lm_loss = logits
                    self.aux_loss = aux_loss.mean()
                # expert choice
                elif len(logits) == 3:
                    _, z_loss, lm_loss = logits

                self.lm_loss = lm_loss.mean()
                self.z_loss = z_loss.mean()

            logits = logits[0]  # logits is the first element of the tuple

        # convert to tensors
        self.logits = torch.tensor(logits)
        self.labels = torch.tensor(labels)

        # extract preds and probs
        self.predictions = torch.argmax(self.logits, dim=-1)
        self.probabilities = torch.softmax(self.logits, dim=-1).detach().cpu().numpy()

    def _filter_none(self, d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    def accuracy(self) -> float:
        return accuracy_score(y_true=self.labels, y_pred=self.predictions)


class ComputeMetricsForMaskedLM(ComputeMetricsBase):
    """
    Metrics class for MLM metrics.
    """

    def __init__(self, return_moe_losses: bool = False):
        super().__init__()
        self.return_moe_losses = return_moe_losses

    def __call__(self, eval_preds: EvalPrediction):
        # process preds
        self._process_eval_preds(eval_preds, return_moe_losses=self.return_moe_losses)

        # mask out padding
        mask = self.labels != -100
        self.logits = self.logits[mask]
        self.labels = self.labels[mask]
        self.predictions = self.predictions[mask]

        return self._filter_none(
            {
                "accuracy": self.accuracy(),
                "perplexity": self.perplexity(),
                "mlm_loss": self.lm_loss,
                "z_loss": self.z_loss,
                "aux_loss": self.aux_loss,
            }
        )

    def perplexity(self):
        ce_loss = F.cross_entropy(
            self.logits.view(-1, self.logits.size(-1)),
            self.labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        perplexity = torch.exp(ce_loss)
        return perplexity.item()


class ComputeMetricsForSequenceClassification(ComputeMetricsBase):
    """
    Metrics class for classification metrics.
    """

    def __init__(
        self,
        positive_label: int,
        num_classes: int,
        multi_class_average: str,
    ):
        super().__init__()
        self.positive_label = positive_label
        self.num_classes = num_classes
        self.multi_class_average = multi_class_average

    def __call__(self, eval_preds):
        # process preds
        self._process_eval_preds(eval_preds)

        # take probs for positive only, used in binary metrics
        self.probabilities = self.probabilities[:, 1]

        # compute metrics
        return self._filter_none(
            {
                "accuracy": self.accuracy(),
                "precision": self.precision(),
                "recall": self.recall(),
                "auroc": self.auroc() if self.num_classes == 2 else None,
                "aupr": self.aupr() if self.num_classes == 2 else None,
                "f1": self.f1(),
                "mcc": self.mcc(),
            }
        )

    def precision(self) -> float:
        if self.num_classes > 2:
            return precision_score(
                y_true=self.labels,
                y_pred=self.predictions,
                average=self.multi_class_average,
            )
        return precision_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )

    def recall(self) -> float:
        if self.num_classes > 2:
            return recall_score(
                y_true=self.labels,
                y_pred=self.predictions,
                average=self.multi_class_average,
            )
        return recall_score(
            y_true=self.labels, y_pred=self.predictions, pos_label=self.positive_label
        )

    def f1(self) -> float:
        if self.num_classes > 2:
            return f1_score(
                y_true=self.labels,
                y_pred=self.predictions,
                average=self.multi_class_average,
            )
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
