"""Classification metrics. Macro F1 is the checkpoint-selection metric."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def _as_arrays(y_true: Iterable, y_pred: Iterable) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(list(y_true)), np.asarray(list(y_pred))


def compute_classification_metrics(
    y_true,
    y_pred,
    class_names: dict[int, str],
    y_score=None,
) -> dict:
    labels = list(range(len(class_names)))
    y_true_arr, y_pred_arr = _as_arrays(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": [],
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist(),
    }

    if y_score is not None:
        scores = np.asarray(y_score)
        k = min(5, scores.shape[1])
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(y_true_arr, scores, k=k, labels=labels)
        )
    else:
        metrics["top5_accuracy"] = None

    for class_index in labels:
        metrics["per_class"].append(
            {
                "class_index": class_index,
                "class_name": class_names[class_index],
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )
    return metrics
