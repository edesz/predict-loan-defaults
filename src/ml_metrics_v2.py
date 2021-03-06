#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import sklearn.metrics as mr
from sklearn.utils.class_weight import compute_sample_weight


def get_eval_metrics(y_test, y_probs, split="test", threshold=0.5, beta=2):
    y_pred_test_selected_threshold = (y_probs >= threshold).astype("int")
    d = {
        f"{split}_recall_binary": recall_binary_scorer(
            y_test, y_pred_test_selected_threshold
        ),
        f"{split}_fpr": -false_positive_rate_scorer(
            y_test, y_pred_test_selected_threshold
        ),
        f"{split}_f2": f2_binary_scorer(
            y_test, y_pred_test_selected_threshold, beta
        ),
        f"{split}_pr_auc": pr_auc_score(y_test, y_probs),
        f"{split}_roc_auc": roc_auc_binary_scorer(
            y_test, y_pred_test_selected_threshold
        ),
    }
    df_scores = pd.DataFrame.from_dict(d, orient="index").T
    return [df_scores, y_pred_test_selected_threshold]


def roc_auc_binary_scorer(y_true, y_pred):
    roc_auc_binary_score = mr.roc_auc_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
    )
    return roc_auc_binary_score


def recall_binary_scorer(y_true, y_pred):
    recall_binary_score = mr.recall_score(
        y_true,
        y_pred,
        average="binary",
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
    )
    return recall_binary_score


def f2_binary_scorer(y_true, y_pred, beta=2):
    f2_binary_score = mr.fbeta_score(
        y_true,
        y_pred,
        beta=beta,
        average="binary",
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
    )
    return f2_binary_score


def false_positive_rate_scorer(y_true, y_pred):
    tn = y_pred[(y_pred == 0) & (y_true == 0)].shape[0]
    fp = y_pred[(y_pred == 1) & (y_true == 0)].shape[0]
    fpr = fp / (fp + tn)
    return -fpr


def threshold_roc_auc_score(ground_truth, predictions, threshold=0.5):
    predicted = (predictions >= threshold).astype("int")
    roc_auc = roc_auc_binary_scorer(ground_truth, predicted)
    return roc_auc


def threshold_recall_score(ground_truth, predictions, threshold=0.5):
    predicted = (predictions >= threshold).astype("int")
    recall = recall_binary_scorer(ground_truth, predicted)
    return recall


def threshold_f2_score(ground_truth, predictions, threshold=0.5, beta=2):
    predicted = (predictions >= threshold).astype("int")
    f2 = f2_binary_scorer(ground_truth, predicted, beta)
    return f2


def threshold_fpr_score(ground_truth, predictions, threshold=0.5):
    predicted = (predictions >= threshold).astype("int")
    fpr = false_positive_rate_scorer(ground_truth, predicted)
    return fpr


def pr_auc_score(ground_truth, y_probs):
    precision, recall, _ = mr.precision_recall_curve(ground_truth, y_probs)
    pr_auc_score = mr.auc(recall, precision)
    return pr_auc_score


def get_scores(y_test, y_probs, t=0.5, beta=2):
    prob_to_label = (y_probs >= t).astype("int")
    return [
        mr.roc_auc_score(y_test, prob_to_label),
        mr.f1_score(
            y_test,
            prob_to_label,
            sample_weight=compute_sample_weight(
                class_weight="balanced", y=y_test
            ),
        ),
        mr.precision_score(
            y_test,
            prob_to_label,
            sample_weight=compute_sample_weight(
                class_weight="balanced", y=y_test
            ),
            zero_division=0,  # default="warn", which is same as 0
        ),
        mr.recall_score(
            y_test,
            prob_to_label,
            sample_weight=compute_sample_weight(
                class_weight="balanced", y=y_test
            ),
        ),
        -1 * false_positive_rate_scorer(y_test, prob_to_label),
        mr.fbeta_score(
            y_test,
            prob_to_label,
            beta=beta,
            sample_weight=compute_sample_weight(
                class_weight="balanced", y=y_test
            ),
        ),
    ]


def get_cm(t, p, cm_labels):
    df_cm = (
        pd.DataFrame(
            mr.confusion_matrix(t, p, labels=cm_labels),
            index=cm_labels,
            columns=cm_labels,
        )
        .rename_axis("actual", axis=0)
        .rename_axis("predicted", axis=1)
    )
    # print(df_cm)
    return df_cm


def rowwise_check_confusion_matrix(tn, fp, fn, tp, cm_labels):
    d = {
        "FN": [1, {"t": 1, "p": 0}],
        "TP": [1, {"t": 1, "p": 1}],
        "TN": [1, {"t": 0, "p": 0}],
        "FP": [1, {"t": 0, "p": 1}],
    }
    # print(d)
    for k, v in d.items():
        expected = get_cm([v[1]["t"]], [v[1]["p"]], cm_labels).iloc[
            v[1]["t"], v[1]["p"]
        ]
        observed = v[0]
        # print(k, v[1]["t"], v[1]["p"], expected, observed)
        assert expected == observed
