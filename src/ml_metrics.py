#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sklearn.metrics as mr
from sklearn.utils.class_weight import compute_sample_weight


def auc_binary_scorer(y_true, y_pred):
    auc_binary_score = mr.roc_auc_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
    )
    return auc_binary_score


def recall_binary_scorer(y_true, y_pred):
    recall_binary_score = mr.recall_score(
        y_true,
        y_pred,
        average="binary",
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
    )
    return recall_binary_score


def false_positive_rate_scorer(y_true, y_pred):
    tn = y_pred[(y_pred == 0) & (y_true == 0)].shape[0]
    fp = y_pred[(y_pred == 1) & (y_true == 0)].shape[0]
    fpr = fp / (fp + tn)
    return -fpr


def threshold_auc_score(ground_truth, predictions, threshold=0.5):
    predicted = (predictions >= threshold).astype("int")
    auc = auc_binary_scorer(ground_truth, predicted)
    return auc


def threshold_recall_score(ground_truth, predictions, threshold=0.5):
    predicted = (predictions >= threshold).astype("int")
    recall = recall_binary_scorer(ground_truth, predicted)
    return recall


def threshold_fpr_score(ground_truth, predictions, threshold=0.5):
    predicted = (predictions >= threshold).astype("int")
    fpr = false_positive_rate_scorer(ground_truth, predicted)
    return fpr
