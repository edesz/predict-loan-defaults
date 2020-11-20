#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.ml_metrics_v2 import get_scores


def plot_metric_based_threshold_tuning_plots(
    y_test,
    y_probs,
    thresholds,
    f2_beta=2,
    legend_position=(1.01, 1),
    show_best_t_by_f1=False,
    show_plot=False,
    fig_size=(8, 4),
):
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(get_scores)(y_test, y_probs, t, f2_beta) for t in thresholds
    )
    scores = executor(tasks)
    df_s = pd.DataFrame(scores)
    (
        roc_auc_scores,
        f1_scores,
        precision_scores,
        recall_scores,
        fpr_scores,
        f2_scores,
    ) = (
        df_s[0].to_list(),
        df_s[1].to_list(),
        df_s[2].to_list(),
        df_s[3].to_list(),
        df_s[4].to_list(),
        df_s[5].to_list(),
    )
    d_threshold_tuning_scores = {}
    metrics = [
        roc_auc_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        fpr_scores,
        f2_scores,
    ]
    metrics_list = ["ROC-AUC", "Precision", "Recall", "F1", "FPR", "F2"]
    df_all_threshold_tuning_scores = (
        pd.DataFrame(
            metrics,
            index=metrics_list,
        ).T
    ).assign(threshold=thresholds)
    if show_plot:
        _, ax = plt.subplots(figsize=fig_size)
        ax.axvline(
            x=0.5,
            ls="--",
            lw=1.25,
            c="k",
            label="Default (0.5)",
        )
    for scores, name in zip(
        metrics,
        metrics_list,
    ):
        d_threshold_tuning_scores[name] = [
            thresholds[np.argmax(scores)],
            scores[np.argmax(scores)],
        ]
        if show_plot:
            ax.plot(thresholds, scores, label=name)
            if show_best_t_by_f1:
                vline_name = (
                    f"t-best[F1] = {d_threshold_tuning_scores['F1'][0]:.3f}"
                )
                ax.axvline(
                    x=d_threshold_tuning_scores["F1"][0],
                    ls="--",
                    lw=1.25,
                    c="darkgreen",
                    label=vline_name,
                )
            ax.legend(
                loc="upper left",
                bbox_to_anchor=legend_position,
                handletextpad=0.2,
                frameon=False,
            )
            ax.set_title(
                "Scoring Metrics, as threshold is changed",
                loc="left",
                fontweight="bold",
            )
    df_threshold_tuning_scores = pd.DataFrame.from_dict(
        d_threshold_tuning_scores, orient="index"
    ).rename(columns={0: "best_threshold", 1: "score"})
    df_all_threshold_tuning_scores = (
        df_all_threshold_tuning_scores.set_index("threshold")
        .unstack()
        .reset_index()
        .rename(columns={"level_0": "metric", 0: "value"})
    )
    return [df_threshold_tuning_scores, df_all_threshold_tuning_scores]
