#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.ml_metrics_v2 import get_scores
from src.threshold_tuning_helpers import get_cost_func


def plot_cost_function_based_threshold_tuning_plots(
    best_pipe, best_pipe_dummy, X_val, y_val, confs, thresholds
):
    dfs = []
    dfps = []
    for pipe in [best_pipe_dummy, best_pipe]:
        df_cost_func = get_cost_func(confs, thresholds, X_val, y_val, pipe)
        d = []
        dts = []
        clf_name = type(pipe.named_steps["clf"]).__name__
        if "Dummy" not in clf_name and len(thresholds) > 1:
            fig = plt.figure(figsize=(8, 12))
            grid = plt.GridSpec(
                df_cost_func["config"].nunique(), 1, hspace=0.35
            )
        for k in df_cost_func["config"].unique().tolist():
            dfp = (
                df_cost_func.set_index(["fp", "fn", "tp", "tn"])
                .pivot(
                    index=["threshold"], columns="config", values="cost_func"
                )
                .reset_index()
            )
            best_t = dfp.set_index("threshold")[k].idxmax()
            return_at_best_t = dfp[dfp["threshold"] == best_t][k].iloc[0]
            num_months = int(re.findall("[0-9]+", confs[k]["n"])[0])
            model_str = r"model$_\mathregular{AVG}$"
            config_record = {
                "r": float(confs[k]["r"]),
                "n": num_months,
                "p": float(confs[k]["p"]),
                "config": k,
                "best_t": best_t,
                "return_at_best_t": return_at_best_t,
                "theoretical": float(confs[k]["ret"]),
                "clf": clf_name,
            }
            for cfk, cfv in config_record.items():
                dfp[cfk] = cfv
            d.append(config_record)
            dts.append(dfp)

            if "Dummy" not in clf_name and len(thresholds) > 1:
                ax = fig.add_subplot(grid[k, 0])
                dfp.plot(x="threshold", y=k, kind="line", ax=ax)
                ax.axvline(x=0.5, ls="--", lw=1.25, c="k")
                ax.fill_between(
                    x=dfp["threshold"],
                    y1=dfp[k],
                    y2=[return_at_best_t] * len(dfp[k]),
                    where=(return_at_best_t > dfp[k])
                    & (dfp["threshold"] >= 0.5)
                    & (dfp["threshold"] <= best_t),
                    facecolor="C0",
                    alpha=0.25,
                )
                ax.get_legend().remove()
                title = (
                    f"r = {float(confs[k]['r']):.2f}%, "
                    f"n = {num_months} months, "
                    f"p = ${float(confs[k]['p']):,.2f}"
                )
                annot = (
                    f"theoretical = {float(confs[k]['ret']):,.2f}\n"
                    f"{model_str} = {return_at_best_t:,.2f}"
                )
                bbox_props = dict(
                    boxstyle="round",
                    fc=(0.8, 0.9, 0.9),
                    ec="b",
                    alpha=0.5,
                    lw=1.25,
                )
                _ = ax.annotate(
                    xy=(0.98, 0.1),
                    xycoords=ax.transAxes,
                    text=annot,
                    ha="right",
                    va="bottom",
                    rotation=0,
                    size=15,
                    bbox=bbox_props,
                )
                ax.set_title(
                    title,
                    loc="left",
                    fontweight="bold",
                )
                ax.set_xlabel(None)
        df_t_tuned = pd.DataFrame.from_records(d)
        dfs.append(df_t_tuned)
        dfps.append(pd.concat(dts))
    return [
        pd.concat(dfs).reset_index(drop=True),
        pd.concat(dts).reset_index(drop=True),
    ]


def plot_metric_based_threshold_tuning_plots(
    y_test,
    y_probs,
    thresholds,
    legend_position=(1.01, 1),
    show_best_t_by_f1=False,
    show_plot=False,
    fig_size=(8, 4),
):
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (delayed(get_scores)(y_test, y_probs, t) for t in thresholds)
    scores = executor(tasks)
    df_s = pd.DataFrame(scores)
    roc_auc_scores, f1_scores, precision_scores, recall_scores, fpr_scores = (
        df_s[0].to_list(),
        df_s[1].to_list(),
        df_s[2].to_list(),
        df_s[3].to_list(),
        df_s[4].to_list(),
    )
    d_threshold_tuning_scores = {}
    df_all_threshold_tuning_scores = (
        pd.DataFrame(
            [
                roc_auc_scores,
                precision_scores,
                recall_scores,
                f1_scores,
                fpr_scores,
            ],
            index=["ROC-AUC", "Precision", "Recall", "F1", "FPR"],
        ).T
    ).assign(threshold=thresholds)
    if show_plot:
        _, ax = plt.subplots(figsize=fig_size)
    for scores, name in zip(
        [
            roc_auc_scores,
            precision_scores,
            recall_scores,
            f1_scores,
            fpr_scores,
        ],
        ["ROC-AUC", "Precision", "Recall", "F1", "FPR"],
    ):
        d_threshold_tuning_scores[name] = [
            thresholds[np.argmax(scores)],
            scores[np.argmax(scores)],
        ]
        if show_plot:
            ax.plot(thresholds, scores, label=name)
            ax.axvline(
                x=0.5,
                ls="--",
                lw=1.25,
                c="k",
                label="Default (0.5)",
            )
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
