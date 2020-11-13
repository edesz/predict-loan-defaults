#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count
from time import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import src.business_helpers as bh
import src.visualization_helpers as vh


def scoring_func(X, y, pipe, threshold=0.5):
    # score = pipe.score(X, y)
    score = bh.calculate_avg_return_vs_theoretical_v2(
        X, y, pipe, threshold
    ).mean()
    return score


def score_splits(X, y, train_idx, test_idx, pipe, threshold=0.5):
    # print(train_idx, test_idx)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    start_time = time()
    pipe.fit(X_train, y_train)
    fit_time = time() - start_time
    train_err = scoring_func(X_train, y_train, pipe, threshold)
    test_err = scoring_func(X_test, y_test, pipe, threshold)
    return {
        "train_size": len(X_train),
        "train_err": train_err,
        "test_err": test_err,
        "fit_time": fit_time,
        "train_idx": train_idx,
        "clf": type(pipe.named_steps["clf"]).__name__,
    }


def score_cv_folds(pipe, X, y, cv, threshold=0.5):
    cv_fold_scores = [
        score_splits(X, y, train_idx, test_idx, pipe, threshold)
        for train_idx, test_idx in cv.split(X=X)
    ]
    df_cv = pd.DataFrame.from_records(cv_fold_scores)
    return df_cv


def learning_curve(pipe, X, y, cv, train_size_blocks=11, threshold=0.5):
    train_sizes = [
        int(i) for i in np.linspace(0, X.shape[0], train_size_blocks)
    ][1:]
    # # Approx. example from docs
    # train_sizes = [
    #     int(i * len(X)) for i in np.linspace(0.1, 1.0, train_size_blocks)
    # ]
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(score_cv_folds)(pipe, X[:r], y[:r], cv, threshold)
        for r in train_sizes
    )
    scores = executor(tasks)
    return [pd.concat(scores), train_sizes]


def manual_learning_curve(
    df,
    alpha=0.2,
    hspace=0.2,
    wspace=0.2,
    axis_tick_label_fontsize=12,
    figsize=(6, 12),
):
    fig = plt.figure(
        figsize=(figsize[0] * df["clf|first"].nunique(), figsize[1])
    )
    grid = plt.GridSpec(
        3, df["clf|first"].nunique(), hspace=hspace, wspace=wspace
    )
    for c, clf in enumerate(df["clf|first"].unique()):
        data = df[df["clf|first"] == clf]
        ax1 = fig.add_subplot(grid[0, c])
        ax2 = fig.add_subplot(grid[1, c])
        ax3 = fig.add_subplot(grid[2, c])
        data.plot(
            x="train_size|",
            y=["train_err|mean", "test_err|mean"],
            marker="o",
            ax=ax1,
        )
        ax1.set_title(
            "Learning Curves - Score ($) vs Train size",
            loc="left",
            fontweight="bold",
        )
        ax1.legend(
            labels=["Training", "Cross-validation"],
            handletextpad=0.2,
            frameon=False,
        )
        ax1.fill_between(
            data["train_size|"],
            data["train_err|mean"] - data["train_err|std"],
            data["train_err|mean"] + data["train_err|std"],
            facecolor="steelblue",
            alpha=alpha,
        )
        ax1.fill_between(
            data["train_size|"],
            data["test_err|mean"] - data["test_err|std"],
            data["test_err|mean"] + data["test_err|std"],
            facecolor="orange",
            alpha=alpha,
        )
        ax1.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

        data.plot(
            x="train_size|",
            y=["fit_time|mean"],
            marker="o",
            ax=ax2,
            c="steelblue",
        )
        ax2.set_title(
            "Scalability of the model - Fit times (sec) vs Train size",
            loc="left",
            fontweight="bold",
        )
        ax2.legend().remove()
        ax2.fill_between(
            data["train_size|"],
            data["fit_time|mean"] - data["fit_time|std"],
            data["fit_time|mean"] + data["fit_time|std"],
            facecolor="steelblue",
            alpha=alpha,
        )
        ax2.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

        data.plot(
            x="fit_time|mean",
            y=["test_err|mean"],
            marker="o",
            ax=ax3,
            c="steelblue",
        )
        ax3.set_title(
            "Performance of the model - Test Score ($) vs Fit times (sec)",
            loc="left",
            fontweight="bold",
        )
        ax3.legend().remove()
        ax3.fill_between(
            data["fit_time|mean"],
            data["test_err|mean"] - data["test_err|std"],
            data["test_err|mean"] + data["test_err|std"],
            facecolor="steelblue",
            alpha=alpha,
        )
        ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel(None)
            ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
            ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
            ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
            _ = vh.customize_splines(ax)
            _ = vh.add_gridlines(ax)
