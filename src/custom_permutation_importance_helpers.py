#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

import src.business_helpers as bh
from src.visualization_helpers import customize_splines


def calculate_permutation_scores(
    X,
    ci,
    col,
    pipe,
    y,
    base_score,
    n_repeats,
    best_t,
    shuffling_idx=None,
    rng=None,
    verbose=False,
):
    # Make copy, else non-writeable DataFrame will be used and
    # 'replace with re-indexed shuffled values' step (i.e. in-place
    #  shuffling) of the rows is not possible as explained here:
    # https://github.com/numpy/numpy/issues/14972
    X = X.copy()
    scores = []
    for repeat_index in range(n_repeats):
        # shuffle index
        rng.shuffle(shuffling_idx)
        # extract shuffled and assign index of previous version
        # - don't care about whether the previous version is the same
        #   as the original, because the same column is being permuted
        #   here multiple times while all others columns are unchanged
        selected_col_shuffled = X.iloc[shuffling_idx, ci]
        selected_col_shuffled.index = X.index
        # replace with re-indexed shuffled values (in-place shuffling)
        X.iloc[:, ci] = selected_col_shuffled
        # predict and score with shuffled column
        # score = bh.calculate_avg_return_vs_theoretical(X, y, pipe, best_t)
        score = bh.calculate_avg_return_vs_theoretical_v2(
            X, y, pipe, best_t
        ).mean()
        if verbose:
            print(f"feature={col}, repeat={repeat_index}, score={score:.2f}")
        # compute permutation importance
        pi = base_score - score
        # bookkeeping
        scores.append(pi)
    return scores


def manual_permutation_importance(
    X, y, pipe, best_t, n_repeats=1, verbose=True
):
    # baseline_score_old = bh.calculate_avg_return_vs_theoretical(
    #     X, y, pipe, best_t
    # )
    baseline_score = bh.calculate_avg_return_vs_theoretical_v2(
        X, y, pipe, best_t
    ).mean()
    shuffling_idx = X.reset_index(drop=True).index.to_numpy()
    rng = np.random.RandomState(42)
    if verbose:
        print(f"baseline score={baseline_score:.2f}")
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(calculate_permutation_scores)(
            X,
            feature_index,
            feature_name,
            pipe,
            y,
            baseline_score,
            n_repeats,
            best_t,
            shuffling_idx,
            rng,
            verbose,
        )
        for feature_index, feature_name in enumerate(list(X))
    )
    importances = executor(tasks)
    importances = pd.DataFrame(importances)
    importances_mean = importances.mean(axis=1)
    return [importances_mean.to_numpy(), importances.to_numpy()]


def manual_plot_permutation_importance(
    X,
    y,
    pipe,
    threshold,
    n_repeats=10,
    split_name="test",
    plot_title="Permutation Importances",
    fig_title_fontsize=14,
    axis_tick_label_fontsize=12,
    axis_label_fontsize=14,
    box_color="cyan",
    fig_size=(8, 8),
):
    importances_mean, importances = manual_permutation_importance(
        X, y, pipe, threshold, n_repeats, False
    )
    sorted_idx = importances_mean.argsort()

    _, ax = plt.subplots(figsize=fig_size)
    sns.boxplot(
        data=importances[sorted_idx][::-1].T,
        orient="h",
        color=box_color,
        saturation=0.5,
        zorder=3,
        ax=ax,
    )
    ax.axvline(x=0, color="k", ls="--", lw=1.25)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(X.columns[sorted_idx][::-1])
    ax.set_title(
        f"{plot_title} ({split_name.title()} split)",
        loc="left",
        fontweight="bold",
        fontsize=fig_title_fontsize,
    )
    ax.set_xlabel(
        f"Change in avg. return (predicted - theoretical), after shuffling "
        f"data {n_repeats} times",
        fontsize=axis_label_fontsize,
    )
    ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    ax.xaxis.grid(True, which="major", color="lightgrey", zorder=0)
    _ = customize_splines(ax)
