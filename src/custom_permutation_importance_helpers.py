#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

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
    rng = check_random_state(rng)
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
            print(f"feat={col}, repeat={repeat_index}, score={score:.2f}\n")
        # compute permutation importance
        pi = base_score - score
        # bookkeeping
        scores.append(pi)
    return scores


def manual_permutation_importance(
    X, y, pipe, best_t, n_repeats=1, verbose=True, parallel=False
):
    # baseline_score_old = bh.calculate_avg_return_vs_theoretical(
    #     X, y, pipe, best_t
    # )
    baseline_score = bh.calculate_avg_return_vs_theoretical_v2(
        X, y, pipe, best_t
    ).mean()
    # permuted_cols = list(
    #     set(list(X)) - set(["int_rate", "loan_amnt", "term"])
    # )
    permuted_cols = [
        c for c in list(X) if c not in ["int_rate", "loan_amnt", "term"]
    ]
    shuffling_idx = X.reset_index(drop=True).index.to_numpy()
    # rng = np.random.RandomState(42)
    random_state = check_random_state(42)
    rng = random_state.randint(np.iinfo(np.int32).max + 1)
    if verbose:
        print(f"baseline score={baseline_score:.2f}")
    if parallel:
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
            for feature_index, feature_name in enumerate(permuted_cols)
        )
        importances = executor(tasks)
    else:
        importances = [
            calculate_permutation_scores(
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
            for feature_index, feature_name in enumerate(permuted_cols)
        ]
    importances = pd.DataFrame(importances)
    # print(importances)
    importances_mean = importances.mean(axis=1)
    return [
        importances_mean.to_numpy(),
        importances.to_numpy(),
        permuted_cols,
    ]


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
    verbose=False,
    parallel=False,
):
    (
        importances_mean,
        importances,
        permuted_cols,
    ) = manual_permutation_importance(
        X, y, pipe, threshold, n_repeats, verbose, parallel
    )
    df_importances_mean = pd.DataFrame(
        importances_mean, index=permuted_cols, columns=["imp"]
    ).sort_values(by=["imp"], ascending=False)
    df_importances = pd.DataFrame(importances, index=permuted_cols).reindex(
        df_importances_mean.index
    )
    _, ax = plt.subplots(figsize=fig_size)
    sns.boxplot(
        data=df_importances.T,
        orient="h",
        color=box_color,
        saturation=0.5,
        zorder=3,
        ax=ax,
    )
    ax.axvline(x=0, color="k", ls="--", lw=1.25)
    ax.set_yticks(range(len(permuted_cols)))
    ax.set_yticklabels(df_importances_mean.index.tolist())
    ax.set_title(
        f"{plot_title} ({split_name.title()} split)",
        loc="left",
        fontweight="bold",
        fontsize=fig_title_fontsize,
    )
    ax.set_xlabel(
        f"Change in avg. return (predicted - true), after shuffling "
        f"data {n_repeats} times",
        fontsize=axis_label_fontsize,
    )
    ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    ax.xaxis.grid(True, which="major", color="lightgrey", zorder=0)
    _ = customize_splines(ax)
    return [
        importances_mean,
        importances,
        permuted_cols,
        df_importances,
        df_importances_mean,
    ]
    # return [
    #     importances_mean,
    #     importances,
    #     permuted_cols,
    # ]
