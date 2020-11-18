#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import src.business_helpers as bh
from src.ml_helpers_v2 import create_pipe
from src.visualization_helpers import add_gridlines, customize_splines


def calculate_looc_scores(
    X_train,
    X_test,
    y_train,
    y_test,
    ci,
    col,
    preprocessor_type,
    numerical_columns,
    nominal_columns,
    clf,
    base_score,
    best_t,
    corr_max_threshold=0.5,
    corr_method="spearman",
    verbose=False,
):
    # print(col, list(X_train), list(X_train.copy().drop(col, axis=1)))
    X_tr = X_train.drop(col, axis=1)
    X_ts = X_test.drop(col, axis=1)
    if col in numerical_columns:
        numerical_columns = list(set(numerical_columns) - set([col]))
    if col in nominal_columns:
        nominal_columns = list(set(nominal_columns) - set([col]))
    _, _, pipe = create_pipe(
        clf,
        preprocessor_type,
        numerical_columns,
        nominal_columns,
        corr_max_threshold,
        corr_method,
    )
    pipe.fit(X_tr, y_train)
    # predict and score with dropped column
    score = bh.calculate_avg_return_vs_theoretical_v2(
        X_ts, y_test, pipe, best_t
    ).mean()
    if verbose:
        print(f"feat={col}, score={score:.2f}\n")
    # compute LOOC importance
    looci = base_score - score
    # bookkeeping
    scores = {col: looci}
    return scores


def manual_looc_importance(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor_type,
    numerical_columns,
    nominal_columns,
    clf,
    best_t,
    corr_max_threshold,
    corr_method,
    verbose=True,
):
    # baseline_score_old = bh.calculate_avg_return_vs_theoretical(
    #     X, y, pipe, best_t
    # )
    _, _, pipe = create_pipe(
        clf,
        preprocessor_type,
        numerical_columns,
        nominal_columns,
        corr_max_threshold,
        corr_method,
    )
    pipe.fit(X_train, y_train)
    baseline_score = bh.calculate_avg_return_vs_theoretical_v2(
        X_test, y_test, pipe, best_t
    ).mean()
    if verbose:
        print(f"baseline score={baseline_score:.2f}\n")
    cols_to_drop = list(
        set(list(X_train)) - set(["loan_amnt", "int_rate", "term"])
    )
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(calculate_looc_scores)(
            X_train,
            X_test,
            y_train,
            y_test,
            feature_index,
            feature_name,
            preprocessor_type,
            numerical_columns,
            nominal_columns,
            clf,
            baseline_score,
            best_t,
            corr_max_threshold,
            corr_method,
            verbose,
        )
        for feature_index, feature_name in enumerate(cols_to_drop)
    )
    scores = executor(tasks)
    importances = (
        pd.DataFrame(scores)
        .unstack()
        .dropna()
        .reset_index(level=0)
        .rename(columns={"level_0": "feature", 0: "looc_imp"})
    )
    return importances


def manual_plot_looc_importance(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor_type,
    numerical_columns,
    nominal_columns,
    clf,
    threshold,
    corr_max_threshold,
    corr_method,
    split_name="test",
    plot_title="LOOC Importances",
    fig_title_fontsize=14,
    axis_tick_label_fontsize=12,
    axis_label_fontsize=14,
    box_color="cyan",
    verbose=False,
    fig_size=(8, 8),
):
    importances = manual_looc_importance(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor_type,
        numerical_columns,
        nominal_columns,
        clf,
        threshold,
        corr_max_threshold,
        corr_method,
        verbose,
    )

    _, ax = plt.subplots(figsize=fig_size)
    importances.sort_values(by=["looc_imp"], ascending=True).set_index(
        "feature"
    ).plot(kind="barh", ax=ax)
    ax.axvline(x=0, color="k", ls="--", lw=1.25)
    ax.get_legend().remove()
    ax.set_title(
        f"{plot_title} ({split_name.title()} split)",
        loc="left",
        fontweight="bold",
        fontsize=fig_title_fontsize,
    )
    ax.set_xlabel(
        "Change in avg. return (predicted - true)",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(None)
    ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    ax.xaxis.grid(True, which="major", color="lightgrey", zorder=0)
    _ = customize_splines(ax)
    _ = add_gridlines(ax)
    return importances
