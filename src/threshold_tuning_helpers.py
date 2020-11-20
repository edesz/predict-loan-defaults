#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count
from time import time

import numpy as np
import pandas as pd
import scipy.stats as st
from joblib import Parallel, delayed

from src.business_helpers import calculate_avg_return_vs_theoretical_v2


def score_single_threshold(X, y, pipe, t, verbose=False):
    start_time = time()
    score = calculate_avg_return_vs_theoretical_v2(X, y, pipe, t)
    duration = time() - start_time
    five_percentile, ninetyfive_percentile = st.t.interval(
        0.95, len(score) - 1, loc=np.mean(score), scale=st.sem(score)
    )
    twentyfive_percentile, seventyfive_percentile = st.t.interval(
        0.75, len(score) - 1, loc=np.mean(score), scale=st.sem(score)
    )
    fifty_percentile, _ = st.t.interval(
        0.50, len(score) - 1, loc=np.mean(score), scale=st.sem(score)
    )
    r = {
        "label": type(pipe.named_steps["clf"]).__name__,
        "threshold": t,
        "mae": score.mean(),
        "mdae": score.median(),
        "stdae": score.std(),
        "count": score.count(),
        "whislo": five_percentile,
        "whishi": ninetyfive_percentile,
        "q1": twentyfive_percentile,
        "med": fifty_percentile,
        "q3": seventyfive_percentile,
        "fliers": [],
        "duration": duration,
    }
    if verbose:
        print(
            f"Model={type(pipe.named_steps['clf']).__name__}, "
            f"threshold={t:.2f}, mae={score.mean():.2f}, "
            f"duration={duration:.2f} sec\n"
        )
    return r


def threshold_tuning_reshaping(df_t_tuned, get_relative_score=False):
    if get_relative_score:
        df_t_tuned = df_t_tuned.assign(
            relative_mae=(df_t_tuned["mae"] / df_t_tuned["mae"].max())
        ).drop(["count"], axis=1)
    df_t_tuned = (
        df_t_tuned.set_index("threshold")
        .unstack()
        .reset_index()
        .rename(columns={"level_0": "metric", 0: "value"})
    )
    return df_t_tuned


def compute_cost_for_thresholds(
    thresholds,
    X_test,
    y_test,
    pipe,
    get_relative_score=False,
    verbose=False,
):
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(score_single_threshold)(
            X_test,
            y_test,
            pipe,
            t,
            verbose,
        )
        for t in thresholds
    )
    d = executor(tasks)
    dataframe_cost_func = pd.DataFrame.from_records(d)
    dataframe_cost_func = threshold_tuning_reshaping(
        dataframe_cost_func, get_relative_score
    )
    return dataframe_cost_func
