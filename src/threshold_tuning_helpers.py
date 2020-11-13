#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix

from src.business_helpers import int_income_calculator


def compute_cost_for_thresholds(
    confs, thresholds, y_test, y_probs, verbose=False
):
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = (
        delayed(probability_threshold_to_cost)(
            y_test,
            y_probs,
            conf["p"],  # FN
            conf["ret"],  # FP
            conf["ret"],  # TN
            0,  # TP
            t,
            counter,
            calc_cost_per_observation=True,
            verbose=verbose,
        )
        for counter, conf in enumerate(confs)
        for t in thresholds
    )
    d = executor(tasks)
    df_cost_func = pd.DataFrame.from_records(d)
    return df_cost_func


def get_combos(X_test, p_type="max"):
    median_loan_amnt = X_test["loan_amnt"].median()
    records = {
        "max": X_test.loc[X_test["loan_amnt"] == X_test["loan_amnt"].max()],
        "min": X_test.loc[X_test["loan_amnt"] == X_test["loan_amnt"].min()],
        "median": X_test.loc[
            X_test["loan_amnt"].sub(median_loan_amnt).abs().idxmin()
        ],
    }
    if p_type == "median":
        d = {}
        d["r_for_median_p"] = records["median"].loc["int_rate"]
        d["n_for_median_p"] = records["median"].loc["term"]
        d["median_p"] = records["median"].loc["loan_amnt"]
        d["return_for_median_p"] = (
            int_income_calculator(
                d["r_for_median_p"],
                d["median_p"],
                int(re.findall("[0-9]+", d["n_for_median_p"])[0]),
            )
            - d["median_p"]
        )
        df_median_r = pd.DataFrame.from_dict(d, orient="index").T
        return {p_type: df_median_r}
    else:
        n_for_min_r = records[p_type].loc[
            records[p_type]["int_rate"] == records[p_type]["int_rate"].min(),
            "term",
        ]
        n_for_max_r = records[p_type].loc[
            records[p_type]["int_rate"] == records[p_type]["int_rate"].max(),
            "term",
        ]
        d = {}
        d["min_r"] = records[p_type]["int_rate"].min()
        d["n_for_min_r"] = n_for_min_r.unique()[0]
        d["p_for_min_r"] = records[p_type]["loan_amnt"].unique()[0]
        d["return_min_r"] = (
            int_income_calculator(
                d["min_r"],
                d["p_for_min_r"],
                int(re.findall("[0-9]+", d["n_for_min_r"])[0]),
            )
            - d["p_for_min_r"]
        )
        df_min_r = pd.DataFrame.from_dict(d, orient="index").T
        d = {}
        d["max_r"] = records[p_type]["int_rate"].max()
        d["n_for_max_r"] = n_for_max_r.unique()[0]
        d["p_for_max_r"] = records[p_type]["loan_amnt"].unique()[0]
        d["return_max_r"] = (
            int_income_calculator(
                d["max_r"],
                d["p_for_max_r"],
                int(re.findall("[0-9]+", d["n_for_max_r"])[0]),
            )
            - d["p_for_max_r"]
        )
        df_max_r = pd.DataFrame.from_dict(d, orient="index").T
        return {p_type: {"min": df_min_r, "max": df_max_r}}


def get_components_of_returns(X):
    d = {}
    for p_type in ["min", "max"]:
        d[f"{p_type}_p_min_r"] = get_combos(X, p_type)[p_type]["min"]
        d[f"{p_type}_p_max_r"] = get_combos(X, p_type)[p_type]["max"]
        for r_type in ["min", "max"]:
            d[f"{p_type}_p_{r_type}_r"].columns = ["r", "n", "p", "ret"]
    d["median_p"] = get_combos(X, p_type="median")["median"]
    d["median_p"].columns = ["r", "n", "p", "ret"]
    confs = list(
        pd.concat(list(d.values()))
        .reset_index(drop=True)
        .drop_duplicates()
        .reset_index(drop=True)
        .to_dict("records")
    )
    # d, confs
    return [d, confs]


def get_cost_func(confs, thresholds, X, y, pipe):
    try:
        y_probs = pipe.predict_proba(X)[:, 1]
    except NotFittedError as e:
        print(f"{repr(e)} - Pipeline is not fitted")
        raise
    df_cost_func = compute_cost_for_thresholds(
        confs, thresholds, y, y_probs, verbose=False
    )
    return df_cost_func


def probability_threshold_to_cost(
    y_test,
    y_probs,
    fn_penalty,
    fp_penalty,
    tn_penalty,
    tp_penalty,
    t=0.5,
    config_idx=0,
    calc_cost_per_observation=False,
    verbose=False,
):
    df_t_cm = pd.DataFrame(
        confusion_matrix(
            y_test,
            np.where(y_probs > t, 1, 0),
            labels=np.sort(np.unique(y_test)),
        ),
        index=np.sort(np.unique(y_test)),
        columns=np.sort(np.unique(y_test)),
    )
    TN = df_t_cm.iloc[0, 0]
    FP = df_t_cm.iloc[0, 1]
    FN = df_t_cm.iloc[1, 0]
    TP = df_t_cm.iloc[1, 1]
    ds = (
        (-fn_penalty * FN)
        + (tp_penalty * TP)
        + (-fp_penalty * FP)
        + (tn_penalty * TN)
    )
    if verbose:
        print(
            f"FN={fn_penalty}, FP={fp_penalty}, TN={tn_penalty}, "
            f"TP={tp_penalty}, t={t}, cost={ds:.2f}"
        )
    if calc_cost_per_observation:
        ds /= len(y_test)
    r = {
        "config": config_idx,
        "threshold": t,
        "cost_func": ds,
        "fn": fn_penalty,
        "fp": fp_penalty,
        "tn": tn_penalty,
        "tp": tp_penalty,
    }
    return r


def threshold_tuning_reshaping(df_t_tuned):
    df_t_tuned["p"] = (
        df_t_tuned["p"].astype(float).apply(lambda x: f"${x:,.0f}")
    )
    rate_formatted_for_display = (
        df_t_tuned["r"].astype(float).apply(lambda x: f"{x:.2f}%")
    )
    df_t_tuned["r"] = rate_formatted_for_display
    df_t_tuned["n"] = df_t_tuned["n"].apply(lambda x: f"{x} mo.")
    df_t_tuned = df_t_tuned.assign(
        cfg=df_t_tuned[["r", "n", "p"]]
        .astype(str)
        .apply(
            lambda x: "(" + ", ".join(x.dropna().values.tolist()) + ")", axis=1
        )
    )[["cfg", "clf", "theoretical", "return_at_best_t"]]
    df_t_tuned = pd.concat(
        [
            df_t_tuned[df_t_tuned["clf"] == "DummyClassifier"][
                ["cfg", "clf", "return_at_best_t"]
            ].rename(columns={"return_at_best_t": "return"}),
            df_t_tuned[df_t_tuned["clf"] != "DummyClassifier"][
                ["cfg", "clf", "return_at_best_t"]
            ].rename(columns={"return_at_best_t": "return"}),
            df_t_tuned[df_t_tuned["clf"] == "DummyClassifier"][
                ["cfg", "clf", "theoretical"]
            ]
            .rename(columns={"theoretical": "return"})
            .replace({"clf": {"DummyClassifier": "Theoretical"}}),
        ],
        ignore_index=True,
    )
    return df_t_tuned
