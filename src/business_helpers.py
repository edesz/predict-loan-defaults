#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def int_income_calculator(r, p, n):
    """
    SOURCE:
    https://www.vertex42.com/ExcelArticles/amortization-calculation.html

    Inputs
    r: annual interest rate (float)

    Returns
    installment: monthly payment on loan (float)
        - monthly payment owed by borrower if loan is approved
        - this matches the value in the installment column from the
          Lending Club loans raw data
    """
    # print(r, p, n)
    if r > 1:
        r = r / 100
    r /= 12
    monthly_installment = p * ((r * (1 + r) ** n) / (((1 + r) ** n) - 1))
    interest_income = monthly_installment * n
    return interest_income


def convert_confusion_matrix_to_business_cost(
    y_true, y_pred, cm_labels, principal, interest_income
):
    df_cm = pd.DataFrame(
        confusion_matrix(
            np.array([y_true]),
            np.array([y_pred]),
            labels=cm_labels,
        ),
        index=cm_labels,
        columns=cm_labels,
    )
    TN = df_cm.iloc[0, 0]
    FP = df_cm.iloc[0, 1]
    FN = df_cm.iloc[1, 0]
    TP = df_cm.iloc[1, 1]
    fn_pen = principal
    tn_pen = interest_income
    fp_pen = interest_income
    tp_pen = 0
    ds = (-fn_pen * FN) + (tp_pen * TP) + (-fp_pen * FP) + (tn_pen * TN)
    return ds


# def calculate_avg_return_vs_theoretical(X, y, pipe, threshold):
#     theoretical_test = (
#         np.vectorize(int_income_calculator)(
#             X["int_rate"],
#             X["loan_amnt"],
#             X["term"].str.extract(r"(\d+)").astype(int).squeeze(),
#         )
#         - X["loan_amnt"]
#     )
#     y_probs = pipe.predict_proba(X)[:, 1]
#     cm_labels = np.sort(np.unique(y))
#     df_t_cm = pd.DataFrame(
#         confusion_matrix(
#             y, np.where(y_probs > threshold, 1, 0), labels=cm_labels
#         ),
#         index=np.sort(np.unique(y)),
#         columns=np.sort(np.unique(y)),
#     )
#     TN = df_t_cm.iloc[0, 0]
#     FP = df_t_cm.iloc[0, 1]
#     FN = df_t_cm.iloc[1, 0]
#     TP = df_t_cm.iloc[1, 1]
#     fn_penalty = X["loan_amnt"]
#     tn_penalty = theoretical_test
#     fp_penalty = theoretical_test
#     tp_penalty = 0
#     ds = (
#         (-fn_penalty * FN)
#         + (tp_penalty * TP)
#         + (-fp_penalty * FP)
#         + (tn_penalty * TN)
#     )
#     ds /= len(y)
#     # root_mean_sq_err = mr.mean_squared_error(
#     #     theoretical_test, ds, squared=False
#     # )
#     # mean_abs_err = mr.mean_absolute_error(theoretical_test, ds)
#     mean_err = (ds - theoretical_test).mean()
#     return mean_err


def rowwise_calculate_avg_return_vs_theoretical(
    principal,
    y,
    t,
    y_probs,
    interest_income,
):
    cm_labels = np.sort(np.unique([0, 1]))
    ds_predicted = convert_confusion_matrix_to_business_cost(
        np.array([y]),
        np.array([np.where(y_probs > t, 1, 0)]),
        cm_labels,
        principal,
        interest_income,
    )
    ds_true = convert_confusion_matrix_to_business_cost(
        np.array([y]),
        np.array([y]),
        cm_labels,
        principal,
        interest_income,
    )
    err = np.abs(ds_predicted - ds_true)
    return err


def calculate_avg_return_vs_theoretical_v2(X, y, pipe, t):
    r = X["int_rate"]
    p = X["loan_amnt"]
    n = X["term"].str.extract(r"(\d+)").astype(int).squeeze()
    interest_income = np.vectorize(int_income_calculator)(r, p, n) - p
    y_probs = pipe.predict_proba(X)[:, 1]
    score = np.vectorize(rowwise_calculate_avg_return_vs_theoretical)(
        p,
        y,
        t,
        pd.Series(y_probs, index=X.index),
        interest_income,
    )
    score = pd.Series(score, index=X.index)
    return score
