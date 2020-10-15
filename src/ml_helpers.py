#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import sklearn.metrics as mr
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import src.ml_metrics as mlm


def fit_predict_threshold(pipe, X, y, train, test, thresh):
    # print(thresh)
    X_trainn, X_testn, y_trainn, y_testn = (
        X.iloc[train],
        X.iloc[test],
        y.iloc[train],
        y.iloc[test],
    )
    # print(len(X_trainn), len(X_testn))
    pipe.fit(X_trainn, y_trainn)
    train_recall = mlm.threshold_recall_score(
        y_trainn, pipe.predict_proba(X_trainn)[:, 1], thresh
    )
    train_fpr = -1 * mlm.threshold_fpr_score(
        y_trainn, pipe.predict_proba(X_trainn)[:, 1], thresh
    )
    train_auc = mlm.threshold_auc_score(
        y_trainn, pipe.predict_proba(X_trainn)[:, 1], thresh
    )
    test_recall = mlm.threshold_recall_score(
        y_testn, pipe.predict_proba(X_testn)[:, 1], thresh
    )
    test_fpr = -1 * mlm.threshold_fpr_score(
        y_testn, pipe.predict_proba(X_testn)[:, 1], thresh
    )
    test_auc = mlm.threshold_auc_score(
        y_testn, pipe.predict_proba(X_testn)[:, 1], thresh
    )
    scores = {
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_fpr": train_fpr,
        "test_fpr": test_fpr,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "threshold": thresh,
    }
    return scores


def base_pipeline(
    preprocessor, use_ros=False, use_rus=False, use_smote=False
) -> Pipeline:
    pipe = Pipeline([("preprocessor", preprocessor)])
    if use_rus:
        pipe.steps.append(
            (
                "rus",
                RandomUnderSampler(
                    sampling_strategy="not minority", random_state=42
                ),
            )
        )
    if use_ros:
        pipe.steps.append(
            (
                "ros",
                RandomOverSampler(
                    sampling_strategy="not majority", random_state=42
                ),
            )
        )
    if use_smote:
        pipe.steps.append(
            ("smote", SMOTE(sampling_strategy="minority", random_state=42))
        )
    return pipe


def multi_model_grid_search(
    clf_list,
    X_train,
    y_train,
    preprocessors,
    parameters,
    thresholds=[0.5],
    use_default_threshold=True,
    preprocessor_type="no_trans",
    sort_by_metrics=[
        "mean_test_recall_binary",
        "mean_test_fpr",
        "mean_test_auc",
    ],
):
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    if use_default_threshold:
        thresholds = [0.5]
        fpr_scorer = mr.make_scorer(
            mlm.false_positive_rate_scorer, greater_is_better=False
        )
        rb_scorer = mr.make_scorer(
            mlm.recall_binary_scorer, greater_is_better=True
        )
        auc_scorer = mr.make_scorer(
            mlm.auc_binary_scorer, greater_is_better=True
        )
        multi_scorers = {
            "fpr": fpr_scorer,
            "recall_binary": rb_scorer,
            "auc": auc_scorer,
        }
    data_thresh = []
    for thresh in thresholds:
        if not use_default_threshold:
            multi_scorers = {
                "recall_binary": mr.make_scorer(
                    mlm.threshold_recall_score,
                    greater_is_better=True,
                    needs_proba=True,
                    threshold=thresh,
                ),
                "fpr": mr.make_scorer(
                    mlm.threshold_fpr_score,
                    greater_is_better=False,
                    needs_proba=True,
                    threshold=thresh,
                ),
                "auc": mr.make_scorer(
                    mlm.threshold_auc_score,
                    greater_is_better=True,
                    needs_proba=True,
                    threshold=thresh,
                ),
            }
        data = []
        for clf in clf_list:
            params_dict = {
                f"clf__{k}": v
                for k, v in parameters[type(clf).__name__].items()
            }
            # print(params_dict)
            pipe = base_pipeline(preprocessors[preprocessor_type])
            pipe.steps.append(["clf", clf])
            # print(list(pipe.named_steps.keys()))
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=params_dict,
                cv=cv,
                scoring=multi_scorers,
                refit="recall_binary",
                return_train_score=True,
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            resamplers = list(
                set(list(pipe.named_steps.keys()))
                - set(["preprocessor", "clf"])
            )
            df_gs = (
                pd.DataFrame(gs.cv_results_)
                .assign(preprocessor_type=preprocessor_type)
                .assign(resamplers=",".join(resamplers))
                .assign(
                    clf_params=type(clf).__name__
                    + "_("
                    + str(params_dict)
                    + ""
                    + ")"
                )
            )
            df_gs.insert(0, "clf", type(clf).__name__)
            data.append(df_gs)
            pipe.steps.pop(1)
        data_thresh.append(pd.concat(data).assign(threshold=thresh))
    df = (
        pd.concat(data_thresh)
        .sort_values(by=sort_by_metrics, ascending=[False, True, False])
        .reset_index(drop=True)
    )
    # display(df)
    return df
