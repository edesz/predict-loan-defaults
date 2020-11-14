#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_best_pipes(
    best_cfg_idx, best_dummy_cfg_idx, df_gs, preprocessor, param_cols
):
    best_clf_params_dict = {
        k.split("__")[1]: v
        for k, v in (
            df_gs.iloc[[best_cfg_idx]][param_cols]
            .dropna(how="any", axis=1)
            .to_dict("records")[0]
        ).items()
    }
    best_dummy_clf_params_dict = {
        k.split("__")[1]: v
        for k, v in (
            df_gs.iloc[[best_dummy_cfg_idx]][param_cols]
            .dropna(how="any", axis=1)
            .to_dict("records")[0]
        ).items()
    }
    best_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                df_gs.loc[best_cfg_idx, "clf_obj"].set_params(
                    **best_clf_params_dict
                ),
            ),
        ]
    )
    best_dummy_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                df_gs.loc[best_dummy_cfg_idx, "clf_obj"].set_params(
                    **best_dummy_clf_params_dict
                ),
            ),
        ]
    )
    # print(best_pipe)
    # print(best_dummy_pipe)
    return [best_pipe, best_dummy_pipe]


def gridsearch(
    X_train,
    y_train,
    params_dicts,
    preprocessor,
    cv,
    multi_scorers,
    threshold=0.5,
):
    scores = [
        gs_train_score(
            X_train,
            y_train,
            params_dict,
            clf_name,
            preprocessor,
            cv,
            multi_scorers,
            threshold,
        )
        for clf_name, params_dict in params_dicts.items()
    ]
    df_scores_validation = pd.concat(
        [score[0] for score in scores], ignore_index=True
    )
    return df_scores_validation


def gs_train_score(
    X_train,
    y_train,
    params_dict,
    clf_name,
    preprocessor,
    cv,
    multi_scorers,
    threshold=0.5,
):
    clf_dict = {
        "DummyClassifier": DummyClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=500, random_state=42
        ),
    }
    clf = clf_dict[clf_name]
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    params_dict = {f"clf__{k}": v for k, v in params_dict.items()}
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=params_dict,
        cv=cv,
        scoring=multi_scorers,
        refit="recall_binary",
        return_train_score=True,
        n_jobs=-1,
        verbose=2,
    )
    gs.fit(X_train, y_train)
    df_gs = (
        pd.DataFrame(gs.cv_results_)
        .assign(clf=type(clf).__name__)
        .assign(clf_obj=clf)
    )
    return [df_gs, gs.best_estimator_]
