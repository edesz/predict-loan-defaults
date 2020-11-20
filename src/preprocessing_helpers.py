#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from src.ml_helpers_v2 import create_pipe


def get_transformed_data(
    clf,
    best_pipe,
    X_train,
    y_train,
    X_test,
    preprocessor_type,
    numerical_columns,
    nominal_columns,
    corr_max_threshold=0.5,
    corr_method="spearman",
):
    # Instantiate transformation (pre-processing) and classification
    # (ML) pipelines
    pipe_trans, pipe_clf, _ = create_pipe(
        clf,
        preprocessor_type,
        numerical_columns,
        nominal_columns,
        corr_max_threshold,
        corr_method,
    )

    # Pre-Process training+validation and testing splits
    X_train_trans = pipe_trans.fit_transform(X_train, y_train)
    X_test_trans = pipe_trans.transform(X_test)

    # Assign column names to columns of pre-processed splits
    # # Get indexes of features with correlation coefficient above specified
    # # threshold
    idxs_wanted = pipe_trans.named_steps["fs"].get_feature_indexes()
    # # Get column names of non-correlated features (after pre-processing)
    cols_no_correlation = (
        pd.Series(
            numerical_columns
            + pipe_trans.named_steps["preprocessor"]
            .named_transformers_["onehot"]
            .get_features()
        )
        .iloc[idxs_wanted]
        .tolist()
    )
    # print(cols_no_correlation)
    X_train_trans.columns = cols_no_correlation
    X_test_trans.columns = cols_no_correlation
    # # Change datatype of one-hot encoded features
    cats_cols_categorized = X_train_trans.columns[
        ~X_train_trans.columns.isin(numerical_columns)
    ].tolist()
    X_train_trans[cats_cols_categorized] = X_train_trans[
        cats_cols_categorized
    ].astype(int)
    X_test_trans[cats_cols_categorized] = X_test_trans[
        cats_cols_categorized
    ].astype(int)

    # Train ML pipeline and verify output agrees with output from full
    # (pre-processing + ML) pipeline
    pipe_clf.fit(X_train_trans, y_train)
    y_test_pred_proba = pipe_clf.predict_proba(X_test_trans)
    y_test_pipe_pred_proba = best_pipe.predict_proba(X_test)
    assert np.array_equal(
        y_test_pred_proba, y_test_pipe_pred_proba, equal_nan=False
    )
    return [pipe_clf, X_train_trans, X_test_trans]
