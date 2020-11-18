#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.preprocessing as spp
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin


class DFNanThresholdColumnDropper(TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        return X.dropna(thresh=self.threshold * len(X), axis=1)

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFColumnDropper(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        cols_to_drop = []
        for c in list(X):
            for cd in self.columns:
                if cd in c:
                    cols_to_drop.append(cd)
        if cols_to_drop:
            return X.drop(cols_to_drop, axis=1)
        else:
            return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFColumnFilterList(TransformerMixin):
    def __init__(self, column_name, column_values):
        self.column_name = column_name
        self.column_values = column_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        # return X[
        #     pd.DataFrame(X[self.column_name].tolist())
        #     .isin(self.column_values)
        #     .any(1)
        # ]
        return X.loc[X[self.column_name].isin(self.column_values)]

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFColumnMapper(TransformerMixin):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        return X.replace(self.mapping_dict)

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFNonUniqueValColDropper(TransformerMixin):
    def __init__(self, num_non_unique_vals):
        self.num_non_unique_vals = num_non_unique_vals

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X = X.loc[:, X.apply(pd.Series.nunique) > self.num_non_unique_vals]
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFDropNaN(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        return X.dropna()

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFPctNumeric(TransformerMixin):
    def __init__(self, cols, str_to_remove):
        self.str_to_remove = str_to_remove
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        for col in self.cols:
            X[col] = (
                X[col]
                .astype(str)
                .str.rstrip(self.str_to_remove)
                .astype("float")
            )
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFOneHotEncoder(TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        if self.columns:
            dummies = pd.get_dummies(X[self.columns])
        else:
            dummies = pd.get_dummies(X)
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=self.columns, axis=1)
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFSingleColumnMapper(TransformerMixin):
    def __init__(self, col, mapping_dict):
        self.col = col
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X[list(self.mapping_dict.keys())[0]] = X[self.col]
        return X.replace(self.mapping_dict)

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFSimpleDtypeChanger(TransformerMixin):
    def __init__(self, col, datatype):
        self.col = col
        self.datatype = datatype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X[self.col] = X[self.col].astype(self.datatype)
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFStandardScaler(TransformerMixin):
    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = spp.StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class DFColumnStdFilter(TransformerMixin):
    def __init__(self, col, n_std):
        self.col = col
        self.n_std = n_std

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        return X[
            np.abs(X[self.col] - X[self.col].mean())
            <= (self.n_std * X[self.col].std())
        ]

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFPowerTransformer(TransformerMixin):
    def __init__(self, method="yeo-johnson"):
        self.method = method

    def fit(self, X, y=None):
        self.pt = spp.PowerTransformer(method=self.method)
        self.pt.fit(X)
        return self

    def transform(self, X):
        Xss = self.pt.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class DFQuantileTransformer(TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution="normal"):
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        self.pt = spp.QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
        )
        self.pt.fit(X)
        return self

    def transform(self, X):
        Xss = self.pt.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class DFHierarchicalClusterSpearmanRank(TransformerMixin):
    def __init__(self, threshold=1):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(corr)
        cluster_ids = hierarchy.fcluster(
            corr_linkage, self.threshold, criterion="distance"
        )
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        self.selected_features = [
            v[0] for v in cluster_id_to_feature_ids.values()
        ]
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        return X.iloc[:, self.selected_features]

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFOneHotEncoderV2(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.ohe = spp.OneHotEncoder(handle_unknown=self.handle_unknown)
        self.ohe.fit(X.to_numpy())
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xohe = self.ohe.transform(X.to_numpy()).toarray()
        self.feature_names_ = list(self.ohe.get_feature_names(list(X)))
        Xtrans = pd.DataFrame(
            Xohe, index=X.index, columns=self.feature_names_
        ).astype(int)
        # print(self.feature_names_)
        return Xtrans

    def get_features(self):
        return self.feature_names_

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFCorrColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, corr_max_threshold=0.5, method="spearman"):
        self.method = method
        self.corr_max_threshold = corr_max_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr(method=self.method)
        corr_abs = corr.abs()
        # Select upper triangle of correlation matrix
        upper = corr_abs.where(
            np.triu(np.ones(corr_abs.shape), k=1).astype(np.bool)
        )
        # Find index of feature columns with correlation greater than
        # corr_max_threshold
        self.to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.corr_max_threshold)
        ]
        # print(to_drop)
        self.corrm = X.columns[~corr_abs.columns.isin(self.to_drop)].tolist()
        excluded_cols = list(set(list(X)) - set(list(self.corrm)))
        assert len(excluded_cols) == len(self.to_drop)
        self.corrm_idxs = X.columns.get_indexer(self.corrm).tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.iloc[:, self.corrm_idxs]

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)

    def get_feature_indexes(self):
        return self.corrm_idxs

    def get_dropped_feature_indexes(self):
        return self.to_drop
