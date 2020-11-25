#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as mr
import sklearn.model_selection as ms
import yellowbrick.classifier as ybc
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

import src.ml_helpers as mlh


def customize_splines(ax: plt.axis) -> plt.axis:
    ax.spines["left"].set_edgecolor("black")
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_zorder(3)
    ax.spines["bottom"].set_edgecolor("black")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_zorder(3)
    ax.spines["top"].set_edgecolor("lightgrey")
    ax.spines["top"].set_linewidth(1)
    ax.spines["right"].set_edgecolor("lightgrey")
    ax.spines["right"].set_linewidth(1)
    return ax


def add_gridlines(ax: plt.axis):
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    return ax


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    cv=None,
    scorer="f1_score",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    legend_coords=(0.9, 1.15),
    axis_tick_label_fontsize=12,
    fig_size=(20, 5),
):
    _, axes = plt.subplots(3, 1, figsize=fig_size)
    axes[0].set_title(
        title + " versus Training size",
        loc="left",
        fontweight="bold",
        fontsize=axis_tick_label_fontsize,
    )
    axes[0].set_xlabel(None)

    train_sizes, train_scores, test_scores, fit_times, _ = ms.learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    # print(train_scores, cv, scorer, train_sizes)
    train_split_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_split_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_split_scores_mean - train_scores_std,
        train_split_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_split_scores_mean - test_scores_std,
        test_split_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_split_scores_mean, "o-", color="r", label="Train"
    )
    axes[0].plot(
        train_sizes, test_split_scores_mean, "o-", color="g", label="Test"
    )
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=legend_coords,
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.4,
    )
    axes[0].xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel(None)
    axes[1].set_title(
        "Fit times (sec) versus Training size",
        loc="left",
        fontweight="bold",
        fontsize=axis_tick_label_fontsize,
    )
    axes[1].xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_split_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_split_scores_mean - test_scores_std,
        test_split_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel(None)
    axes[2].set_title(
        "Test score versus Training time (sec)",
        loc="left",
        fontweight="bold",
        fontsize=axis_tick_label_fontsize,
    )
    for ax in axes:
        ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
        _ = customize_splines(ax)


def builtin_plot_permutation_importances(
    pipe,
    X_train,
    X_test,
    y_train,
    y_test,
    scorer,
    n_repeats,
    wspace=0.5,
    fig_title_fontsize=16,
    fig_title_vertical_pos=1.1,
    axis_tick_label_fontsize=12,
    axis_label_fontsize=14,
    box_color="cyan",
    fig_size=(12, 6),
):
    scorer_name = scorer._score_func.__name__.split("_score")[0].replace(
        "threshold_", ""
    )
    fig_title = (
        f"{scorer_name.upper()} Permutation Importances using "
        f"{type(pipe.named_steps['clf']).__name__}"
    )
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(
        fig_title,
        fontsize=fig_title_fontsize,
        fontweight="bold",
        y=fig_title_vertical_pos,
    )
    grid = plt.GridSpec(1, 2, wspace=wspace)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    for Xs, ys, ax, split_name in zip(
        [X_train, X_test], [y_train, y_test], [ax1, ax2], ["train", "test"]
    ):
        result = permutation_importance(
            pipe,
            Xs,
            ys,
            scoring=scorer,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )
        sorted_idx = result.importances_mean.argsort()
        sns.boxplot(
            data=result.importances[sorted_idx][::-1].T,
            orient="h",
            color=box_color,
            saturation=0.5,
            zorder=3,
            ax=ax,
        )
        ax.axvline(x=0, color="k", ls="--")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(Xs.columns[sorted_idx][::-1])
        ax.set_title(
            f"{split_name.title()}",
            loc="left",
            fontweight="bold",
            fontsize=axis_tick_label_fontsize,
        )
        ax.set_xlabel(
            f"Change in avg. score, over {n_repeats} passes through the data",
            fontsize=axis_label_fontsize,
        )
        ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
        ax.xaxis.grid(True, which="major", color="lightgrey", zorder=0)
        _ = customize_splines(ax)


def plot_coefs(
    coefs,
    ptitle="Coefficient variability",
    axis_tick_label_font_size=12,
    fig_size=(9, 7),
):
    _, ax = plt.subplots(figsize=fig_size)
    # sns.swarmplot(data=coefs, orient="h", color="k", alpha=1, zorder=3)
    sns.boxplot(
        data=coefs, orient="h", color="cyan", saturation=0.5, zorder=3, ax=ax
    )
    ax.axvline(x=0, color="k", ls="--")
    ax.set_title(
        ptitle,
        loc="left",
        fontweight="bold",
        fontsize=axis_tick_label_font_size,
    )
    ax.xaxis.set_tick_params(labelsize=axis_tick_label_font_size)
    ax.yaxis.set_tick_params(labelsize=axis_tick_label_font_size)
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    _ = customize_splines(ax)


def iterate_discrimination_thresholds(pipe, X, y, cv, threshold_values_array):
    executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
    tasks = [
        delayed(mlh.fit_predict_threshold)(pipe, X, y, train, test, thresh)
        for thresh in np.nditer(threshold_values_array)
        for train, test in cv.split(X, y)
    ]
    scores = executor(tasks)
    df_scores = pd.DataFrame.from_records(scores).astype({"threshold": str})
    # display(df_scores.head(8))
    return df_scores


def plot_discrimination_threshold(
    scores,
    pipe,
    metrics=["recall", "fpr", "auc"],
    split_types=["test"],
    ax=None,
):
    grouped_scores = scores.groupby("threshold").agg(
        {
            k: ["mean", "min", "max"]
            for k in [f"{s}_{m}" for m in metrics for s in split_types]
        }
    )
    # display(manual_scores_grouped)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for split_type in split_types:
        xyd = {}
        for metric in ["recall", "fpr", "auc"]:
            for aggfunc in ["mean", "min", "max"]:
                xyd[f"{split_type}_{metric}_{aggfunc}"] = grouped_scores.loc[
                    slice(None), [(f"{split_type}_{metric}", aggfunc)]
                ]
                if aggfunc in ["min", "max"]:
                    xyd[f"{split_type}_{metric}_{aggfunc}"] = xyd[
                        f"{split_type}_{metric}_{aggfunc}"
                    ].squeeze()
                else:
                    xyd[f"{split_type}_{metric}_{aggfunc}"].columns = (
                        xyd[f"{split_type}_{metric}_{aggfunc}"]
                        .columns.map("_".join)
                        .str.strip("_")
                    )
        for metric in metrics:
            xyd[f"{split_type}_{metric}_mean"].squeeze().plot(
                ax=ax,
                label="_".join(
                    xyd[f"{split_type}_{metric}_mean"]
                    .squeeze()
                    .name.split("_", 2)[:2]
                ),
            )
            ax.fill_between(
                xyd[f"{split_type}_{metric}_{aggfunc}"].squeeze().index,
                xyd[f"{split_type}_{metric}_min"],
                xyd[f"{split_type}_{metric}_max"],
                alpha=1.0,
                lw=1,
            )
    ax.legend()
    ax.set_xlabel(None)
    ax.set_title(
        (
            "Discrimination Threshold Plot for "
            f"{type(pipe.named_steps['clf']).__name__}"
        )
    )


def show_yb_grid(
    estimator,
    X_test,
    y_test,
    classes,
    X,
    y,
    threshold_cv,
    threshold_values,
    wspace=0.6,
    hspace=0.6,
    fig_size: tuple = (16, 8),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(3, 2, hspace=hspace, wspace=wspace)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[2, 0])
    ax6 = fig.add_subplot(grid[2, 1])
    for k, (viz_func, ax) in enumerate(
        zip(
            [
                ybc.ClassPredictionError,
                ybc.ConfusionMatrix,
                ybc.ClassificationReport,
                ybc.ROCAUC,
            ],
            [ax1, ax3, ax4, ax5],
        )
    ):
        visualizer = viz_func(
            estimator,
            classes=classes,
            is_fitted="auto",
            ax=ax,
        )
        visualizer.fit(X_test, y_test)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        if k == 0:
            ax.get_legend().remove()
    plot_discrimination_threshold(
        iterate_discrimination_thresholds(
            estimator, X, y, threshold_cv, threshold_values
        ),
        estimator,
        ["recall", "fpr", "auc"],
        ["test"],
        ax2,
    )
    plot_roc_curve(estimator, X_test, y_test, ax=ax6)


def plot_cross_validated_coefs(
    pipe,
    numerical_columns,
    nominal_columns,
    X_train,
    X_test,
    y_train,
    y_test,
    scorer,
    n_repeats=5,
    n_splits=5,
    axis_tick_label_fontsize=12,
    fig_size=(8, 12),
):
    feature_names = (
        pipe.named_steps["preprocessor"]
        .named_transformers_["onehot"]
        .get_feature_names(input_features=nominal_columns)
    )
    feature_names = np.concatenate([numerical_columns, feature_names])
    cv_model = ms.cross_validate(
        pipe,
        X=pd.concat([X_train, X_test]),
        y=pd.concat([y_train, y_test]),
        cv=ms.RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        ),
        scoring=scorer,
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1,
    )
    coefs = pd.DataFrame(
        [
            est.named_steps["clf"].coef_.flatten()
            for est in cv_model["estimator"]
        ],
        columns=feature_names,
    )
    coefs = coefs[coefs.mean(axis=0).sort_values(ascending=False).index]
    plot_coefs(
        coefs, "Coefficient variability", axis_tick_label_fontsize, fig_size
    )


def plot_grouped_bar_chart(
    df, groupby, col_to_plot, wspace=0.5, fig_size=(8, 4)
):
    misclassified_str = "(pct. misclassified)"
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(1, 2, wspace=wspace)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    (
        100 * (df[df[col_to_plot]][groupby].value_counts(normalize=True))
    ).sort_values(ascending=False).sort_values(ascending=True).plot.barh(
        ax=ax1, rot=0, zorder=3
    )
    ax1.set_ylabel(None)
    ax1.set_title(
        f"{groupby.title()} {misclassified_str}", loc="left", fontweight="bold"
    )
    (100 * (df[groupby].value_counts(normalize=True))).sort_values(
        ascending=True
    ).plot.barh(ax=ax2, rot=0, zorder=3)
    ax2.set_title(
        groupby.title() + " (pct. overall)", loc="left", fontweight="bold"
    )
    for ax in [ax1, ax2]:
        ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
        _ = customize_splines(ax)
        _ = add_gridlines(ax)


def plot_grouped_histogram(
    df,
    col_to_plot,
    legend_loc=(0.9, 1.1),
    alpha=0.5,
    wspace=0.2,
    fig_size=(8, 6),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(1, 2, wspace=wspace)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    dfm = df[df["misclassified"]]
    ax1.hist(dfm[dfm["is_default"] == 1][col_to_plot], alpha=alpha, label="M")
    ax1.hist(
        df[df["is_default"] == 1][col_to_plot],
        alpha=alpha,
        label="All",
    )
    ax1.set_title(
        "Defaulted Loans by " + col_to_plot.title(),
        loc="left",
        fontweight="bold",
    )
    ax1.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=legend_loc,
        handletextpad=0.2,
        columnspacing=0.2,
        frameon=False,
    )
    ptitle = f"Paid on-time loans by {col_to_plot.title()}"
    dfm = df[df["misclassified"]]
    ax2.hist(dfm[dfm["is_default"] == 0][col_to_plot], alpha=alpha, label="M")
    ax2.hist(df[df["is_default"] == 0][col_to_plot], alpha=alpha, label="All")
    ax2.set_title(ptitle, loc="left", fontweight="bold")
    ax2.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=legend_loc,
        handletextpad=0.2,
        columnspacing=0.2,
        frameon=False,
    )
    for ax in [ax1, ax2]:
        _ = add_gridlines(ax)


def plot_roc_curve(
    pipe, X_test, y_test, handletextpad=0.5, ax=None, fig_size=(6, 6)
):
    y_score = pipe.predict_proba(X_test)
    y_test_binarized = pd.get_dummies(y_test).to_numpy()
    fpr, tpr, roc_auc = ({} for i in range(3))
    for i in range(y_test.nunique()):
        fpr[i], tpr[i], _ = roc_curve(
            y_test_binarized[:, i],
            y_score[:, i],
            sample_weight=compute_sample_weight(
                class_weight="balanced", y=y_test
            ),
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    if not ax:
        _, ax = plt.subplots(figsize=fig_size)
    for class_val in list(np.sort(y_test.unique())):
        ax.plot(
            fpr[class_val],
            tpr[class_val],
            lw=2,
            label=f"ROC of class {class_val}, AUC = {roc_auc[class_val]:.2f}",
        )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    model_name = type(pipe.named_steps["clf"]).__name__
    ax.set_title(f"Manual ROC (TPR vs FPR) Curves for {model_name}")
    ax.legend(
        loc="lower right",
        frameon=False,
        handletextpad=handletextpad,
    )
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    _ = customize_splines(ax)


def plot_pr_roc_curves(
    y_test,
    y_probs,
    est_name,
    axis_tick_label_fontsize=12,
    wspace=0.1,
    legend_position=(0.35, 1.1),
    f2_beta=2,
    fig_size=(12, 4),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(1, 2, wspace=wspace)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])

    p, r, p_thresholds = mr.precision_recall_curve(y_test, y_probs)
    f2score = ((1 + (f2_beta ** 2)) * p * r) / (((f2_beta ** 2) * p) + r)
    ix = np.argmax(f2score)
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    ax1.plot([0, 1], [no_skill, no_skill], ls="--", label="No Skill")
    ax1.plot(r, p, label=est_name)
    ax1.set_title("Precision-Recall", loc="left", fontweight="bold")
    ax1.annotate(str(np.round(p_thresholds[ix], 3)), (r[ix], p[ix]))
    ax1.scatter(
        r[ix],
        p[ix],
        marker="o",
        color="black",
        label="Best",
        zorder=3,
    )

    fpr, tpr, r_thresholds = mr.roc_curve(y_test, y_probs)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    no_skill = [0, 1]
    ax2.plot(no_skill, no_skill, ls="--", label="No Skill")
    ax2.plot(fpr, tpr, label=est_name)
    ax2.set_title("ROC-AUC", loc="left", fontweight="bold")
    ax2.annotate(str(np.round(r_thresholds[ix], 3)), (fpr[ix], tpr[ix]))
    ax2.scatter(
        fpr[ix],
        tpr[ix],
        marker="o",
        color="black",
        label="Best",
        zorder=3,
    )
    ax2.legend(
        loc="upper left",
        bbox_to_anchor=legend_position,
        columnspacing=0.4,
        handletextpad=0.2,
        frameon=False,
        ncol=3,
    )

    for ax in [ax1, ax2]:
        _ = customize_splines(ax)
        ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.grid(which="both", axis="both", color="lightgrey", zorder=10)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)


def plot_lower_corr_heatmap(
    df_corr,
    ptitle,
    lw=1,
    annot_dict={True: ".2f"},
    ptitle_y_loc=1,
    show_cbar=False,
    cbar_shrink_factor=1,
    fig_size=(10, 10),
):
    _, ax = plt.subplots(figsize=fig_size)
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    sns.heatmap(
        df_corr,
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=list(annot_dict.keys())[0],
        cbar=show_cbar,
        linewidths=lw,
        cbar_kws={"shrink": cbar_shrink_factor},
        fmt=list(annot_dict.values())[0],
    )
    ax.set_title(ptitle, loc="left", fontweight="bold", y=ptitle_y_loc)
    ax.tick_params(left=False, bottom=False)


def plot_single_column_histogram(df, colname, ptitle, fig_size=(8, 4)):
    _, ax = plt.subplots(figsize=fig_size)
    df[colname].plot(kind="hist", ax=ax, lw=1.25, edgecolor="w", label="")
    ax.set_title(ptitle, fontweight="bold", loc="left")
    ax.set_ylabel(None)
    ax.axvline(x=df[colname].median(), label="Median", color="k", ls="--")
    ax.axvline(x=df[colname].mean(), label="Avg", color="r", ls="--")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.76, 1.1),
        ncol=2,
        handletextpad=0.2,
        columnspacing=0.2,
    )


def plot_boxplot_using_quantiles(
    boxes,
    ptitle,
    axis_tick_label_fontsize=12,
    fig_size=(6, 4),
):
    _, ax = plt.subplots(figsize=fig_size)
    bxp1 = ax.bxp(
        boxes,
        positions=[1, 1.5],
        widths=0.35,
        showfliers=False,
        patch_artist=True,
        whiskerprops=dict(linewidth=1.25, color="black"),
        capprops=dict(linewidth=1.25, color="black"),
        boxprops=dict(linewidth=1.25),
        medianprops=dict(linewidth=1.5, color="cyan"),
    )
    for patch in bxp1["boxes"]:
        patch.set(facecolor="steelblue")
    ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    _ = customize_splines(ax)
    _ = add_gridlines(ax)
    ax.set_xlabel(None)
    ax.set_title(
        ptitle,
        loc="left",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))


def plot_multiple_boxplots(
    df,
    x,
    y_s,
    ptitles,
    axis_tick_label_fontsize=12,
    x_ticks_formatter="{x:,.0f}",
    fig_size=(12, 4),
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(1, len(y_s), wspace=0.2)
    for c in range(2):
        ax = fig.add_subplot(grid[0, c])
        sns.boxplot(x=x, y=y_s[c], ax=ax, data=df)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if x_ticks_formatter:
            ax.yaxis.set_major_formatter(
                mtick.StrMethodFormatter(x_ticks_formatter)
            )
        ax.set_title(ptitles[c], loc="left", fontweight="bold")
        ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.grid(True, color="lightgrey")
        ax.spines["bottom"].set_edgecolor("black")
        ax.spines["bottom"].set_linewidth(1.5)


def plot_multi_catplot(
    df,
    x,
    y,
    cat_columns,
    ptitles,
    x_ticks_formatter="{x:,.0f}",
    plot_color="red",
    axis_tick_label_fontsize=12,
    fig_height=4,
    fig_aspect_ratio=1.25,
):
    g = sns.catplot(
        data=df,
        kind="bar",
        x=x,
        col=cat_columns,
        y=y,
        sharey=False,
        palette=sns.color_palette([plot_color]),
        alpha=1,
        height=fig_height,
        aspect=fig_aspect_ratio,
        legend=False,
    )
    for ax, ptitle in zip([g.axes[0][0], g.axes[0][1]], ptitles):
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.set_title(None)
        if x_ticks_formatter:
            ax.yaxis.set_major_formatter(
                mtick.StrMethodFormatter(x_ticks_formatter)
            )
        if ptitle == ptitles[-1]:
            ax.xaxis.set_ticks_position("top")
            ax.yaxis.set_ticks_position("right")
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
        ax.spines["top"].set_edgecolor("black")
        ax.spines["top"].set_linewidth(1.5)
        ax.set_title(ptitle, loc="left", fontweight="bold")
        ax.xaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.set_tick_params(labelsize=axis_tick_label_fontsize)
        ax.yaxis.grid(True, color="lightgrey")
