#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import altair as alt
import pandas as pd


def plot_heatmap(
    df, ptitle, x, y, xtitle="", ytitle="", annot_fmt=".3f", ptitle_offset=-5
):
    if ptitle != "":
        ptitle = alt.TitleParams(ptitle, offset=ptitle_offset)
    base = (
        alt.Chart(df, title=ptitle)
        .mark_rect()
        .encode(
            x=alt.X(x, title=xtitle),
            y=alt.Y(y, title=ytitle),
        )
    )
    heatmap = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "value:Q",
            scale=alt.Scale(type="log", scheme="yelloworangered"),
            legend=None,
        ),
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("value:Q", format=annot_fmt),
        color=alt.condition(
            alt.datum.value > df["value"].mean(),
            alt.value("white"),
            alt.value("black"),
        ),
    )
    crchart = heatmap + text
    return crchart


def plot_altair_grid(
    df_t_cm,
    df_cr,
    ptitle_offset=-5,
    cpe_figsize=(200, 300),
    cm_figsize=(200, 300),
    cr_figsize=[(300, 300), (100, 300)],
):
    df_cm_reshaped = (
        df_t_cm.unstack().reset_index().rename(columns={0: "value"})
    )
    # display(df_cm_reshaped)
    df_cr_metrics_reshaped = (
        df_cr.iloc[0:2][["precision", "recall", "f1-score"]]
        .unstack()
        .reset_index()
        .rename(columns={"level_0": "metric", "level_1": "class", 0: "value"})
    )
    # display(df_cr_metrics_reshaped)
    df_cr_support_reshaped = (
        df_cr.iloc[0:2][["support"]]
        .unstack()
        .reset_index()
        .rename(columns={"level_0": "metric", "level_1": "class", 0: "value"})
    )
    # display(df_cr_support_reshaped)

    y_axis_title = "number of predicted class"
    cpe_chart = (
        alt.Chart(
            df_cm_reshaped,
            title=alt.TitleParams(
                "Class Prediction Error", offset=ptitle_offset - 2
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("predicted:N", axis=alt.Axis(title="")),
            y=alt.Y("sum(value):Q", axis=alt.Axis(title=y_axis_title)),
            color=alt.Color(
                "actual:N",
                legend=alt.Legend(offset=3.5),
            ),
            tooltip=[
                alt.Tooltip("actual", title="Actual"),
                alt.Tooltip("predicted", title="Predicted"),
                alt.Tooltip("sum(value)", title="Number", format=",.0f"),
            ],
        )
        .properties(width=cpe_figsize[0], height=cpe_figsize[1])
    )

    cm_chart = plot_heatmap(
        df_cm_reshaped,
        ptitle="Confusion Matrix",
        x="predicted:N",
        y="actual:N",
        xtitle="Predicted Class",
        ytitle="Actual Class",
        annot_fmt=",.0f",
        ptitle_offset=ptitle_offset,
    ).properties(width=cpe_figsize[0], height=cpe_figsize[1])

    cr_metrics = plot_heatmap(
        df_cr_metrics_reshaped,
        ptitle="Classification Report",
        x="metric:N",
        y="class:N",
        xtitle="",
        ytitle="",
        annot_fmt=".3f",
        ptitle_offset=ptitle_offset,
    ).properties(width=cr_figsize[0][0], height=cr_figsize[0][1])
    cr_support = plot_heatmap(
        df_cr_support_reshaped,
        ptitle="",
        x="metric:N",
        y="class:N",
        xtitle="",
        ytitle="",
        annot_fmt=",.0f",
    ).properties(width=cr_figsize[1][0], height=cr_figsize[1][1])
    cr_chart = (cr_metrics + cr_support).resolve_scale(color="independent")
    combo = alt.hconcat(cpe_chart, cm_chart, cr_chart).configure_title(
        fontSize=12, anchor="middle"
    )
    return combo


def alt_plot_metric_based_threshold_tuning_plots(
    df, ptitle_offset=-5, legend_offset=5, figsize=(450, 300)
):
    highlight = alt.selection(
        type="single", on="mouseover", fields=["metric"], nearest=True
    )
    base = alt.Chart(
        df,
        title=alt.TitleParams(
            "Scoring Metrics, as threshold is changed", offset=ptitle_offset
        ),
    ).encode(
        x=alt.X("threshold:Q", title="threshold"),
        y=alt.Y("value:Q", title=""),
        color=alt.Color(
            "metric:N", legend=alt.Legend(offset=legend_offset, title="")
        ),
        tooltip=[
            alt.Tooltip("metric", title="Metric"),
            alt.Tooltip("threshold", title="Threshold", format=".2f"),
            alt.Tooltip("value", title="Value", format=".2f"),
        ],
    )

    overlay = pd.DataFrame({"default": [0.5]})
    rules = (
        alt.Chart(overlay).mark_rule().encode(x=alt.X("default:Q", title=""))
    )

    points_opacity = alt.value(0)
    points = (
        base.mark_circle()
        .encode(opacity=points_opacity)
        .add_selection(highlight)
    )

    lines = base.mark_line().encode(
        size=alt.condition(~highlight, alt.value(1.5), alt.value(3))
    )

    combo = (points + lines + rules).properties(
        width=figsize[0], height=figsize[1]
    )
    return combo
