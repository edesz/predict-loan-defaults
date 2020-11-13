#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn as sns


def plot_returns(
    df_t_tuned,
    ptitle,
    annotation_text,
    axis_tick_fontsize=12,
    annotation_text_fontsize=10,
    annotation_loc=(0.99, 0.01),
    fig_size=(8, 4),
):
    g = sns.catplot(
        y="cfg",
        x="return",
        hue="clf",
        data=df_t_tuned,
        kind="bar",
        legend=False,
        height=fig_size[1],
        aspect=fig_size[0] / fig_size[1],
    )
    ax = g.axes[0, 0]
    ax.set_ylabel(None)
    ax.axvline(x=0, ls="--", lw=1.5, c="k")
    ax.set_title(
        ptitle,
        loc="left",
        fontweight="bold",
    )
    ax.get_xaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        handletextpad=0.25,
        prop={"size": axis_tick_fontsize},
    )
    ax.set_xlabel(None)
    ax.xaxis.set_tick_params(labelsize=axis_tick_fontsize)
    ax.yaxis.set_tick_params(labelsize=axis_tick_fontsize)
    ax.grid(which="both", axis="both", color="lightgrey", zorder=0)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    sns.despine(
        fig=None,
        ax=None,
        top=False,
        right=False,
        left=False,
        bottom=False,
        offset=None,
        trim=False,
    )
    bbox_props = dict(boxstyle=None, pad=0, fc="None", ec=None, lw=0)
    ax.annotate(
        xy=annotation_loc,
        xycoords=ax.transAxes,
        text=annotation_text,
        ha="right",
        va="bottom",
        rotation=0,
        size=annotation_text_fontsize,
        bbox=bbox_props,
    )
