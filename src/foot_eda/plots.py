"""Plot functions, one per question. Draw frames from analysis.py; no aggregation here."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch

from foot_eda.config import PITCH_LENGTH, PITCH_WIDTH

_NO_DATA = "No shots for this filter"


def _pitch() -> VerticalPitch:
    return VerticalPitch(
        pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, half=True
    )


def _yerr(t: pd.DataFrame) -> np.ndarray:
    # Clip: float rounding can make Wilson bounds cross conversion by ~1e-17.
    return np.clip(
        np.vstack([t["conversion"] - t["ci_low"], t["ci_high"] - t["conversion"]]), 0, None
    )


def _placeholder() -> Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, _NO_DATA, ha="center", va="center", fontsize=14, transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def plot_shot_map(locs: pd.DataFrame) -> Figure:
    """Q1: hexbin shot density (left) and goal scatter (right)."""
    if locs.empty:
        return _placeholder()
    pitch = _pitch()
    fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(12, 7))
    hb = pitch.hexbin(
        locs["x"], locs["y"], ax=axs[0], gridsize=(12, 8), cmap="Blues", edgecolors="none"
    )
    fig.colorbar(hb, ax=axs[0], shrink=0.6, label="Shots")
    axs[0].set_title("Shot density")
    goals = locs[locs["is_goal"].astype(bool)]
    pitch.scatter(
        goals["x"], goals["y"], ax=axs[1], s=25, c="crimson", edgecolors="black", alpha=0.7
    )
    axs[1].set_title(f"Goals (n={len(goals)})")
    return fig


def plot_zone_conversion(grid: pd.DataFrame) -> Figure:
    """Q2: conversion heatmap per cell, annotated goals/shots; masked cells grey and blank."""
    if grid.empty:
        return _placeholder()
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=(7, 8))
    vmax = float(grid["conversion"].max()) if grid["conversion"].notna().any() else 1.0
    norm, cmap = Normalize(vmin=0, vmax=vmax or 1.0), plt.get_cmap("YlOrRd")
    for r in grid.itertuples():
        masked = pd.isna(r.conversion)
        # Vertical pitch: horizontal axis is y, vertical axis is x.
        ax.add_patch(
            Rectangle(
                (r.y0, r.x0),
                r.y1 - r.y0,
                r.x1 - r.x0,
                facecolor="lightgrey" if masked else cmap(norm(r.conversion)),
                edgecolor="white",
                alpha=0.9,
                zorder=2,
            )
        )
        if not masked:
            ax.text(
                (r.y0 + r.y1) / 2,
                (r.x0 + r.x1) / 2,
                f"{int(r.goals)}/{int(r.shots)}",
                ha="center",
                va="center",
                fontsize=7,
                zorder=3,
            )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Conversion")
    ax.set_title("Conversion by zone (grey = too few shots)")
    return fig


def plot_distance_curve(curve: pd.DataFrame, predict: Callable[[np.ndarray], np.ndarray]) -> Figure:
    """Q3: binned conversion with Wilson CI bars plus the logistic fit."""
    if curve.empty:
        return _placeholder()
    fig, ax = plt.subplots(figsize=(8, 5))
    mid = (curve["d0"] + curve["d1"]) / 2
    yerr = _yerr(curve)
    ax.errorbar(mid, curve["conversion"], yerr=yerr, fmt="o", capsize=3, label="Binned (95% CI)")
    xs = np.linspace(float(curve["d0"].min()), float(curve["d1"].max()), 100)
    ys = np.asarray(predict(xs), dtype=float)
    if not np.isnan(ys).all():
        ax.plot(xs, ys, color="crimson", label="Logistic fit")
    ax.set_xlabel("Distance to goal centre (m)")
    ax.set_ylabel("Conversion")
    ax.legend()
    return fig


def plot_goal_share(table: pd.DataFrame) -> Figure:
    """Q4: share of goals per box zone, labelled with percentages."""
    if table.empty:
        return _placeholder()
    fig, ax = plt.subplots(figsize=(6, 4))
    share = table["share"].fillna(0)
    bars = ax.bar(table["box_zone"], share)
    ax.bar_label(bars, labels=[f"{s:.1%}" for s in share])
    ax.set_ylim(0, max(float(share.max()) * 1.15, 0.1))
    ax.set_ylabel("Share of goals")
    return fig


def plot_rate_comparison(table: pd.DataFrame, group_col: str) -> Figure:
    """Q5: conversion with CI bars per group, each labelled with its shot count."""
    if table.empty:
        return _placeholder()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = table[group_col].astype(str)
    ax.bar(labels, table["conversion"], yerr=_yerr(table), capsize=4)
    for i, (hi, n) in enumerate(zip(table["ci_high"], table["shots"], strict=True)):
        ax.annotate(f"n={int(n)}", (i, hi), xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_ylim(0, float(table["ci_high"].max()) * 1.15)
    ax.set_ylabel("Conversion")
    ax.set_xlabel(group_col)
    return fig
