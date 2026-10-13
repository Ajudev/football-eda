"""Analysis functions, one per question. Pure: DataFrame in, DataFrame out."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from foot_eda.config import PITCH_LENGTH, PITCH_WIDTH

_RATE_COLS = [
    "shots",
    "goals",
    "conversion",
    "ci_low",
    "ci_high",
    "mean_understat_xg",
    "median_distance",
]
BOX_ZONE_ORDER = ["six_yard", "penalty_area", "outside"]


def rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Shots, goals, conversion with Wilson 95% CI, mean xG and median distance per group."""
    if df.empty:
        return pd.DataFrame(columns=[*by, *_RATE_COLS])
    out = (
        df.groupby(by, observed=True)
        .agg(
            shots=("is_goal", "size"),
            goals=("is_goal", "sum"),
            mean_understat_xg=("understat_xg", "mean"),
            median_distance=("distance", "median"),
        )
        .reset_index()
    )
    out["conversion"] = out["goals"] / out["shots"]
    out["ci_low"], out["ci_high"] = proportion_confint(out["goals"], out["shots"], method="wilson")
    return out[[*by, *_RATE_COLS]]


def non_penalty(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["situation"] != "penalty"] if not df.empty else df


def shot_and_goal_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Q1: x, y, is_goal for non-penalty shots."""
    return non_penalty(df).reindex(columns=["x", "y", "is_goal"]).reset_index(drop=True)


def zone_grid(
    df: pd.DataFrame,
    x_bin: float = 5.0,
    y_bin: float = 6.8,
    x_min: float = 52.5,
    min_shots: int = 20,
) -> pd.DataFrame:
    """Q2: conversion per grid cell of the attacking half; cells under min_shots get NaN."""
    d = non_penalty(df)
    d = d[d["x"] >= x_min] if not d.empty else d
    if d.empty:
        return pd.DataFrame(columns=["x0", "y0", *_RATE_COLS, "x1", "y1"])
    n_x = int(np.ceil(round((PITCH_LENGTH - x_min) / x_bin, 9)))
    n_y = int(np.ceil(round(PITCH_WIDTH / y_bin, 9)))
    ix = np.clip(np.floor((d["x"] - x_min) / x_bin), 0, n_x - 1)
    iy = np.clip(np.floor(d["y"] / y_bin), 0, n_y - 1)
    d = d.assign(x0=x_min + ix * x_bin, y0=iy * y_bin)
    out = rate_table(d, ["x0", "y0"])
    out["x1"] = (out["x0"] + x_bin).clip(upper=PITCH_LENGTH)
    out["y1"] = (out["y0"] + y_bin).clip(upper=PITCH_WIDTH)
    out.loc[out["shots"] < min_shots, "conversion"] = np.nan
    return out


def distance_curve(
    df: pd.DataFrame, bin_width: float = 2.0, max_distance: float = 40.0
) -> pd.DataFrame:
    """Q3: conversion per distance bin [d0, d1) for non-penalty shots below max_distance."""
    d = non_penalty(df)
    d = d[d["distance"] < max_distance] if not d.empty else d
    if d.empty:
        return pd.DataFrame(columns=["d0", "d1", *_RATE_COLS])
    d = d.assign(d0=np.floor(d["distance"] / bin_width) * bin_width)
    out = rate_table(d, ["d0"])
    out.insert(1, "d1", out["d0"] + bin_width)
    return out


def fit_distance_logit(
    df: pd.DataFrame,
) -> tuple[pd.Series, Callable[[np.ndarray], np.ndarray]]:
    """Q3: logistic fit of is_goal on distance (non-penalty). Empty params if not fittable."""
    d = non_penalty(df)
    if d.empty or d["is_goal"].nunique() < 2 or d["distance"].nunique() < 2:
        return pd.Series(dtype=float), lambda dist: np.full(np.shape(dist), np.nan)
    model = sm.Logit(d["is_goal"].astype(float), sm.add_constant(d[["distance"]])).fit(disp=0)
    params = model.params

    def predict(dist: np.ndarray) -> np.ndarray:
        z = params["const"] + params["distance"] * np.asarray(dist, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))

    return params, predict


def goal_share_by_box_zone(df: pd.DataFrame, include_penalties: bool) -> pd.DataFrame:
    """Q4: share of goals per box zone, fixed row order."""
    d = df if include_penalties else non_penalty(df)
    cols = ["box_zone", "goals", "share", "shots", "conversion"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    t = rate_table(d, ["box_zone"]).set_index("box_zone").reindex(BOX_ZONE_ORDER)
    t[["shots", "goals"]] = t[["shots", "goals"]].fillna(0).astype(int)
    total = t["goals"].sum()
    t["share"] = t["goals"] / total if total else np.nan
    return t.reset_index()[cols]


def situation_table(df: pd.DataFrame) -> pd.DataFrame:
    return rate_table(df, ["situation"])


def header_table(df: pd.DataFrame) -> pd.DataFrame:
    return rate_table(df, ["is_header"])


def situation_by_header(df: pd.DataFrame) -> pd.DataFrame:
    return rate_table(df, ["situation", "is_header"])
