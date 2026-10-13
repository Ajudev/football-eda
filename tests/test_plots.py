import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from foot_eda import analysis as a
from foot_eda import plots as p
from foot_eda.features import add_features


@pytest.fixture
def shots() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(60, 105, n)
    y = rng.uniform(5, 63, n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "understat_situation": rng.choice(["OpenPlay", "FromCorner", "DirectFreekick"], n),
            "body_part": rng.choice(["RightFoot", "Head"], n),
            "understat_xg": rng.uniform(0, 0.5, n),
        }
    )
    d = np.sqrt((105 - x) ** 2 + (34 - y) ** 2)
    df["is_goal"] = rng.uniform(size=n) < np.exp(-d / 12) * 0.5
    return add_features(df)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _cases(df: pd.DataFrame):
    _, predict = a.fit_distance_logit(df)
    return {
        "shot_map": lambda: p.plot_shot_map(a.shot_and_goal_locations(df)),
        "zone": lambda: p.plot_zone_conversion(a.zone_grid(df, min_shots=5)),
        "distance": lambda: p.plot_distance_curve(a.distance_curve(df), predict),
        "goal_share": lambda: p.plot_goal_share(a.goal_share_by_box_zone(df, True)),
        "situation": lambda: p.plot_rate_comparison(a.situation_table(df), "situation"),
        "header": lambda: p.plot_rate_comparison(a.header_table(df), "is_header"),
    }


CASES = ["shot_map", "zone", "distance", "goal_share", "situation", "header"]


@pytest.mark.parametrize("name", CASES)
def test_returns_figure(shots, name):
    assert isinstance(_cases(shots)[name](), Figure)


@pytest.mark.parametrize("name", CASES)
def test_empty_returns_placeholder(shots, name):
    fig = _cases(shots.iloc[0:0])[name]()
    assert isinstance(fig, Figure)
    assert [t.get_text() for t in fig.axes[0].texts] == ["No shots for this filter"]


def test_zone_masked_cells_not_annotated(shots):
    grid = a.zone_grid(shots, min_shots=10_000)  # everything masked
    fig = p.plot_zone_conversion(grid)
    assert not fig.axes[0].texts


def test_zone_mixed_masked_cells(shots):
    grid = a.zone_grid(shots, min_shots=10)
    n_ok = int(grid["conversion"].notna().sum())
    assert 0 < n_ok < len(grid)
    ax = p.plot_zone_conversion(grid).axes[0]
    assert len(ax.texts) == n_ok


def test_distance_curve_label_and_fit(shots):
    ax = p.plot_distance_curve(a.distance_curve(shots), a.fit_distance_logit(shots)[1]).axes[0]
    assert ax.get_xlabel() == "Distance to goal centre (m)"
    assert len(ax.lines) >= 1


def test_goal_share_percent_labels(shots):
    ax = p.plot_goal_share(a.goal_share_by_box_zone(shots, True)).axes[0]
    labels = [t.get_text() for t in ax.texts]
    assert labels
    assert all(x.endswith("%") for x in labels)


def test_goal_share_no_goals_placeholder(shots):
    df = shots.assign(is_goal=False)
    fig = p.plot_goal_share(a.goal_share_by_box_zone(df, True))
    assert [t.get_text() for t in fig.axes[0].texts] == ["No goals for this filter"]


def test_shot_map_no_goals(shots):
    assert isinstance(
        p.plot_shot_map(a.shot_and_goal_locations(shots.assign(is_goal=False))), Figure
    )


def test_rate_comparison_n_labels_and_tick_map(shots):
    t = a.header_table(shots)
    ax = p.plot_rate_comparison(t, "is_header", {False: "Foot", True: "Header"}).axes[0]
    assert len([x for x in ax.texts if x.get_text().startswith("n=")]) == len(t)
    assert {x.get_text() for x in ax.get_xticklabels()} <= {"Foot", "Header"}


def test_shot_map_colorbar_does_not_overlap_pitches(shots):
    fig = p.plot_shot_map(a.shot_and_goal_locations(shots))
    cax = fig.axes[-1]
    for ax in fig.axes[:2]:
        assert not cax.get_position().overlaps(ax.get_position())


def test_rate_comparison_partial_labels(shots):
    t = a.situation_table(shots)
    first = t["situation"].iloc[0]
    ax = p.plot_rate_comparison(t, "situation", {first: "Renamed"}).axes[0]
    ticks = [x.get_text() for x in ax.get_xticklabels()]
    assert ticks[0] == "Renamed"
    assert ticks[1:] == [str(v) for v in t["situation"].iloc[1:]]


def test_zone_background_starts_at_halfway(shots):
    grid = a.zone_grid(shots, min_shots=5)
    grid = grid[grid["x0"] > 60]  # no shots in the first rows
    ax = p.plot_zone_conversion(grid).axes[0]
    bg = next(r for r in ax.patches if r.get_zorder() == 1 and hasattr(r, "get_y"))
    assert bg.get_y() == 52.5
