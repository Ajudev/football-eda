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


CASES = list(_cases(pd.DataFrame(columns=["x"])))


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
