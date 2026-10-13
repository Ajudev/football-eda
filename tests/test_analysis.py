import numpy as np
import pandas as pd
import pytest

from foot_eda import analysis as a
from foot_eda.features import add_features


@pytest.fixture
def shots() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(60, 105, n)
    y = rng.uniform(5, 63, n)
    sit = rng.choice(["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick", "Penalty"], n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "understat_situation": sit,
            "body_part": rng.choice(["RightFoot", "Head"], n),
            "understat_xg": rng.uniform(0, 0.5, n),
        }
    )
    d = np.sqrt((105 - x) ** 2 + (34 - y) ** 2)
    df["is_goal"] = rng.uniform(size=n) < np.exp(-d / 12) * 0.5
    return add_features(df)


@pytest.fixture
def empty(shots) -> pd.DataFrame:
    return shots.iloc[0:0]


def test_rate_table_counts_and_ci():
    df = pd.DataFrame(
        {
            "g": ["a"] * 4 + ["b"] * 2,
            "is_goal": [True, False, False, False, True, True],
            "understat_xg": 0.1,
            "distance": [10.0, 12.0, 14.0, 16.0, 5.0, 7.0],
        }
    )
    t = a.rate_table(df, ["g"]).set_index("g")
    assert t.loc["a", ["shots", "goals"]].tolist() == [4, 1]
    assert t.loc["a", "conversion"] == 0.25
    assert t.loc["a", "median_distance"] == 13.0
    assert ((t.ci_low <= t.conversion) & (t.conversion <= t.ci_high)).all()


def test_q1_excludes_penalties(shots):
    out = a.shot_and_goal_locations(shots)
    assert list(out.columns) == ["x", "y", "is_goal"]
    assert len(out) == (shots["situation"] != "penalty").sum()


def test_zone_grid_masks_and_excludes_penalties(shots):
    out = a.zone_grid(shots, min_shots=20)
    assert out["shots"].sum() == ((shots.situation != "penalty") & (shots.x >= 52.5)).sum()
    assert out.loc[out.shots < 20, "conversion"].isna().all()
    assert out.loc[out.shots >= 20, "conversion"].notna().all()
    assert (out.x1 > out.x0).all()
    assert (out.y1 <= 68).all()


def test_distance_curve_and_fit(shots):
    curve = a.distance_curve(shots)
    assert curve["shots"].sum() == ((shots.situation != "penalty") & (shots.distance <= 40)).sum()
    params, predict = a.fit_distance_logit(shots)
    assert params["distance"] < 0
    p = predict(np.array([5.0, 30.0]))
    assert p[0] > p[1]


def test_q4_shares(shots):
    for inc in (True, False):
        t = a.goal_share_by_box_zone(shots, include_penalties=inc)
        assert t["box_zone"].tolist() == a.BOX_ZONE_ORDER
        assert abs(t["share"].sum() - 1) < 1e-9
    with_pen = a.goal_share_by_box_zone(shots, True)["shots"].sum()
    assert with_pen == len(shots)
    assert a.goal_share_by_box_zone(shots, False)["shots"].sum() < with_pen


def test_q4_missing_zone_kept():
    df = pd.DataFrame(
        {
            "box_zone": ["outside", "outside"],
            "situation": "open_play",
            "is_goal": [True, False],
            "understat_xg": 0.1,
            "distance": 20.0,
        }
    )
    t = a.goal_share_by_box_zone(df, False)
    assert t["goals"].tolist() == [0, 0, 1]
    assert t["share"].tolist() == [0, 0, 1]


def test_q5_tables(shots):
    assert set(a.situation_table(shots)["situation"]) == {
        "open_play",
        "set_piece",
        "direct_free_kick",
        "penalty",
    }
    assert a.header_table(shots)["shots"].sum() == len(shots)
    assert a.situation_by_header(shots)["shots"].sum() == len(shots)


@pytest.mark.parametrize(
    "fn",
    [
        lambda d: a.rate_table(d, ["situation"]),
        a.shot_and_goal_locations,
        a.zone_grid,
        a.distance_curve,
        lambda d: a.goal_share_by_box_zone(d, True),
        lambda d: a.goal_share_by_box_zone(d, False),
        a.situation_table,
        a.header_table,
        a.situation_by_header,
    ],
)
@pytest.mark.parametrize("make", [lambda e: e, lambda e: pd.DataFrame()])
def test_empty_input(fn, make, empty):
    assert fn(make(empty)).empty


def test_fit_empty(empty):
    params, predict = a.fit_distance_logit(empty)
    assert params.empty
    assert np.isnan(predict(np.array([1.0]))).all()
