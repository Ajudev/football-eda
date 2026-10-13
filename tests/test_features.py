import pandas as pd
import pytest

from foot_eda.features import add_features


def _frame(points, situation="OpenPlay", body_part="RightFoot"):
    return pd.DataFrame(
        {
            "x": [p[0] for p in points],
            "y": [p[1] for p in points],
            "understat_situation": situation,
            "body_part": body_part,
        }
    )


@pytest.mark.parametrize(
    ("x", "y", "zone"),
    [
        (99.5, 24.84, "six_yard"),
        (99.5, 43.16, "six_yard"),
        (99.4, 34, "penalty_area"),
        (99.5, 24.8, "penalty_area"),
        (88.5, 13.84, "penalty_area"),
        (88.5, 54.16, "penalty_area"),
        (88.4, 34, "outside"),
        (95, 13.8, "outside"),
    ],
)
def test_box_zone_boundaries(x, y, zone):
    assert add_features(_frame([(x, y)]))["box_zone"].iloc[0] == zone


def test_distance():
    out = add_features(_frame([(105, 34), (95, 34)]))
    assert out["distance"].tolist() == [0, 10]


def test_situation_mapping():
    df = _frame([(100, 34)] * 5)
    df["understat_situation"] = ["Penalty", "DirectFreekick", "FromCorner", "SetPiece", "OpenPlay"]
    assert add_features(df)["situation"].tolist() == [
        "penalty",
        "direct_free_kick",
        "set_piece",
        "set_piece",
        "open_play",
    ]


def test_unknown_situation_raises():
    with pytest.raises(ValueError, match="Mystery"):
        add_features(_frame([(100, 34)], situation="Mystery"))


def test_corner_header():
    out = add_features(_frame([(100, 34)], situation="FromCorner", body_part="Head"))
    assert out["situation"].iloc[0] == "set_piece"
    assert bool(out["is_header"].iloc[0]) is True


def test_does_not_mutate_input():
    df = _frame([(100, 34)])
    add_features(df)
    assert "distance" not in df.columns
