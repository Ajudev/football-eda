from pathlib import Path

import pandas as pd
import pytest

from foot_eda import config, data

FIXTURE = Path(__file__).parent / "fixtures" / "understat_shots_sample.csv"


@pytest.fixture
def out() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, dtype={"season": str})
    return data.normalise_shots(raw)


def test_schema_and_own_goal_dropped(out):
    assert list(out.columns) == data.SCHEMA_COLUMNS
    assert len(out) == 6
    assert 1005 not in set(out["shot_id"])


def test_team_home_and_away(out):
    teams = out.set_index("shot_id")["team"]
    assert teams[1001] == "Liverpool"  # home
    assert teams[1002] == "Bournemouth"  # away


def test_coordinates_in_metres(out):
    assert (out["x"] == out["x_frac"] * config.PITCH_LENGTH).all()
    assert (out["y"] == out["y_frac"] * config.PITCH_WIDTH).all()


def test_is_goal_only_for_goal(out):
    assert (out["is_goal"] == (out["result"] == "Goal")).all()
    assert out["is_goal"].sum() == 2


def test_explicit_head_penalty_other_labels(out):
    r = out.set_index("shot_id")
    assert r.loc[1003, "body_part"] == "Head"
    assert r.loc[1004, "understat_situation"] == "Penalty"
    assert r.loc[1006, "body_part"] == "OtherBodyPart"
    assert r.loc[1003, "last_action"] == "Aerial"
    assert set(out["body_part"]) <= {"RightFoot", "LeftFoot", "Head", "OtherBodyPart"}


@pytest.mark.parametrize("col", ["body_part", "situation"])
def test_na_label_raises(col):
    raw = pd.read_csv(FIXTURE, dtype={"season": str})
    raw.loc[0, col] = pd.NA
    with pytest.raises(ValueError, match=col):
        data.normalise_shots(raw)


def test_fetch_shots_dedups_refresh_and_writes(tmp_path, monkeypatch):
    import soccerdata as sd

    raw = pd.read_csv(FIXTURE, dtype={"season": str})
    calls = []

    class FakeUnderstat:
        def __init__(self, leagues, seasons, no_cache):
            calls.append(no_cache)

        def _read_match(self, url, match_id):
            return None

        def read_shot_events(self):
            return raw.drop(columns="last_action")

    monkeypatch.setattr(sd, "Understat", FakeUnderstat)
    monkeypatch.setattr(config, "PROCESSED_PATH", tmp_path / "p" / "shots.parquet")
    df = data.fetch_shots(refresh=True)
    assert calls == [True] * len(config.LEAGUES) * len(config.SEASONS)
    assert df["shot_id"].is_unique and len(df) == 6
    assert config.PROCESSED_PATH.exists()


def test_unknown_label_raises():
    raw = pd.read_csv(FIXTURE, dtype={"season": str})
    raw.loc[0, "result"] = "Mystery"
    with pytest.raises(ValueError, match="result"):
        data.normalise_shots(raw)


def test_load_shots_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_PATH", tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError, match="foot_eda.data"):
        data.load_shots()
