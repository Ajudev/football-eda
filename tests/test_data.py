from datetime import date
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


def test_current_season():
    assert data.current_season(date(2026, 10, 14)) == "2627"
    assert data.current_season(date(2026, 6, 30)) == "2526"
    assert data.current_season(date(2026, 7, 1)) == "2627"
    assert data.current_season(date(2099, 8, 1)) == "9900"


def test_available_seasons():
    s = data.available_seasons(date(2026, 10, 14))
    assert s[0] == "1415" and s[-1] == "2627" and len(s) == 13


@pytest.mark.parametrize(
    "leagues, seasons",
    [
        (["XXX-League"], ["2425"]),
        (config.LEAGUES, ["1314"]),
        (config.LEAGUES, ["2728"]),
        (config.LEAGUES, ["2025"]),
        ([], ["2425"]),
        (config.LEAGUES, []),
    ],
)
def test_validate_selection_rejects(leagues, seasons):
    with pytest.raises(ValueError):
        data.validate_selection(leagues, seasons)


def test_validate_selection_ok():
    data.validate_selection(config.LEAGUES, ["2425"])


EPL, LIGA = "ENG-Premier League", "ESP-La Liga"


def _rows(pairs, start=1, n=2):
    base = data.normalise_shots(pd.read_csv(FIXTURE, dtype={"season": str})).iloc[:1]
    rows, i = [], start
    for league, season in pairs:
        for _ in range(n):
            r = base.copy()
            r["shot_id"], r["league"], r["season"] = i, league, season
            rows.append(r)
            i += 1
    return pd.concat(rows, ignore_index=True)


def test_merge_shots_replaces_only_fetched_pair():
    existing = _rows([(EPL, "2425"), (EPL, "2526"), (LIGA, "2526")])
    new = _rows([(EPL, "2526")], start=100, n=3)
    out = data.merge_shots(existing, new, [EPL], ["2526"])
    counts = out.groupby(["league", "season"]).size().to_dict()
    assert counts == {(EPL, "2425"): 2, (EPL, "2526"): 3, (LIGA, "2526"): 2}
    assert out["shot_id"].is_unique and list(out.columns) == data.SCHEMA_COLUMNS


def test_merge_shots_no_existing():
    new = _rows([(EPL, "2526")])
    out = data.merge_shots(None, new, [EPL], ["2526"])
    assert len(out) == 2 and list(out.columns) == data.SCHEMA_COLUMNS


def _fake_reader(monkeypatch, frames_by_season, calls=None, boom=False):
    import soccerdata as sd

    class Fake:
        def __init__(self, leagues, seasons, no_cache):
            self.league, self.season = leagues, seasons
            if calls is not None:
                calls.append(no_cache)

        def _read_match(self, url, match_id):
            return None

        def read_shot_events(self):
            if boom:
                raise RuntimeError("network down")
            return frames_by_season(self.league, self.season)

    monkeypatch.setattr(sd, "Understat", Fake)


def _raw_for(league, season, n=2, start=1):
    raw = pd.read_csv(FIXTURE, dtype={"season": str}).drop(columns="last_action")
    raw = raw[raw["result"] != "Own Goal"].iloc[:n].copy()
    raw["league"], raw["season"] = league, season
    raw["shot_id"] = range(start, start + n)
    return raw


def test_fetch_keeps_other_seasons(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_PATH", tmp_path / "shots.parquet")
    for season, start in (("2425", 1), ("2526", 100)):
        _fake_reader(monkeypatch, lambda l, s, st=start: _raw_for(l, s, start=st))
        data.fetch_shots([EPL], [season])
    out = data.load_shots()
    assert set(out["season"]) == {"2425", "2526"} and len(out) == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_fetch_failure_leaves_file(tmp_path, monkeypatch):
    path = tmp_path / "shots.parquet"
    monkeypatch.setattr(config, "PROCESSED_PATH", path)
    data._write_atomic(_rows([(EPL, "2425")]))
    before = path.read_bytes()
    _fake_reader(monkeypatch, None, boom=True)
    with pytest.raises(RuntimeError):
        data.fetch_shots([EPL], ["2526"])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_fetch_zero_rows_raises(tmp_path, monkeypatch):
    path = tmp_path / "shots.parquet"
    monkeypatch.setattr(config, "PROCESSED_PATH", path)
    data._write_atomic(_rows([(EPL, "2425")]))
    before = path.read_bytes()
    _fake_reader(monkeypatch, lambda l, s: _raw_for(l, s, n=0))
    with pytest.raises(ValueError, match="no shots"):
        data.fetch_shots([EPL], ["2526"])
    assert path.read_bytes() == before


def test_fetch_refresh_and_invalid_before_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_PATH", tmp_path / "shots.parquet")
    calls: list = []
    _fake_reader(monkeypatch, lambda l, s: _raw_for(l, s), calls=calls)
    with pytest.raises(ValueError):
        data.fetch_shots([EPL], ["1314"])
    assert calls == []
    data.fetch_shots([EPL], ["2526"], refresh=True)
    assert calls == [True]


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    path = tmp_path / "shots.parquet"
    monkeypatch.setattr(config, "PROCESSED_PATH", path)
    data._write_atomic(_rows([(EPL, "2425"), (EPL, "2526"), (LIGA, "2425")], n=3))
    return path


def test_remove_seasons(seeded):
    assert data.remove_seasons([EPL], ["2425"]) == 3
    counts = data.load_shots().groupby(["league", "season"]).size().to_dict()
    assert counts == {(EPL, "2526"): 3, (LIGA, "2425"): 3}


def test_remove_nothing_keeps_mtime(seeded):
    before = seeded.stat().st_mtime_ns
    assert data.remove_seasons([LIGA], ["2526"]) == 0
    assert seeded.stat().st_mtime_ns == before


def test_remove_all_deletes_file(seeded):
    assert data.remove_seasons([EPL, LIGA], ["2425", "2526"]) == 9
    with pytest.raises(FileNotFoundError):
        data.load_shots()


def test_remove_invalid_keeps_file(seeded):
    before = seeded.read_bytes()
    with pytest.raises(ValueError):
        data.remove_seasons(["bad"], ["2425"])
    with pytest.raises(ValueError):
        data.remove_seasons([EPL], ["1314"])
    assert seeded.read_bytes() == before


def test_remove_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_PATH", tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.remove_seasons([EPL], ["2425"])
