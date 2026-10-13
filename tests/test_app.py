import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from foot_eda import data

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    league = np.where(np.arange(n) % 2 == 0, "ENG-Premier League", "ESP-La Liga")
    return pd.DataFrame(
        {
            "shot_id": np.arange(n),
            "league": league,
            "season": "2526",
            "date": pd.Timestamp("2025-09-01") + pd.to_timedelta(rng.integers(0, 60, n), "D"),
            "team": np.where(league == "ENG-Premier League", "Arsenal", "Barcelona"),
            "x": rng.uniform(80, 105, n),
            "y": rng.uniform(15, 53, n),
            "body_part": rng.choice(["RightFoot", "Head"], n),
            "understat_situation": rng.choice(["OpenPlay", "FromCorner", "Penalty"], n),
            "result": rng.choice(["Goal", "SavedShot"], n),
            "understat_xg": rng.uniform(0, 0.5, n),
        }
    ).assign(is_goal=lambda d: d["result"] == "Goal")


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(data, "load_shots", _raw)
    return AppTest.from_file(APP, default_timeout=60)


def test_renders(app):
    at = app.run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert not at.warning
    assert at.metric[0].value == "300"


def test_league_filter_changes_shots(app):
    at = app.run()
    at.multiselect[0].set_value(["ESP-La Liga"]).run()
    assert not at.exception
    assert at.metric[0].value == "150"


def test_empty_filter_warns(app):
    at = app.run()
    at.multiselect[0].set_value([]).run()
    assert not at.exception
    assert len(at.warning) == 1


def test_missing_file_shows_error(monkeypatch):
    def boom():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(data, "load_shots", boom)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.error) == 1


def test_team_filter(app):
    at = app.run()
    assert at.multiselect[2].options == ["Arsenal", "Barcelona"]
    at.multiselect[2].set_value(["Arsenal"]).run()
    assert not at.exception
    assert at.metric[0].value == "150"
    at.multiselect[0].set_value(["ESP-La Liga"]).run()
    assert at.multiselect[2].options == ["Barcelona"]


def test_season_defaults_to_latest(monkeypatch):
    def two_seasons():
        old = _raw().assign(season="2425", shot_id=lambda d: d["shot_id"] + 1000)
        return pd.concat([old, _raw()], ignore_index=True)

    monkeypatch.setattr(data, "load_shots", two_seasons)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.multiselect[1].value == ["2526"]
    assert at.metric[0].value == "300"


def test_missing_file_error_names_fetch_command(monkeypatch):
    def boom():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(data, "load_shots", boom)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert "uv run python -m foot_eda.data" in at.error[0].value


def test_q5_info_header_labels_and_credit(app):
    at = app.run()
    assert "Counter-attacks" in at.info[0].value
    assert any("Understat" in c.value for c in at.caption)


def test_large_q1_table_does_not_crash(monkeypatch):
    # Styler limit is 262,144 cells = 87,381 rows x 3 cols.
    big = pd.concat([_raw()] * 300, ignore_index=True).assign(
        understat_situation="OpenPlay", result="Goal"
    )
    big["is_goal"] = True
    monkeypatch.setattr(data, "load_shots", lambda: big)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception


def test_header_table_uses_labels(app):
    at = app.run()
    tables = [d.value for d in at.dataframe]
    assert any(
        "is_header" in t.columns and set(t["is_header"]) <= {"Header", "Non-header"} for t in tables
    )


def test_cache_reloads_when_parquet_changes(monkeypatch, tmp_path):
    from foot_eda import config

    path = tmp_path / "shots.parquet"
    path.write_bytes(b"")
    calls = []

    def counted():
        calls.append(1)
        return _raw()

    monkeypatch.setattr(config, "PROCESSED_PATH", path)
    monkeypatch.setattr(data, "load_shots", counted)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.run()
    assert len(calls) == 1
    os.utime(path, (1_000_000, 1_000_000))
    at.run()
    assert len(calls) == 2
    assert not at.exception
