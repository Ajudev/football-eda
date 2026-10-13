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
