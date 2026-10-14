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
    at.multiselect(key="f_league").set_value(["ESP-La Liga"]).run()
    assert not at.exception
    assert at.metric[0].value == "150"


def test_empty_filter_warns(app):
    at = app.run()
    at.multiselect(key="f_league").set_value([]).run()
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
    assert at.multiselect(key="f_team").options == ["Arsenal", "Barcelona"]
    at.multiselect(key="f_team").set_value(["Arsenal"]).run()
    assert not at.exception
    assert at.metric[0].value == "150"
    at.multiselect(key="f_league").set_value(["ESP-La Liga"]).run()
    assert at.multiselect(key="f_team").options == ["Barcelona"]


def test_season_defaults_to_latest(monkeypatch):
    def two_seasons():
        old = _raw().assign(season="2425", shot_id=lambda d: d["shot_id"] + 1000)
        return pd.concat([old, _raw()], ignore_index=True)

    monkeypatch.setattr(data, "load_shots", two_seasons)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.multiselect(key="f_season").value == ["2526"]
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


# --- Data section (Phase 5b) ---


def _fake_fetch(calls, exc=None):
    def fake(leagues=None, seasons=None, refresh=False):
        calls.append({"leagues": leagues, "seasons": seasons, "refresh": refresh})
        if exc:
            raise exc
        return _raw()

    return fake


def _click(at, key):
    at.button(key=key).click().run()
    return at


def test_import_calls_fetch_and_shows_success(app, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "fetch_shots", _fake_fetch(calls))
    at = app.run()
    at.selectbox(key="import_season").set_value("2425")
    _click(at, "import_btn")
    assert calls == [
        {"leagues": at.multiselect(key="data_leagues").value, "seasons": ["2425"], "refresh": False}
    ]
    assert calls[0]["leagues"]
    assert not at.exception
    assert "Imported 300 shots for 2425" in at.success[0].value


def test_refresh_current_season_and_force(app, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "fetch_shots", _fake_fetch(calls))
    at = app.run()
    _click(at, "refresh_btn")
    at.checkbox(key="force_refresh").check()
    _click(at, "refresh_btn")
    assert [(c["seasons"], c["refresh"]) for c in calls] == [
        ([data.current_season()], False),
        ([data.current_season()], True),
    ]


def test_value_error_shown_and_tabs_render(app, monkeypatch):
    monkeypatch.setattr(data, "fetch_shots", _fake_fetch([], ValueError("bad season")))
    at = _click(app.run(), "import_btn")
    assert any("bad season" in e.value for e in at.error)
    assert len(at.tabs) == 5


def test_network_error_keeps_old_data(app, monkeypatch):
    monkeypatch.setattr(data, "fetch_shots", _fake_fetch([], ConnectionError("down")))
    at = _click(app.run(), "import_btn")
    assert any("Fetch failed" in e.value for e in at.error)
    assert at.metric[0].value == "300"


def test_missing_file_keeps_import_button(monkeypatch):
    def boom():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(data, "load_shots", boom)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.button(key="import_btn")
    assert not at.exception


@pytest.fixture
def _clear_resource():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_lock_held_warns_and_skips_fetch(app, monkeypatch, _clear_resource):
    import sys
    import threading

    held = threading.Lock()
    held.acquire()
    real = threading.Lock

    def lock_factory():
        # Only the app's `_fetch_lock` gets the pre-acquired lock.
        return held if sys._getframe(1).f_code.co_filename == APP else real()

    monkeypatch.setattr(threading, "Lock", lock_factory)
    calls = []
    monkeypatch.setattr(data, "fetch_shots", _fake_fetch(calls))
    at = _click(app.run(), "import_btn")
    assert any("already running" in w.value for w in at.warning)
    assert calls == []


def test_import_options_exclude_current_season(app):
    assert data.current_season() not in app.run().selectbox(key="import_season").options


def test_remove_requires_confirm_and_calls_remove(app, monkeypatch):
    calls = []

    def fake_remove(leagues, seasons):
        calls.append((leagues, seasons))
        return 5

    monkeypatch.setattr(data, "remove_seasons", fake_remove)
    at = app.run()
    assert at.button(key="remove_btn").disabled
    at.checkbox(key="confirm_remove_2526").check().run()
    assert not at.button(key="remove_btn").disabled
    _click(at, "remove_btn")
    assert calls == [(at.multiselect(key="data_leagues").value, ["2526"])]
    assert "Removed 5 shots for 2526" in at.success[0].value


def test_remove_options_match_data(app):
    assert app.run().selectbox(key="remove_season").options == ["2526"]


def test_remove_zero_rows_shows_info(app, monkeypatch):
    monkeypatch.setattr(data, "remove_seasons", lambda leagues, seasons: 0)
    at = app.run()
    at.checkbox(key="confirm_remove_2526").check().run()
    _click(at, "remove_btn")
    assert any("nothing removed" in i.value for i in at.info)
    assert not at.success


def _hold_lock(monkeypatch):
    import sys
    import threading

    held = threading.Lock()
    held.acquire()
    real = threading.Lock
    monkeypatch.setattr(
        threading, "Lock", lambda: held if sys._getframe(1).f_code.co_filename == APP else real()
    )


def test_remove_confirmation_resets_after_remove(app, monkeypatch):
    monkeypatch.setattr(data, "remove_seasons", lambda leagues, seasons: 5)
    at = app.run()
    at.checkbox(key="confirm_remove_2526").check().run()
    _click(at, "remove_btn")
    assert not at.checkbox(key="confirm_remove_2526").value
    assert at.button(key="remove_btn").disabled


def test_missing_file_hides_remove_widgets(monkeypatch):
    def boom():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(data, "load_shots", boom)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not any(b.key == "remove_btn" for b in at.button)
    assert not any(s.key == "remove_season" for s in at.selectbox)


def test_remove_lock_held_warns_and_skips(app, monkeypatch, _clear_resource):
    _hold_lock(monkeypatch)
    calls = []
    monkeypatch.setattr(data, "remove_seasons", lambda *a: calls.append(a) or 1)
    at = app.run()
    at.checkbox(key="confirm_remove_2526").check().run()
    _click(at, "remove_btn")
    assert any("already running" in w.value for w in at.warning)
    assert calls == []


def test_remove_error_shown_and_keeps_metrics(app, monkeypatch):
    def boom(leagues, seasons):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(data, "remove_seasons", boom)
    at = app.run()
    at.checkbox(key="confirm_remove_2526").check().run()
    _click(at, "remove_btn")
    assert any("Remove failed" in e.value for e in at.error)
    assert at.metric[0].value == "300"
