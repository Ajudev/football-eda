"""Understat shot ingest.

Spike findings (soccerdata 1.9.1, `sd.Understat(...).read_shot_events()`, checked 2025/26):
- Client used: soccerdata. understatapi fallback not needed.
- Index: league, season, game, team, player. `team` is already the SHOOTING team,
  so no h_team/a_team/h_a logic is needed. `reset_index()` before `normalise_shots`.
- Columns: league_id, season_id, game_id, date, shot_id, team_id, player_id,
  assist_player_id, assist_player, xg, location_x, location_y, minute, body_part,
  situation, result. soccerdata drops `lastAction`; `fetch_shots` captures it from the
  raw match JSON (wrapping `_read_match`) and adds a `last_action` column.
- Labels are prettified: "Right Foot", "Left Foot", "Other", "Open Play", "From Corner",
  "Set Piece", "Direct Freekick", "Missed Shot", "Blocked Shot", "Saved Shot",
  "Shot On Post", "Own Goal", "Goal".
- soccerdata's lookup tables lack Understat's `Head`, `OtherBodyPart` (it has the plural
  `OtherBodyParts`) and `Penalty`, and turn ANY unmapped raw label into NA.
  `_extend_soccerdata_labels` adds them, so NA now means a genuinely unknown label
  and `normalise_shots` raises ValueError on it.
"""

import argparse
import os
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from foot_eda import config

SCHEMA_COLUMNS = [
    "shot_id", "match_id", "league", "season", "date", "team", "player", "minute",
    "x_frac", "y_frac", "x", "y", "body_part", "understat_situation", "last_action",
    "result", "is_goal", "understat_xg",
]  # fmt: skip

_BODY_PART = {
    "Right Foot": "RightFoot",
    "Left Foot": "LeftFoot",
    "Other": "OtherBodyPart",
    "Head": "Head",
}
_SITUATION = {
    "Open Play": "OpenPlay",
    "From Corner": "FromCorner",
    "Set Piece": "SetPiece",
    "Direct Freekick": "DirectFreekick",
    "Penalty": "Penalty",
}
_RESULT = {
    "Goal": "Goal",
    "Missed Shot": "MissedShots",
    "Blocked Shot": "BlockedShot",
    "Saved Shot": "SavedShot",
    "Shot On Post": "ShotOnPost",
}


def _map(s: pd.Series, mapping: dict[Any, Any], name: str) -> pd.Series:
    key = s.astype(object).where(s.notna(), None)
    out = key.map(mapping)
    bad = key[out.isna()].unique()
    if len(bad):
        raise ValueError(f"Unknown {name} value(s): {list(bad)}")
    return out


def normalise_shots(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure: soccerdata `read_shot_events().reset_index()` frame -> shots schema."""
    df = raw[raw["result"] != "Own Goal"].copy()
    out = pd.DataFrame(
        {
            "shot_id": df["shot_id"].astype("int64"),
            "match_id": df["game_id"].astype("int64"),
            "league": df["league"].astype(str),
            "season": df["season"].astype(str),
            "date": pd.to_datetime(df["date"]),
            "team": df["team"].astype(str),
            "player": df["player"].astype(str),
            "minute": df["minute"].astype("int64"),
            "x_frac": df["location_x"].astype("float64"),
            "y_frac": df["location_y"].astype("float64"),
            "body_part": _map(df["body_part"], _BODY_PART, "body_part"),
            "understat_situation": _map(df["situation"], _SITUATION, "situation"),
            "last_action": df.get("last_action"),
            "result": _map(df["result"], _RESULT, "result"),
            "understat_xg": df["xg"].astype("float64"),
        }
    )
    out["x"] = out["x_frac"] * config.PITCH_LENGTH
    out["y"] = out["y_frac"] * config.PITCH_WIDTH
    out["is_goal"] = out["result"] == "Goal"
    return out[SCHEMA_COLUMNS].reset_index(drop=True)


def _extend_soccerdata_labels() -> None:
    """Teach soccerdata the Understat labels it maps to NA (see module docstring)."""
    from soccerdata import understat

    understat.SHOT_BODY_PARTS.update({"Head": "Head", "OtherBodyPart": "Other"})
    understat.SHOT_SITUATIONS.update({"Penalty": "Penalty"})


def current_season(today: date | None = None) -> str:
    """Season code such as "2627"; a season starts in July."""
    today = today or date.today()
    y = today.year if today.month >= 7 else today.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def available_seasons(today: date | None = None) -> list[str]:
    """Every season code from "1415" to the current season, oldest first."""
    last = current_season(today)
    out, y = [], 2014
    while True:
        code = f"{y % 100:02d}{(y + 1) % 100:02d}"
        out.append(code)
        if code == last:
            return out
        y += 1


def validate_selection(leagues: list[str], seasons: list[str]) -> None:
    """Trust boundary for user-chosen leagues/seasons. Raises ValueError."""
    if not leagues:
        raise ValueError("No leagues selected.")
    if not seasons:
        raise ValueError("No seasons selected.")
    for league in leagues:
        if league not in config.LEAGUES:
            raise ValueError(f"Unknown league: {league!r}")
    valid = available_seasons()
    for season in seasons:
        if season not in valid:
            raise ValueError(f"Unknown season: {season!r}")


def merge_shots(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    leagues: list[str],
    seasons: list[str],
) -> pd.DataFrame:
    """Pure: replace the (league, season) pairs in leagues x seasons with `new`."""
    if existing is None:
        kept = existing
    else:
        replaced = existing["league"].isin(leagues) & existing["season"].isin(seasons)
        kept = existing[~replaced]
    df = pd.concat([f for f in (kept, new) if f is not None], ignore_index=True)
    df = df.drop_duplicates("shot_id", keep="last")
    return df[SCHEMA_COLUMNS].reset_index(drop=True)


def _write_atomic(df: pd.DataFrame) -> None:
    path = config.PROCESSED_PATH
    tmp = path.with_suffix(".parquet.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def remove_seasons(leagues: list[str], seasons: list[str]) -> int:
    """Remove (league, season) rows from the parquet. Returns rows removed."""
    validate_selection(leagues, seasons)
    df = load_shots()
    gone = df["league"].isin(leagues) & df["season"].isin(seasons)
    n = int(gone.sum())
    if n == 0:
        return 0
    rest = df[~gone]
    if rest.empty:
        config.PROCESSED_PATH.unlink()
    else:
        _write_atomic(rest.reset_index(drop=True))
    return n


def fetch_shots(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch league/seasons from Understat and merge them into the parquet file.

    Returns the fetched rows only. Any failure before the write leaves the file untouched.
    """
    leagues = config.LEAGUES if leagues is None else leagues
    seasons = config.SEASONS if seasons is None else seasons
    validate_selection(leagues, seasons)
    import soccerdata as sd  # lazy: keeps normalise/load importable without the client

    _extend_soccerdata_labels()
    frames = []
    for league in leagues:
        for season in seasons:
            reader = sd.Understat(leagues=league, seasons=season, no_cache=refresh)
            last_action: dict[int, str | None] = {}
            read_match = reader._read_match

            def capture(
                url: str,
                match_id: int,
                _read: Callable[[str, int], dict[str, Any] | None] = read_match,
                _store: dict[int, str | None] = last_action,
            ) -> dict[str, Any] | None:
                match = _read(url, match_id)
                for side in (match or {}).get("shotsData", {}).values():
                    for shot in side:
                        _store[int(shot["id"])] = shot.get("lastAction")
                return match

            reader._read_match = capture
            raw = reader.read_shot_events().reset_index()
            raw["last_action"] = raw["shot_id"].map(last_action)
            frames.append(normalise_shots(raw))
    fetched = pd.concat(frames, ignore_index=True).drop_duplicates("shot_id").reset_index(drop=True)
    if fetched.empty:
        raise ValueError(f"Understat returned no shots for {leagues} {seasons}")
    existing = load_shots() if config.PROCESSED_PATH.exists() else None
    _write_atomic(merge_shots(existing, fetched, leagues, seasons))
    return fetched


def load_shots() -> pd.DataFrame:
    if not config.PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_PATH} not found. Run: uv run python -m foot_eda.data"
        )
    return pd.read_parquet(config.PROCESSED_PATH)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Understat shots and merge into parquet")
    p.add_argument("--refresh", action="store_true", help="bypass soccerdata cache")
    p.add_argument("--league", action="append", help="repeatable; default: all configured")
    p.add_argument("--season", action="append", help="repeatable, e.g. 2425; default: config")
    a = p.parse_args()
    df = fetch_shots(a.league, a.season, refresh=a.refresh)
    print(f"fetched {len(df)} shots; {len(load_shots())} total in {config.PROCESSED_PATH}")


if __name__ == "__main__":
    main()
