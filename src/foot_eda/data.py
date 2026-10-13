"""Understat shot ingest.

Spike findings (soccerdata 1.9.1, `sd.Understat(...).read_shot_events()`, checked 2025/26):
- Client used: soccerdata. understatapi fallback not needed.
- Index: league, season, game, team, player. `team` is already the SHOOTING team,
  so no h_team/a_team/h_a logic is needed. `reset_index()` before `normalise_shots`.
- Columns: league_id, season_id, game_id, date, shot_id, team_id, player_id,
  assist_player_id, assist_player, xg, location_x, location_y, minute, body_part,
  situation, result. There is NO `last_action` column (schema column is left null).
- Labels are prettified: "Right Foot", "Left Foot", "Other", "Open Play", "From Corner",
  "Set Piece", "Direct Freekick", "Missed Shot", "Blocked Shot", "Saved Shot",
  "Shot On Post", "Own Goal", "Goal".
- soccerdata's lookup tables lack `Head` and `Penalty`, so they come back as NA:
  body_part NA -> Head (xG ~0.14, mostly corners/set pieces),
  situation NA -> Penalty (all at x=0.885, y=0.5, xG ~0.76, ~84% goals).
  Any other unknown label raises ValueError.
"""

import argparse

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
    None: "Head",
}
_SITUATION = {
    "Open Play": "OpenPlay",
    "From Corner": "FromCorner",
    "Set Piece": "SetPiece",
    "Direct Freekick": "DirectFreekick",
    None: "Penalty",
}
_RESULT = {
    "Goal": "Goal",
    "Missed Shot": "MissedShots",
    "Blocked Shot": "BlockedShot",
    "Saved Shot": "SavedShot",
    "Shot On Post": "ShotOnPost",
}


def _map(s: pd.Series, mapping: dict, name: str) -> pd.Series:
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
            "last_action": None,
            "result": _map(df["result"], _RESULT, "result"),
            "understat_xg": df["xg"].astype("float64"),
        }
    )
    out["x"] = out["x_frac"] * config.PITCH_LENGTH
    out["y"] = out["y_frac"] * config.PITCH_WIDTH
    out["is_goal"] = out["result"] == "Goal"
    return out[SCHEMA_COLUMNS].reset_index(drop=True)


def fetch_shots(refresh: bool = False) -> pd.DataFrame:
    """Fetch all configured league/seasons from Understat and write the parquet file."""
    import soccerdata as sd  # lazy: keeps normalise/load importable without the client

    frames = []
    for league in config.LEAGUES:
        for season in config.SEASONS:
            reader = sd.Understat(leagues=league, seasons=season, no_cache=refresh)
            frames.append(normalise_shots(reader.read_shot_events().reset_index()))
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("shot_id")
        .reset_index(drop=True)
    )
    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PATH, index=False)
    return df


def load_shots() -> pd.DataFrame:
    if not config.PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_PATH} not found. Run: uv run python -m foot_eda.data"
        )
    return pd.read_parquet(config.PROCESSED_PATH)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Understat shots to parquet")
    p.add_argument("--refresh", action="store_true", help="bypass soccerdata cache")
    df = fetch_shots(refresh=p.parse_args().refresh)
    print(f"wrote {len(df)} shots to {config.PROCESSED_PATH}")


if __name__ == "__main__":
    main()
