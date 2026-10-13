"""Derived shot features. Pure functions: no I/O, no plotting."""

import numpy as np
import pandas as pd

from foot_eda.config import PITCH_LENGTH, PITCH_WIDTH

GOAL_X = float(PITCH_LENGTH)
GOAL_Y = PITCH_WIDTH / 2

# Standard pitch markings; rounded so inclusive boundary checks are not hit by float noise.
SIX_YARD_X_MIN = round(PITCH_LENGTH - 5.5, 2)  # 99.5
SIX_YARD_Y_MIN = round(GOAL_Y - 9.16, 2)  # 24.84
SIX_YARD_Y_MAX = round(GOAL_Y + 9.16, 2)  # 43.16
PENALTY_X_MIN = round(PITCH_LENGTH - 16.5, 2)  # 88.5
PENALTY_Y_MIN = round(GOAL_Y - 20.16, 2)  # 13.84
PENALTY_Y_MAX = round(GOAL_Y + 20.16, 2)  # 54.16

SITUATION_MAP: dict[str, str] = {
    "Penalty": "penalty",
    "DirectFreekick": "direct_free_kick",
    "FromCorner": "set_piece",
    "SetPiece": "set_piece",
    "OpenPlay": "open_play",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with distance, box_zone, situation and is_header added."""
    unknown = set(df["understat_situation"].unique()) - SITUATION_MAP.keys()
    if unknown:
        raise ValueError(f"Unknown understat_situation values: {sorted(map(str, unknown))}")

    out = df.copy()
    out["distance"] = np.sqrt((GOAL_X - out["x"]) ** 2 + (GOAL_Y - out["y"]) ** 2)

    six_yard = (out["x"] >= SIX_YARD_X_MIN) & out["y"].between(SIX_YARD_Y_MIN, SIX_YARD_Y_MAX)
    penalty_area = (out["x"] >= PENALTY_X_MIN) & out["y"].between(PENALTY_Y_MIN, PENALTY_Y_MAX)
    out["box_zone"] = np.select(
        [six_yard, penalty_area], ["six_yard", "penalty_area"], default="outside"
    )
    out["situation"] = out["understat_situation"].map(SITUATION_MAP)
    out["is_header"] = out["body_part"] == "Head"
    return out
