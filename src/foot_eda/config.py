from pathlib import Path

LEAGUES: list[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASONS: list[str] = ["2526"]

PITCH_LENGTH = 105
PITCH_WIDTH = 68

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "shots.parquet"
