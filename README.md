# foot-eda

Streamlit dashboard exploring shot data for the top five European leagues (2025/26 by default).
Tabs: shot/goal locations, conversion by pitch zone, distance curve, goal share by box zone,
and headers vs open play vs set pieces. Counter-attacks are not covered (Understat has no tag).

## Setup
```
uv sync
```

## Fetch data (once, needs network)
```
uv run python -m foot_eda.data            # add --refresh to bypass the cache
uv run python -m foot_eda.data --season 2425 --league "ENG-Premier League"   # repeatable flags
```
Writes `data/processed/shots.parquet`. Fetches merge into the file: only the fetched
(league, season) pairs are replaced, all other rows are kept. Writes are atomic, and a failed
or empty fetch leaves the file unchanged. The CLI prints the fetched count and the file total.
Seasons run from `1415` to the current season. The dashboard's sidebar Data section can import,
refresh and remove seasons (remove deletes rows from the parquet only; the file is deleted if no
rows remain). Otherwise the app only reads this file.

## Run
```
uv run streamlit run app.py
```

## Tests
```
uv run pytest -q
```

## Data source and terms
Shot data comes from [Understat](https://understat.com) via `soccerdata`. It is for personal,
non-commercial use. Check Understat's terms before any public deployment.

Understat coordinates are 0-1 fractions, scaled to a nominal 105 x 68 m pitch. Real pitch sizes
vary, so box-zone boundaries are approximate to about 1 m.
