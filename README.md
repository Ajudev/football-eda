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
```
Writes `data/processed/shots.parquet`. The app only reads this file and never touches the network.

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
