"""Streamlit dashboard: shot EDA on the local Understat parquet. No network access."""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from foot_eda import analysis, config, plots
from foot_eda.data import load_shots
from foot_eda.features import add_features

FETCH_CMD = "uv run python -m foot_eda.data"
PCT_COLS = ["conversion", "ci_low", "ci_high", "share"]
SITUATION_LABELS = {
    "open_play": "Open play",
    "set_piece": "Set piece",
    "direct_free_kick": "Direct free kick",
    "penalty": "Penalty",
}
HEADER_LABELS = {True: "Header", False: "Non-header"}

st.set_page_config(page_title="Shot EDA", layout="wide")


@st.cache_data
def get_shots() -> pd.DataFrame:
    return add_features(load_shots())


def show(fig, table: pd.DataFrame, name: str) -> None:
    st.pyplot(fig)
    plt.close(fig)
    fmt = {c: "{:.1%}" for c in PCT_COLS if c in table.columns}
    st.dataframe(table.style.format(fmt, na_rep="–"), width="stretch")
    st.download_button(
        "Download CSV", table.to_csv(index=False), f"{name}.csv", "text/csv", key=f"dl_{name}"
    )


try:
    with st.spinner("Loading shots..."):
        shots = get_shots()
except FileNotFoundError:
    st.error(f"Shot data not found. Fetch it first with: `{FETCH_CMD}`")
    st.stop()

st.title("Shot EDA: top 5 European leagues")

with st.sidebar:
    st.header("Filters")
    all_leagues = sorted(shots["league"].unique())
    leagues = st.multiselect("League", all_leagues, default=all_leagues)
    all_seasons = sorted(shots["season"].unique())
    seasons = st.multiselect("Season", all_seasons, default=all_seasons[-1:])
    teams = st.multiselect(
        "Team (empty = all)", sorted(shots.loc[shots["league"].isin(leagues), "team"].unique())
    )
    include_pens = st.toggle("Include penalties in Q4", value=False)
    min_shots = st.slider("Min shots per Q2 grid cell", 5, 100, 20)

df = shots[shots["season"].isin(seasons) & shots["league"].isin(leagues)]
if teams:
    df = df[df["team"].isin(teams)]

if df.empty:
    st.warning("No shots for this filter. Widen the league, season or team selection.")
else:
    goals = int(df["is_goal"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shots", f"{len(df):,}")
    c2.metric("Goals", f"{goals:,}")
    c3.metric("Conversion", f"{goals / len(df):.1%}")
    c4.metric("Date range", f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}")

tabs = st.tabs(["Q1 Locations", "Q2 Zones", "Q3 Distance", "Q4 Goal share", "Q5 Situations"])

with tabs[0]:
    st.caption("Where are shots taken and where are goals scored? (non-penalty)")
    locs = analysis.shot_and_goal_locations(df)
    show(plots.plot_shot_map(locs), locs, "q1_locations")

with tabs[1]:
    st.caption("What is the conversion rate by pitch zone? Grey cells have too few shots.")
    grid = analysis.zone_grid(df, min_shots=min_shots)
    show(plots.plot_zone_conversion(grid), grid, "q2_zones")

with tabs[2]:
    st.caption("How does scoring probability fall off with distance? (non-penalty)")
    curve = analysis.distance_curve(df)
    _, predict = analysis.fit_distance_logit(df)
    show(plots.plot_distance_curve(curve, predict), curve, "q3_distance")

with tabs[3]:
    st.caption(
        "What share of goals come from the six-yard box, penalty area and outside the box? "
        "Pitch is a nominal 105 × 68 m, so zone edges are approximate to about 1 m."
    )
    share = analysis.goal_share_by_box_zone(df, include_pens)
    show(plots.plot_goal_share(share), share, "q4_goal_share")

with tabs[4]:
    st.caption("How do headers, open play and set pieces differ?")
    st.info("Counter-attacks are not covered: Understat has no counter-attack tag.")
    sit = analysis.situation_table(df)
    show(plots.plot_rate_comparison(sit, "situation", SITUATION_LABELS), sit, "q5_situation")
    hdr = analysis.header_table(df)
    show(plots.plot_rate_comparison(hdr, "is_header", HEADER_LABELS), hdr, "q5_header")

credit = "Data: [Understat](https://understat.com), for personal, non-commercial use."
if config.PROCESSED_PATH.exists():
    mtime = datetime.fromtimestamp(config.PROCESSED_PATH.stat().st_mtime)
    credit += f" Parquet last updated: {mtime:%Y-%m-%d %H:%M}."
st.caption(credit)
