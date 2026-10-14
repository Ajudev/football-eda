"""Streamlit dashboard: shot EDA on the local Understat parquet.

Reads only the local parquet. The sidebar Data section is the one exception: its buttons call
`data.fetch_shots` (network) and `data.remove_seasons` (local), guarded by a process-wide lock.
"""

import threading
from collections.abc import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from foot_eda import analysis, config, data, plots
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
def get_shots(mtime: float) -> pd.DataFrame:
    """`mtime` only keys the cache so a refreshed parquet is reloaded."""
    return add_features(data.load_shots())


@st.cache_resource
def _fetch_lock() -> threading.Lock:
    """One lock per server process, shared by all sessions."""
    return threading.Lock()


def run_data_action(action: Callable[[], str], label: str) -> None:
    """Run a data call under the process lock. `action` returns the success message.

    An empty message means nothing changed (only remove can do that): show info, no rerun.
    """
    lock = _fetch_lock()
    if not lock.acquire(blocking=False):
        st.warning("A data fetch is already running.")
        return
    error, msg = None, ""
    try:
        with st.status(f"{label} in progress...", expanded=True) as status:
            try:
                msg = action()
            except ValueError as e:
                error = str(e)
            except Exception as e:  # noqa: BLE001 - keep old data, show a short message
                error = f"{label} failed: {type(e).__name__}: {e}"
            status.update(
                label=f"{label} failed" if error else f"{label} done",
                state="error" if error else "complete",
            )
    finally:
        lock.release()
    if error:
        st.error(error)
    elif msg:
        st.session_state["fetch_msg"] = msg
        st.rerun()
    else:
        st.info("No rows matched; nothing removed.")


def show(fig: Figure, table: pd.DataFrame, name: str) -> None:
    st.pyplot(fig)
    plt.close(fig)
    fmt: dict[str, str | Callable[[object], str] | None] = {
        c: "{:.1%}" for c in PCT_COLS if c in table.columns
    }
    # Styler refuses > 262,144 cells; Q1 has no pct columns, so skip styling there.
    styled = table.style.format(fmt, na_rep="–") if fmt else table
    st.dataframe(styled, width="stretch")
    st.download_button(
        "Download CSV", table.to_csv(index=False), f"{name}.csv", "text/csv", key=f"dl_{name}"
    )


st.title("Shot EDA: top 5 European leagues")
if fetch_msg := st.session_state.pop("fetch_msg", None):
    st.success(fetch_msg)

current = data.current_season()
data_box = st.sidebar.expander("Data")
with data_box:
    data_leagues = st.multiselect(
        "Leagues to update", config.LEAGUES, default=config.LEAGUES, key="data_leagues"
    )
    import_options = [s for s in reversed(data.available_seasons()) if s != current]
    import_season = st.selectbox("Past season", import_options, key="import_season")
    if st.button("Import season", key="import_btn"):
        run_data_action(
            lambda: (
                f"Imported {len(data.fetch_shots(data_leagues, [import_season])):,} shots for {import_season}."
            ),
            "Fetch",
        )
    force = st.checkbox("Force full re-download", key="force_refresh")
    if st.button(f"Refresh current season ({current})", key="refresh_btn"):
        run_data_action(
            lambda: (
                f"Refreshed {len(data.fetch_shots(data_leagues, [current], refresh=force)):,} shots for {current}."
            ),
            "Fetch",
        )

# Consume the click flag before the load so a failed load cannot leave it armed for a later run.
remove_armed = st.session_state.pop("remove_armed", False)
parquet_mtime = config.PROCESSED_PATH.stat().st_mtime if config.PROCESSED_PATH.exists() else 0.0
try:
    with st.spinner("Loading shots..."):
        shots = get_shots(parquet_mtime)
except FileNotFoundError:
    st.error(
        "Shot data not found. Import a season from the Data section in the sidebar, "
        f"or run: `{FETCH_CMD}`"
    )
    st.stop()

with data_box:
    st.divider()
    remove_season = st.selectbox(
        "Season to remove", sorted(shots["season"].unique(), reverse=True), key="remove_season"
    )
    # Per-season key: changing the target drops the confirmation. The click callback
    # snapshots the box and clears it, so every attempt (ok or failed) needs a fresh tick.
    confirm_key = f"confirm_remove_{remove_season}"
    confirm = st.checkbox("Yes, permanently remove this season's data", key=confirm_key)

    def _arm() -> None:
        st.session_state["remove_armed"] = st.session_state[confirm_key]
        st.session_state[confirm_key] = False

    st.button("Remove season", key="remove_btn", disabled=not confirm, on_click=_arm)
    if remove_armed:

        def _remove() -> str:
            n = data.remove_seasons(data_leagues, [remove_season])
            return f"Removed {n:,} shots for {remove_season}." if n else ""

        run_data_action(_remove, "Remove")

with st.sidebar:
    st.header("Filters")
    all_leagues = sorted(shots["league"].unique())
    leagues = st.multiselect("League", all_leagues, default=all_leagues, key="f_league")
    all_seasons = sorted(shots["season"].unique())
    seasons = st.multiselect("Season", all_seasons, default=all_seasons[-1:], key="f_season")
    teams = st.multiselect(
        "Team (empty = all)",
        sorted(shots.loc[shots["league"].isin(leagues), "team"].unique()),
        key="f_team",
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
    hdr_plot = plots.plot_rate_comparison(hdr, "is_header", HEADER_LABELS)
    show(hdr_plot, hdr.assign(is_header=hdr["is_header"].map(HEADER_LABELS)), "q5_header")

credit = "Data: [Understat](https://understat.com), for personal, non-commercial use."
if parquet_mtime:
    mtime = datetime.fromtimestamp(parquet_mtime)
    credit += f" Parquet last updated: {mtime:%Y-%m-%d %H:%M}."
st.caption(credit)
