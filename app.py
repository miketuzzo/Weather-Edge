import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt
import philly_edge as pe

st.set_page_config(page_title="Weather Edge", layout="centered")

# Auto-refresh when open (30 minutes)
st_autorefresh(interval=30*60*1000, key="autorefresh_30m")

# -----------------------
# Cities (explicit settlement station definitions)
# station_obs = what we use to fetch observed temps for settlement/perf
# station_label = what we show in the UI as the settlement station name
# -----------------------
CITIES = {
    "Philadelphia": {"series": "KXHIGHPHIL", "station_obs": "KPHL", "station_label": "Philadelphia Intl (KPHL)", "lat": 39.872,  "lon": -75.241},
    "Los Angeles":  {"series": "KXHIGHLAX",  "station_obs": "KLAX", "station_label": "Los Angeles Intl (KLAX)", "lat": 33.9425, "lon": -118.4081},
    "Denver":       {"series": "KXHIGHDEN",  "station_obs": "KDEN", "station_label": "Denver Intl (KDEN)", "lat": 39.8561, "lon": -104.6737},
    "Miami":        {"series": "KXHIGHMIA",  "station_obs": "KMIA", "station_label": "Miami Intl (KMIA)", "lat": 25.7959, "lon": -80.2870},
    "NYC":          {"series": "KXHIGHNY",   "station_obs": "NYC",  "station_label": "Central Park (KNYC / NYC)", "lat": 40.7790, "lon": -73.96925},
    "Chicago":      {"series": "KXHIGHCHI",  "station_obs": "KMDW", "station_label": "Chicago Midway (KMDW)", "lat": 41.7868, "lon": -87.7522},
    "Austin":       {"series": "KXHIGHAUS",  "station_obs": "KAUS", "station_label": "Austin–Bergstrom (KAUS)", "lat": 30.1945, "lon": -97.6699},
}

def apply_city(cfg):
    pe.SERIES_TICKER = cfg["series"]
    pe.STATION = cfg["station_obs"]
    pe.LAT = cfg["lat"]
    pe.LON = cfg["lon"]

def american_odds_from_prob(p: float):
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def fmt_american(o):
    if o is None:
        return ""
    return f"+{o}" if o > 0 else str(o)

def value_color(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return "color: #22c55e;" if v > 0 else "color: #ef4444;"

# -----------------------
# Manual refresh
# -----------------------
st.markdown("## Weather Edge — Multi-City (Daily High)")
st.caption("Leaderboard ranks cities by their best Value% (highest → lowest). Settlement station shown in City view.")

# Update outcomes for past days (historical tracking)
if hasattr(pe, "perf_update_outcomes"):
    pe.perf_update_outcomes()

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(show_spinner=False)
def compute_city_snapshot(city_name: str):
    cfg = CITIES[city_name]
    apply_city(cfg)

    sigma = pe.calibrate_sigma(days_back=14)

    bucket_markets = pe.get_today_bucket_markets()
    if not bucket_markets:
        return None, None, sigma, []

    labels = [bm["label"] for bm in bucket_markets]
    bucket_bounds = [(bm["label"], bm["lo"], bm["hi"]) for bm in bucket_markets]
    probs = pe.model_probs_for_buckets(bucket_bounds, sigma)

    rows = []
    for bm in bucket_markets:
        label = bm["label"]
        m = bm["market"]

        p_model = float(probs.get(label, 0.0))
        yes_ask = pe.yes_ask_prob(m)
        vol = m.get("volume") or m.get("trade_volume") or m.get("volume_24h")

        value = None if yes_ask is None else (p_model - yes_ask)
        odds = american_odds_from_prob(yes_ask) if yes_ask is not None else None

        rows.append({
            "Contract": label,
            "YES ask %": None if yes_ask is None else yes_ask * 100.0,
            "Odds": fmt_american(odds),
            "Volume": vol,
            "Value %": None if value is None else value * 100.0,
            "Model %": p_model * 100.0,
        })

    df = pd.DataFrame(rows)

    best = None
    cand = df.dropna(subset=["Value %"]).copy()
    if len(cand):
        best = cand.sort_values("Value %", ascending=False).iloc[0].to_dict()

    return df, best, sigma, labels

# -----------------------
# Build leaderboard
# -----------------------
leader_rows = []
snapshots = {}

for city_name in CITIES.keys():
    df, best, sigma, labels = compute_city_snapshot(city_name)
    snapshots[city_name] = (df, sigma, labels)

    # Log today's snapshot (deduped per city/day) if tracking is present
    if hasattr(pe, "perf_log_snapshot") and best is not None:
        pe.perf_log_snapshot(
            date_s=pe._today_local_date_str(),
            city=city_name,
            station=CITIES[city_name]["station_obs"],
            sigma_f=sigma,
            labels=labels,
            best_contract=best.get("Contract"),
            yes_ask_prob=(best.get("YES ask %")/100.0 if best.get("YES ask %") is not None else None),
            model_prob=(best.get("Model %")/100.0 if best.get("Model %") is not None else None),
            value_prob=(best.get("Value %")/100.0 if best.get("Value %") is not None else None),
        )

    if best is None:
        leader_rows.append({
            "City": city_name,
            "Best contract": "(no market data)",
            "Value %": None,
            "YES ask %": None,
            "Model %": None,
            "Odds": "",
            "σ": sigma,
        })
    else:
        leader_rows.append({
            "City": city_name,
            "Best contract": best.get("Contract"),
            "Value %": best.get("Value %"),
            "YES ask %": best.get("YES ask %"),
            "Model %": best.get("Model %"),
            "Odds": best.get("Odds", ""),
            "σ": sigma,
        })

lb = pd.DataFrame(leader_rows)
lb["_sort"] = lb["Value %"].fillna(-1e18)
lb = lb.sort_values("_sort", ascending=False).drop(columns=["_sort"])

styled_lb = (
    lb.style
      .format({"Value %": "{:+.1f}%", "YES ask %": "{:.1f}%", "Model %": "{:.1f}%", "σ": "{:.2f}"})
      .map(value_color, subset=["Value %"])
)

st.subheader("Best bet by city (ranked)")
st.dataframe(styled_lb, width="stretch", hide_index=True)

# -----------------------
# City view + settlement station label + forecast graph
# -----------------------
st.subheader("City view")
default_city = (
    lb.dropna(subset=["Value %"]).iloc[0]["City"]
    if (len(lb.dropna(subset=["Value %"])) > 0)
    else "Philadelphia"
)
city_pick = st.selectbox("Select a city", lb["City"].tolist(), index=list(lb["City"]).index(default_city))

df_city, sigma_city, _labels_city = snapshots[city_pick]
cfg = CITIES[city_pick]
st.caption(f"Settlement station: {cfg['station_label']}")

if df_city is None or df_city.empty:
    st.info("No bucket data returned right now for this city.")
else:
    st.caption(f"{city_pick} — σ(auto): {sigma_city:.2f}°F  |  Price = YES ask  |  Value = Model − Price")

    table = df_city[["Contract", "YES ask %", "Odds", "Volume", "Value %", "Model %"]].copy()
    table["Volume"] = pd.to_numeric(table["Volume"], errors="coerce")

    styled = (
        table.style
          .format({"YES ask %": "{:.1f}%", "Volume": "{:,.0f}", "Value %": "{:+.1f}%", "Model %": "{:.1f}%"})
          .map(value_color, subset=["Value %"])
    )

    st.dataframe(styled, width="stretch", hide_index=True)

    # Observed now + high so far (settlement station)
    try:
        apply_city(cfg)
        obs = pe.obs_latest_and_high_today()
        if obs:
            o1, o2 = st.columns(2)
            o1.metric("Observed temp (latest)", f"{obs['latest_temp_f']:.1f}°F", help=f"Time: {obs['latest_time_local']}")
            o2.metric("Observed HIGH so far", f"{obs['high_so_far_f']:.1f}°F")
    except Exception:
        pass

    # Two charts: past 24h observed + next 24h forecast
try:
    apply_city(cfg)

    # ---- Past 24h observed (NWS station) ----
    past = pe.nws_obs_past_hours_station(24)
    df_p = pd.DataFrame(past)
    if not df_p.empty:
        df_p = df_p.sort_values("time_local").rename(columns={"time_local":"time"})

        ymin = float(df_p["temp_f"].min()) - 2.0
        ymax = float(df_p["temp_f"].max()) + 2.0

        st.subheader("Observed — past 24 hours (NWS station)")
        chart_p = (
            alt.Chart(df_p)
            .mark_line(point=True)
            .encode(
                x=alt.X("time:T", axis=alt.Axis(format="%b %-d %-I %p", tickCount=10, title=None)),
                y=alt.Y("temp_f:Q", scale=alt.Scale(domain=[ymin, ymax]), axis=alt.Axis(title="°F")),
                tooltip=[
                    alt.Tooltip("time:T", title="Time", format="%b %-d %-I:%M %p"),
                    alt.Tooltip("temp_f:Q", title="Temp (°F)", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart_p, use_container_width=True)
    else:
        st.caption("Observed chart: no station observations returned for the past 24 hours.")

    # ---- Next 24h forecast (NWS hourly) ----
    fut = pe.nws_hourly_forecast_next_hours(24)
    df_f = pd.DataFrame(fut)
    if not df_f.empty:
        df_f = df_f.sort_values("time_local").rename(columns={"time_local":"time"})

        # Full-degree change markers (forecast only)
        df_f["deg"] = df_f["temp_f"].round(0).astype(int)
        df_f["deg_prev"] = df_f["deg"].shift(1)
        df_marks = df_f[df_f["deg_prev"].isna() | (df_f["deg"] != df_f["deg_prev"])].copy()

        ymin2 = float(df_f["temp_f"].min()) - 2.0
        ymax2 = float(df_f["temp_f"].max()) + 2.0

        st.subheader("Forecast — next 24 hours (NWS hourly)")
        base = alt.Chart(df_f).encode(
            x=alt.X("time:T", axis=alt.Axis(format="%b %-d %-I %p", tickCount=10, title=None)),
            y=alt.Y("temp_f:Q", scale=alt.Scale(domain=[ymin2, ymax2]), axis=alt.Axis(title="°F")),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %-d %-I:%M %p"),
                alt.Tooltip("temp_f:Q", title="Temp (°F)", format=".1f"),
            ],
        )

        line = base.mark_line(point=True)

        pts = alt.Chart(df_marks).mark_point(filled=True, size=70).encode(
            x="time:T", y="temp_f:Q",
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %-d %-I:%M %p"),
                alt.Tooltip("temp_f:Q", title="Temp (°F)", format=".1f"),
                alt.Tooltip("deg:Q", title="Rounded °F", format="d"),
            ],
        )

        lbl = alt.Chart(df_marks).mark_text(dy=-10).encode(
            x="time:T", y="temp_f:Q",
            text=alt.Text("deg:Q", format="d"),
        )

        chart_f = alt.layer(line, pts, lbl).properties(height=260)
        st.altair_chart(chart_f, use_container_width=True)
    else:
        st.caption("Forecast chart: no hourly forecast returned for the next 24 hours.")

except Exception as e:
    st.warning(f"Charts unavailable: {e}")

# -----------------------
# Historical performance (if available)
# -----------------------
if hasattr(pe, "perf_load_df"):
    st.subheader("Historical performance")
    perf = pe.perf_load_df()
    done = perf.dropna(subset=["observed_high_f", "winning_contract", "profit"]).copy()

    if done.empty:
        st.info("No settled history yet. After a day completes, outcomes will populate here.")
    else:
        done = done.sort_values(["date","city"], ascending=[False, True])
        done["YES ask %"] = (done["yes_ask_prob"] * 100).round(1)
        done["Model %"] = (done["model_prob"] * 100).round(1)
        done["Value %"] = (done["value_prob"] * 100).round(1)
        done["Profit %"] = (done["profit"] * 100).round(1)

        show = done[["date","city","best_contract","winning_contract","observed_high_f","YES ask %","Model %","Value %","won","Profit %"]]
        st.dataframe(show, width="stretch", hide_index=True)

        st.subheader("Performance by city")
        grp = done.groupby("city").agg(
            days=("date","count"),
            win_rate=("won", lambda x: round(x.mean()*100, 1)),
            avg_profit_pct=("profit", lambda x: round(x.mean()*100, 2)),
            total_profit_pct=("profit", lambda x: round(x.sum()*100, 1)),
        ).reset_index().sort_values("total_profit_pct", ascending=False)

        st.dataframe(grp, width="stretch", hide_index=True)
