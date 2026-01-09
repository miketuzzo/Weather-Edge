
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt
import philly_edge as pe
from datetime import datetime, timezone
import time
from zoneinfo import ZoneInfo
import os
from typing import Optional

# --- Deploy check (confirms Streamlit redeployed your latest push) ---
ET_TZ = ZoneInfo("America/New_York")

def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def get_git_sha_short() -> str:
    # 1) CI env vars (works on some deploy setups)
    for k in ("GITHUB_SHA", "COMMIT_SHA", "RENDER_GIT_COMMIT", "VERCEL_GIT_COMMIT_SHA"):
        v = os.getenv(k, "").strip()
        if v:
            return v[:7]

    # 2) Try reading .git (works on many Streamlit Cloud deployments)
    head = _read_text(".git/HEAD")
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        sha = _read_text(f".git/{ref}")
        if sha:
            return sha[:7]
    elif len(head) >= 7:
        return head[:7]

    return ""

def _parse_git_log_epoch(line: str) -> Optional[int]:
    """Parse an epoch seconds timestamp from a git log line.
    Typical format ends with: '<epoch> <tz>\t<message>'
    We parse from the right to avoid issues with committer names containing spaces.
    """
    try:
        if not line:
            return None
        left = line.split("\t", 1)[0].strip()
        parts = left.split()
        # last two tokens are <epoch> <tz>
        if len(parts) < 2:
            return None
        epoch_s = int(parts[-2])
        return epoch_s
    except Exception:
        return None

def get_deploy_time_et() -> Optional[datetime]:
    """Best-effort 'last deployed' timestamp.
    On Streamlit Cloud, .git logs often exist and give the commit time.
    Fallback to app.py mtime if needed.
    """
    # 1) Try git logs (preferred)
    for log_path in (
        ".git/logs/HEAD",
        ".git/logs/refs/heads/main",
        ".git/logs/refs/remotes/origin/main",
    ):
        log_txt = _read_text(log_path)
        if not log_txt:
            continue
        last = log_txt.splitlines()[-1].strip()
        epoch = _parse_git_log_epoch(last)
        if epoch:
            return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(ET_TZ)

    # 2) Fallback: file modified time (not perfect, but stable)
    try:
        mtime = os.path.getmtime(__file__)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone(ET_TZ)
    except Exception:
        return None

DEPLOY_SHA = get_git_sha_short() or "unknown"
DEPLOYED_AT_ET = get_deploy_time_et()
APP_LOADED_ET = datetime.now(tz=ET_TZ)

# Make charts crisp on Safari/mobile (avoid blurry canvas scaling)
try:
    alt.renderers.set_embed_options(renderer="svg")
except Exception:
    pass

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
    "NYC":          {"series": "KXHIGHNY",   "station_obs": "KNYC", "station_label": "Central Park (KNYC)", "lat": 40.7790, "lon": -73.96925},
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

# Odds guardrails:
# - Exclude very expensive favorites (<= -300)
# - Warn (but do NOT exclude) on big underdogs (>= +250)
ODDS_EXCLUDE_FAVORITE_AT_OR_BELOW = -300
ODDS_WARN_LONGSHOT_AT_OR_ABOVE = 250

def is_odds_too_expensive(odds_str: Optional[str]) -> bool:
    if not odds_str:
        return False
    try:
        s = str(odds_str).strip()
        # allow '+120' or '-250'
        if s.startswith('+'):
            return False
        if s.startswith('-'):
            val = int(s)
            return val <= ODDS_EXCLUDE_FAVORITE_AT_OR_BELOW
        # if it's somehow numeric without sign
        val = int(s)
        return val <= ODDS_EXCLUDE_FAVORITE_AT_OR_BELOW
    except Exception:
        return False


# Warn on very large underdogs (longshots)
def is_odds_longshot(odds_str: Optional[str]) -> bool:
    """Return True if odds are a very large underdog (>= +250)."""
    if not odds_str:
        return False
    try:
        s = str(odds_str).strip()
        if s.startswith('+'):
            val = int(s[1:])
            return val >= ODDS_WARN_LONGSHOT_AT_OR_ABOVE
        # negative or unsigned numeric are not longshots in this sense
        return False
    except Exception:
        return False

def market_lock_info(df: pd.DataFrame, best_contract=None):
    """
    Detect when the market is essentially 'locked' (one contract ~certain).
    Returns: (status_str, dominant_contract, dominant_yes_ask_pct, is_locked, is_not_viable)
    """
    if df is None or getattr(df, "empty", True) or ("YES ask %" not in df.columns) or ("Contract" not in df.columns):
        return ("", None, None, False, False)

    s = pd.to_numeric(df["YES ask %"], errors="coerce")
    if s.isna().all():
        return ("", None, None, False, False)

    top_idx = int(s.idxmax())
    top_val = float(s.loc[top_idx])
    top_contract = str(df.loc[top_idx, "Contract"])

    s2 = s.drop(index=top_idx)
    second_val = float(s2.max()) if len(s2) else float("nan")

    # "Locked" heuristic: one contract >= 97.5% AND gap to #2 >= 90 points
    is_locked = (top_val >= 97.5) and (pd.isna(second_val) or (top_val - second_val >= 90.0))

    # If locked and our "best" is NOT the dominant contract, it's effectively not viable
    is_not_viable = bool(is_locked and best_contract and (str(best_contract) != top_contract))

    if is_not_viable:
        status = "⛔ Not viable (market locked)"
    elif is_locked:
        status = "🔒 Market locked"
    else:
        status = "Live"

    return (status, top_contract, top_val, is_locked, is_not_viable)

def value_color(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return "color: #22c55e;" if v > 0 else "color: #ef4444;"
# -----------------------
# Lock times (global): 9:30 CST and 12:00 CST
# -----------------------
LOCK_TZ = ZoneInfo("America/Chicago")
LOCK_HOUR = 9
LOCK_MIN = 30
LOCK2_HOUR = 12
LOCK2_MIN = 0

def now_cst():
    return datetime.now(tz=LOCK_TZ)

def lock_date_str_cst():
    return now_cst().strftime("%Y-%m-%d")

def is_after_lock_cst():
    n = now_cst()
    return (n.hour, n.minute) >= (LOCK_HOUR, LOCK_MIN)

def is_after_lock2_cst():
    n = now_cst()
    return (n.hour, n.minute) >= (LOCK2_HOUR, LOCK2_MIN)


def render_overall_best_bet(snapshot_tables: dict):
    """Render a single global best-bet banner scanning ALL buckets across ALL cities.
    Accuracy-first, lightly market-blended; Value% used only for tiebreaks.
    """

    # weights
    W_MODEL = 0.90   # primary: model win-probability
    W_MKT = 0.10     # secondary: market wisdom (YES ask)

    st.markdown(
        """
        <div style="padding:14px 16px;border-radius:14px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">
          <div style="font-size:14px;opacity:0.75;margin-bottom:6px;">Overall best bet (accuracy-first)</div>
        """,
        unsafe_allow_html=True,
    )

    best = None

    for city, df in snapshot_tables.items():
        if df is None or getattr(df, "empty", True):
            continue
        if ("Model %" not in df.columns) or ("YES ask %" not in df.columns):
            continue

        cand = df.dropna(subset=["Model %", "YES ask %"]).copy()
        if cand.empty:
            continue

        # probabilities 0..1
        cand["_model_p"] = pd.to_numeric(cand["Model %"], errors="coerce") / 100.0
        cand["_mkt_p"] = pd.to_numeric(cand["YES ask %"], errors="coerce") / 100.0
        cand = cand.dropna(subset=["_model_p", "_mkt_p"]).copy()
        if cand.empty:
            continue

        # Exclude "not worth betting" heavy favorites (odds <= -300)
        if "Odds" in cand.columns:
            cand = cand[~cand["Odds"].apply(is_odds_too_expensive)].copy()
            if cand.empty:
                continue

        # If market is locked, the dominant contract is effectively decided.
        # Treat it as "no longer a bet" and prefer the next-best non-dominant contract.
        _st, _dom, _dom_yes, _locked, _not_viable = market_lock_info(df, best_contract=None)
        if _locked and _dom is not None:
            cand = cand[cand["Contract"].astype(str) != str(_dom)].copy()
            if cand.empty:
                # Nothing to bet anymore for this city
                continue

        # accuracy-first score (market-blended)
        cand["_acc_score"] = (W_MODEL * cand["_model_p"]) + (W_MKT * cand["_mkt_p"])

        # value tiebreak (0 if missing)
        if "Value %" in cand.columns:
            cand["_value_p"] = pd.to_numeric(cand["Value %"], errors="coerce").fillna(0.0) / 100.0
        else:
            cand["_value_p"] = 0.0

        # top row for this city: best acc_score, then model_p, then value_p
        top_city = cand.sort_values(
            ["_acc_score", "_model_p", "_value_p"],
            ascending=[False, False, False],
        ).iloc[0]

        # choose best across cities with same ordering
        if best is None:
            best = {"city": city, "row": top_city}
        else:
            b = best["row"]
            a = top_city
            if (
                (a["_acc_score"] > b["_acc_score"])
                or (a["_acc_score"] == b["_acc_score"] and a["_model_p"] > b["_model_p"])
                or (
                    a["_acc_score"] == b["_acc_score"]
                    and a["_model_p"] == b["_model_p"]
                    and a["_value_p"] > b["_value_p"]
                )
            ):
                best = {"city": city, "row": top_city}

    if best is None:
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("No market data available yet.")
        return

    city = best["city"]
    row = best["row"]

    contract = str(row.get("Contract", ""))
    yes_ask = row.get("YES ask %")
    model = row.get("Model %")
    odds_str = str(row.get("Odds", "") or "")
    longshot = is_odds_longshot(odds_str)

    try:
        acc_score = float(row.get("_acc_score", 0.0))
    except Exception:
        acc_score = 0.0

    # Value% (display only)
    try:
        val = float(row.get("Value %"))
    except Exception:
        val = float("nan")

    c1, c2, c3 = st.columns([1.1, 2.1, 1.2])
    c1.metric("City", city)
    c2.metric("Contract", contract)

    # Edge badge (green/red, neutral if missing)
    if pd.isna(val):
        edge_txt = "—"
        edge_color = "#9ca3af"
    else:
        edge_txt = f"{val:+.1f}%"
        edge_color = "#22c55e" if val > 0 else "#ef4444"

    yes_txt = "" if pd.isna(yes_ask) else f"{float(yes_ask):.1f}%"
    model_txt = "" if pd.isna(model) else f"{float(model):.1f}%"

    c3.markdown(
        f"""
        <div style="height:100%;display:flex;flex-direction:column;justify-content:center;align-items:flex-end;">
          <div style="font-size:12px;opacity:0.7;">Edge (Value %)</div>
          <div style="font-size:26px;font-weight:700;color:{edge_color};line-height:1;">{edge_txt}</div>
          <div style="font-size:12px;opacity:0.7;margin-top:6px;">YES ask: {yes_txt} · Forecast win: {model_txt} · Final rank: {acc_score*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    msg = f"Top pick: **{city} — {contract}** (Final rank **{acc_score*100:.1f}%**)"
    if odds_str:
        msg += f" · Odds: **{odds_str}**"
    st.success(msg)

    if longshot:
        st.warning(
            f"⚠️ Longshot odds (**{odds_str}**). High payout, lower hit-rate. "
            f"Consider skipping unless you have a strong edge. (Warn threshold: +{ODDS_WARN_LONGSHOT_AT_OR_ABOVE})"
        )

    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=24*60*60)
def get_city_sigma(city_name: str) -> float:
    """Calibrate sigma infrequently so the app doesn't hang on first load."""
    cfg = CITIES[city_name]
    apply_city(cfg)
    try:
        # fewer days = much faster
        return float(pe.calibrate_sigma(days_back=3))
    except Exception:
        return 2.0


# -----------------------
# Manual refresh
# -----------------------
st.markdown("## Weather Edge — Multi-City (Daily High)")
st.caption("Leaderboard ranks cities by their best Value% (highest → lowest). Settlement station shown in City view.")
deployed_txt = (
    DEPLOYED_AT_ET.strftime('%Y-%m-%d %I:%M %p %Z')
    if DEPLOYED_AT_ET is not None
    else "unknown"
)

st.caption(
    f"Deploy check — commit `{DEPLOY_SHA}` · deployed {deployed_txt} · page loaded {APP_LOADED_ET.strftime('%Y-%m-%d %I:%M %p %Z')}"
)

# --- Live-vs-local sanity checks (helps debug "why is live different?") ---
perf_path = os.path.join("data", "performance.csv")
perf_exists = os.path.exists(perf_path)
perf_rows = None
perf_mtime = None
perf_note = ""
try:
    if perf_exists:
        perf_mtime = datetime.fromtimestamp(os.path.getmtime(perf_path), tz=timezone.utc).astimezone(ET_TZ)
        _tmp = pd.read_csv(perf_path)
        perf_rows = int(len(_tmp))
        # quick peek at last recorded date/strategy
        _last_date = _tmp["date"].dropna().astype(str).iloc[-1] if ("date" in _tmp.columns and len(_tmp)) else ""
        _last_strat = _tmp["strategy"].dropna().astype(str).iloc[-1] if ("strategy" in _tmp.columns and len(_tmp)) else ""
        if _last_date or _last_strat:
            perf_note = f" · last: {_last_date} {_last_strat}".strip()
except Exception as _e:
    perf_note = f" · perf read error: {_e}" 

# Show a compact debug line so you can confirm the live server has your same data file
perf_mtime_txt = perf_mtime.strftime('%Y-%m-%d %I:%M %p %Z') if perf_mtime is not None else "—"
perf_rows_txt = str(perf_rows) if perf_rows is not None else ("0" if perf_exists else "missing")
st.caption(f"Data check — performance.csv: {perf_rows_txt} rows · mtime {perf_mtime_txt}{perf_note}")

best_bet_slot = st.container()

load_status = st.empty()
load_status.info("Loading live markets… (first load can take ~10–20s)")


# Update outcomes for past days (historical tracking)
# IMPORTANT: surface any errors so we know why the Historical tab is empty.
@st.cache_data(show_spinner=False, ttl=60*60)
def _update_outcomes_cached() -> str:
    if not hasattr(pe, "perf_update_outcomes"):
        return ""
    try:
        pe.perf_update_outcomes()
        return ""
    except Exception as e:
        return str(e)

_outcome_err = _update_outcomes_cached()
if _outcome_err:
    st.warning(
        "Historical outcome update failed (site will still load).\n"
        "This usually means the station/NOAA fetch failed or rate-limited.\n\n"
        f"Details: {_outcome_err}"
    )

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(show_spinner=False, ttl=120)
def compute_city_snapshot(city_name: str, fast: bool = False):
    """
    Returns:
      df (DataFrame): per-bucket table (may be empty)
      best (dict|None): best row by accuracy-first score (may be None)
      sigma (float): calibrated sigma (best-effort)
      labels (list): bucket labels (may be empty)
      err (str): non-fatal error message for UI ("" if OK)
    """
    cfg = CITIES[city_name]
    apply_city(cfg)
    sigma = 2.0 if fast else get_city_sigma(city_name)

    # Fetch markets (can fail if env vars missing / API issues)
    try:
        bucket_markets = pe.get_today_bucket_markets()
    except Exception as e:
        empty = pd.DataFrame(columns=["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"])
        return empty, None, sigma, [], str(e)

    if not bucket_markets:
        empty = pd.DataFrame(columns=["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"])
        return empty, None, sigma, [], "No market data"

    labels = [bm["label"] for bm in bucket_markets]
    bucket_bounds = [(bm["label"], bm["lo"], bm["hi"]) for bm in bucket_markets]

    # Model probabilities (best effort but should usually work)
    try:
        probs = pe.model_probs_for_buckets(bucket_bounds, sigma)
    except Exception as e:
        empty = pd.DataFrame(columns=["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"])
        return empty, None, sigma, labels, str(e)

    rows = []
    for bm in bucket_markets:
        label = bm["label"]
        m = bm["market"]

        p_model = float(probs.get(label, 0.0))

        try:
            yes_ask = pe.yes_ask_prob(m)  # 0..1 or None
        except Exception:
            yes_ask = None

        vol = m.get("volume") or m.get("trade_volume") or m.get("volume_24h")
        value = None if yes_ask is None else (p_model - yes_ask)
        odds = american_odds_from_prob(yes_ask) if yes_ask is not None else None

        rows.append({
            "Contract": label,
            "YES ask %": None if yes_ask is None else yes_ask * 100.0,
            "Odds": fmt_american(odds),
            "Volume": vol,
            "Value %": None if value is None else value * 100.0,
            "Forecast win %": p_model * 100.0,
        })

    df = pd.DataFrame(rows)

    # If the market is effectively locked (one contract ~certain), don't call a different contract the "best bet".
    status_str, dom_contract, dom_yes, is_locked, is_not_viable = market_lock_info(df, best_contract=None)

    # Pick logic
    # If the market is locked (one contract ~certain), the dominant contract is effectively decided.
    # Treat it as "not a bet" and choose the next-best contract that is NOT the dominant one.
    best = None
    locked_dom = str(dom_contract) if (is_locked and dom_contract is not None) else None

    # 2) Otherwise: accuracy-first, lightly market-blended.
    #    Score = 0.90*Model + 0.10*Market(YES ask). Value% only breaks ties.
    if best is None:
        cand = df.dropna(subset=["Forecast win %", "YES ask %"]).copy()
        # Exclude "not worth betting" heavy favorites (odds <= -300)
        if "Odds" in cand.columns:
            cand = cand[~cand["Odds"].apply(is_odds_too_expensive)].copy()
        if len(cand):
            cand["_model_p"] = pd.to_numeric(cand["Forecast win %"], errors="coerce") / 100.0
            cand["_mkt_p"] = pd.to_numeric(cand["YES ask %"], errors="coerce") / 100.0
            cand["_value_p"] = pd.to_numeric(cand.get("Value %", 0.0), errors="coerce").fillna(0.0) / 100.0
            cand = cand.dropna(subset=["_model_p", "_mkt_p"]).copy()
            # If market is locked, skip the dominant (already-decided) contract
            if locked_dom is not None:
                cand = cand[cand["Contract"].astype(str) != locked_dom].copy()
            if cand.empty:
                best = None
            if len(cand):
                cand["_acc"] = 0.90 * cand["_model_p"] + 0.10 * cand["_mkt_p"]
                top = cand.sort_values(["_acc", "_model_p", "_value_p"], ascending=[False, False, False]).iloc[0]
                best = top.to_dict()
                try:
                    best["Acc score %"] = float(top["_acc"]) * 100.0
                except Exception:
                    best["Acc score %"] = None

    # 3) Final fallback (should be rare): use Value%.
    if best is None:
        cand = df.dropna(subset=["Value %"]).copy()
        # Exclude "not worth betting" heavy favorites (odds <= -300)
        if "Odds" in cand.columns:
            cand = cand[~cand["Odds"].apply(is_odds_too_expensive)].copy()
        # If market is locked, skip the dominant (already-decided) contract
        if locked_dom is not None:
            cand = cand[cand["Contract"].astype(str) != locked_dom].copy()
        if len(cand):
            best = cand.sort_values("Value %", ascending=False).iloc[0].to_dict()
            best["Acc score %"] = None

    return df, best, sigma, labels, ""

# -----------------------
# Build leaderboard + logging
# -----------------------
leader_rows = []
snapshots = {}

# Prepare lock directory and lock file paths (two lock times)
lock_dir = os.path.join("data", "locks")
os.makedirs(lock_dir, exist_ok=True)

lock_file_0930 = os.path.join(lock_dir, f"LOCKED_0930_{lock_date_str_cst()}_CST.txt")
lock_file_1200 = os.path.join(lock_dir, f"LOCKED_1200_{lock_date_str_cst()}_CST.txt")

locked_0930 = os.path.isfile(lock_file_0930)
locked_1200 = os.path.isfile(lock_file_1200)

after_0930 = is_after_lock_cst()
after_1200 = is_after_lock2_cst()

DO_LOCK_0930 = False
DO_LOCK_1200 = False

if after_0930 and not locked_0930:
    try:
        with open(lock_file_0930, "w") as f:
            f.write(f"Locked picks for {lock_date_str_cst()} 09:30 CST\n")
        DO_LOCK_0930 = True
    except Exception:
        DO_LOCK_0930 = False

if after_1200 and not locked_1200:
    try:
        with open(lock_file_1200, "w") as f:
            f.write(f"Locked picks for {lock_date_str_cst()} 12:00 CST\n")
        DO_LOCK_1200 = True
    except Exception:
        DO_LOCK_1200 = False

for city_name in CITIES.keys():
    df, best, sigma, labels, err = compute_city_snapshot(city_name, fast=True)
    snapshots[city_name] = (df, sigma, labels, err)

    # Append snapshot row for each city if possible
    if hasattr(pe, "snap_append_row") and df is not None and not df.empty:
        cand = df.dropna(subset=["Value %"]).copy()
        if len(cand):
            top_city = cand.sort_values("Value %", ascending=False).iloc[0]
            ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            ts_local = now_cst().strftime("%Y-%m-%d %H:%M:%S")
            date_local = lock_date_str_cst()
            station = CITIES[city_name]["station_obs"]

            yes_ask_prob = (float(top_city["YES ask %"]) / 100.0) if pd.notna(top_city["YES ask %"]) else None
            model_prob = (float(top_city["Model %"]) / 100.0) if pd.notna(top_city["Model %"]) else None
            value_prob = (float(top_city["Value %"]) / 100.0) if pd.notna(top_city["Value %"]) else None
            volume = top_city.get("Volume", None)

            try:
                pe.snap_append_row(
                    ts_utc=ts_utc,
                    ts_local=ts_local,
                    date_local=date_local,
                    city=city_name,
                    station=station,
                    sigma_f=float(sigma),
                    contract=str(top_city["Contract"]),
                    yes_ask_prob=yes_ask_prob,
                    model_prob=model_prob,
                    value_prob=value_prob,
                    volume=volume,
                )
            except Exception:
                pass

    # Log official (graded) picks at lock times once per day (9:30 CST and 12:00 CST)
    if DO_LOCK_0930 and hasattr(pe, "perf_log_snapshot") and best is not None:
        try:
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
                strategy="lock_0930",
            )
        except Exception:
            pass

    if DO_LOCK_1200 and hasattr(pe, "perf_log_snapshot") and best is not None:
        try:
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
                strategy="lock_1200",
            )
        except Exception:
            pass

    if best is None:
        leader_rows.append({
            "City": city_name,
            "Best contract": "(no market data)",
            "Acc score %": None,
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
            "Acc score %": best.get("Acc score %"),
            "Value %": best.get("Value %"),
            "YES ask %": best.get("YES ask %"),
            "Model %": best.get("Model %"),
            "Odds": best.get("Odds", ""),
            "σ": sigma,
        })

load_status.empty()

lb = pd.DataFrame(leader_rows)

# Rename leaderboard columns for clarity
lb = lb.rename(columns={"Acc score %": "Final rank %", "Model %": "Forecast win %"})

# Ensure numeric columns are real numbers (None -> NaN) so Styler formatters
# don't crash with "unsupported format string passed to NoneType".
for _col in ["Final rank %", "Value %", "YES ask %", "Forecast win %", "σ"]:
    if _col in lb.columns:
        lb[_col] = pd.to_numeric(lb[_col], errors="coerce")

# Rank cities by accuracy-first score (fallback to Value% if missing)
if "Final rank %" in lb.columns:
    lb["_sort"] = lb["Final rank %"].fillna(-1e18)
else:
    lb["_sort"] = lb["Value %"].fillna(-1e18)
lb = lb.sort_values("_sort", ascending=False).drop(columns=["_sort"])

# Live market status column (locked / not viable)
def _status_for_city(city: str, best_contract: Optional[str]) -> str:
    df0 = snapshots.get(city, (None, None, None, ""))[0]
    status_str, dom_contract, dom_yes, is_locked, is_not_viable = market_lock_info(df0, best_contract=best_contract)
    if is_not_viable and dom_contract is not None and dom_yes is not None:
        return f"⛔ Locked to {dom_contract} ({float(dom_yes):.1f}%)"
    if is_locked and dom_contract is not None and dom_yes is not None:
        return f"🔒 Locked to {dom_contract} ({float(dom_yes):.1f}%)"
    return status_str or "Live"

lb["Status"] = [
    _status_for_city(row["City"], row.get("Best contract"))
    for _, row in lb.iterrows()
]

# Nice column order for the leaderboard
_cols = [
    "City",
    "Status",
    "Best contract",
    "Final rank %",
    "Forecast win %",
    "YES ask %",
    "Value %",
    "Odds",
    "σ",
]
lb = lb[[c for c in _cols if c in lb.columns]]

snapshot_tables = {city: snapshots[city][0] for city in snapshots}
with best_bet_slot:
    render_overall_best_bet(snapshot_tables)

# Show any non-fatal data errors so the page doesn't look "blank" when an API call fails
errs = {c: snapshots[c][3] for c in snapshots if len(snapshots[c]) > 3 and snapshots[c][3]}
if errs:
    st.warning(
        "Some live data calls failed (the app will still load):\n"
        + "\n".join([f"- {c}: {m}" for c, m in errs.items()])
    )

styled_lb = (
    lb.style
      .format(
          {"Final rank %": "{:.1f}%", "Value %": "{:+.1f}%", "YES ask %": "{:.1f}%", "Forecast win %": "{:.1f}%", "σ": "{:.2f}"},
          na_rep="—",
      )
      .map(value_color, subset=["Value %"])
)

st.caption(
    "Legend: Forecast win% = model-only probability. Final rank% = 90% model + 10% market (YES ask). "
    "Value% = (Forecast − Price)."
)
st.subheader("Best bet by city (ranked)")
st.caption(f"Odds guardrails: exclude favorites <= {ODDS_EXCLUDE_FAVORITE_AT_OR_BELOW} · warn longshots >= +{ODDS_WARN_LONGSHOT_AT_OR_ABOVE}")
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

df_city, best_city, sigma_city, _labels_city, err_city = compute_city_snapshot(city_pick, fast=False)
cfg = CITIES[city_pick]
st.caption(f"Settlement station: {cfg['station_label']}")

if err_city:
    st.warning(f"{city_pick} live data error: {err_city}")

if df_city is None or df_city.empty:
    st.info("No bucket data returned right now for this city.")
else:
    st.caption(f"{city_pick} — σ(auto): {sigma_city:.2f}°F  |  Price = YES ask  |  Value = Forecast − Price")

    table = df_city[["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"]].copy()
    table["Volume"] = pd.to_numeric(table["Volume"], errors="coerce")
    table["⚠️"] = table["Odds"].apply(lambda s: "⚠️" if is_odds_longshot(s) else "")

    styled = (
        table.style
          .format({"YES ask %": "{:.1f}%", "Volume": "{:,.0f}", "Value %": "{:+.1f}%", "Forecast win %": "{:.1f}%"})
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

# Two charts: past 12h observed + next 12h forecast
try:
    apply_city(cfg)

    # ---- Past 12h observed (NWS station) ----
    past = pe.nws_obs_past_hours_station(12)
    df_p = pd.DataFrame(past)
    if not df_p.empty:
        df_p = df_p.sort_values("time_local").rename(columns={"time_local":"time"})

        # Downsample: keep only points when the observed temp changes (full-degree), plus the first point
        df_p["deg"] = df_p["temp_f"].round(0).astype(int)
        df_p["deg_prev"] = df_p["deg"].shift(1)
        df_p = df_p[df_p["deg_prev"].isna() | (df_p["deg"] != df_p["deg_prev"])].copy()

        ymin = float(df_p["temp_f"].min()) - 2.0
        ymax = float(df_p["temp_f"].max()) + 2.0

        st.subheader("Observed — past 12 hours (NWS station)")
        chart_p = (
            alt.Chart(df_p)
            .mark_line(point=True)
            .encode(
                x=alt.X("time:T", axis=alt.Axis(format="%b %-d %-I %p", tickCount=6, title=None)),
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
        st.caption("Observed chart: no station observations returned for the past 12 hours.")

    # ---- Next 12h forecast (NWS hourly) ----
    fut = pe.nws_hourly_forecast_next_hours(12)
    df_f = pd.DataFrame(fut)
    if not df_f.empty:
        df_f = df_f.sort_values("time_local").rename(columns={"time_local":"time"})

        # Full-degree change markers (forecast only)
        df_f["deg"] = df_f["temp_f"].round(0).astype(int)
        df_f["deg_prev"] = df_f["deg"].shift(1)
        df_marks = df_f[df_f["deg_prev"].isna() | (df_f["deg"] != df_f["deg_prev"])].copy()

        ymin2 = float(df_f["temp_f"].min()) - 2.0
        ymax2 = float(df_f["temp_f"].max()) + 2.0

        st.subheader("Forecast — next 12 hours (NWS hourly)")
        base = alt.Chart(df_f).encode(
            x=alt.X("time:T", axis=alt.Axis(format="%b %-d %-I %p", tickCount=6, title=None)),
            y=alt.Y("temp_f:Q", scale=alt.Scale(domain=[ymin2, ymax2]), axis=alt.Axis(title="°F")),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %-d %-I:%M %p"),
                alt.Tooltip("temp_f:Q", title="Temp (°F)", format=".1f"),
            ],
        )

        line = base.mark_line(point=True)

        pts = alt.Chart(df_marks).mark_point(filled=True, size=70).encode(
            x=alt.X("time:T", axis=alt.Axis(format="%b %-d %-I %p", tickCount=6, title=None)), y="temp_f:Q",
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %-d %-I:%M %p"),
                alt.Tooltip("temp_f:Q", title="Temp (°F)", format=".1f"),
                alt.Tooltip("deg:Q", title="Rounded °F", format="d"),
            ],
        )

        lbl = alt.Chart(df_marks).mark_text(dy=-10).encode(
            x=alt.X("time:T", axis=alt.Axis(format="%b %-d %-I %p", tickCount=6, title=None)), y="temp_f:Q",
            text=alt.Text("deg:Q", format="d"),
        )

        chart_f = alt.layer(line, pts, lbl).properties(height=260)
        st.altair_chart(chart_f, use_container_width=True)
    else:
        st.caption("Forecast chart: no hourly forecast returned for the next 12 hours.")

except Exception as e:
    st.warning(f"Charts unavailable: {e}")

# -----------------------
# Historical performance (if available)
# -----------------------
if hasattr(pe, "perf_load_df"):
    st.subheader("Historical performance")
    if not os.path.exists(perf_path):
        st.info(
            "No performance.csv found on this server, so there’s no history to show. "
            "Your local tracker writes to data/performance.csv on your machine; the live site won’t see that file unless you "
            "persist it (e.g., commit it, upload it to storage, or run tracking on the server)."
        )
    try:
        perf = pe.perf_load_df()
    except Exception as e:
        st.error(f"Failed to load performance history: {e}")
        perf = pd.DataFrame()
    done = perf.dropna(subset=["observed_high_f", "profit", "won"]).copy()

    if done.empty:
        st.info(
            "No settled history to display yet.\n\n"
            "If you *expect* settled rows (e.g., yesterday finished), the most common causes are:\n"
            "• outcomes couldn't be fetched from the weather source for that station/date\n"
            "• the app container restarted and hasn't re-run outcome updates yet\n\n"
            "Tip: refresh once, and check for a warning above about outcome-update failure."
        )
    else:
        # Normalize types
        done["profit"] = pd.to_numeric(done["profit"], errors="coerce")
        done["won"] = pd.to_numeric(done["won"], errors="coerce")
        done = done.dropna(subset=["profit", "won"]).copy()

        # Ensure strategy exists
        if "strategy" not in done.columns:
            done["strategy"] = "lock_0930"
        done["strategy"] = done["strategy"].fillna("lock_0930")

        # Keep only the lock strategies we care about
        keep_strats = ["lock_0930", "lock_1200"]
        done = done[done["strategy"].isin(keep_strats)].copy()

        # Limit history for speed (last N dates with settled outcomes)
        MAX_HISTORY_DAYS = 30
        try:
            _recent_dates = sorted(done["date"].unique(), reverse=True)[:MAX_HISTORY_DAYS]
            done = done[done["date"].isin(_recent_dates)].copy()
        except Exception:
            pass

        st.markdown("### Daily summary by lock time (W–L only)")

        daily = (
            done.groupby(["date", "strategy"], as_index=False)
                .agg(
                    bets=("won", "count"),
                    wins=("won", "sum"),
                )
        )

        def _pivot_cell(df_in: pd.DataFrame, strat: str, col: str):
            return df_in[df_in["strategy"] == strat].set_index("date")[col]

        dates = pd.Index(sorted(daily["date"].unique(), reverse=True), name="date")
        out = pd.DataFrame({"date": dates})

        for strat, label in [("lock_0930", "09:30 CST"), ("lock_1200", "12:00 CST")]:
            bets_s = _pivot_cell(daily, strat, "bets").reindex(dates)
            wins_s = _pivot_cell(daily, strat, "wins").reindex(dates)

            out[f"{label} W-L"] = [
                "—" if pd.isna(b) else f"{int(w)}/{int(b)}"
                for w, b in zip(wins_s, bets_s)
            ]

        wl_cols = [c for c in out.columns if c.endswith("W-L")]

        def _bg_wl(s: str):
            if not isinstance(s, str) or s == "—":
                return ""
            try:
                w, b = s.split("/")
                w = int(w); b = int(b)
                if b == 0:
                    return ""
                if w == b:
                    return "background-color: rgba(34,197,94,0.18);"
                if w == 0:
                    return "background-color: rgba(239,68,68,0.18);"
                return "background-color: rgba(250,204,21,0.14);"  # yellow-ish for mixed
            except Exception:
                return ""

        styled_out = out.style.applymap(_bg_wl, subset=wl_cols)
        st.dataframe(styled_out, width="stretch", hide_index=True)

        # ------------------------------------------------------------------
        # Combined: Performance by city + inline settled rows
        # ------------------------------------------------------------------
        st.subheader("Performance by city")
        st.caption("Click a city to see its settled rows. Win% = % of locked picks that matched the winning contract.")

        # Build a compact city summary table (Overall / by lock)
        tabs = st.tabs(["Overall", "09:30 CST", "12:00 CST"])

        def _city_summary(df_in: pd.DataFrame) -> pd.DataFrame:
            g = (
                df_in.groupby("city", as_index=False)
                    .agg(
                        bets=("won", "count"),
                        wins=("won", "sum"),
                        win_rate=("won", "mean"),
                    )
            )
            g["Win%"] = (pd.to_numeric(g["win_rate"], errors="coerce") * 100.0).round(1)
            g = g.drop(columns=["win_rate"], errors="ignore")
            g = g[["city", "bets", "wins", "Win%"]]
            g = g.sort_values(["Win%", "wins"], ascending=[False, False])
            g = g.rename(columns={"city": "City", "bets": "Bets", "wins": "Wins"})
            return g

        def _style_city(df_in: pd.DataFrame):
            def _bg_win(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "background-color: rgba(34,197,94,0.18);" if v >= 50.0 else "background-color: rgba(239,68,68,0.18);"

            return (
                df_in.style
                    .format({"Win%": "{:.1f}%"}, na_rep="—")
                    .applymap(_bg_win, subset=["Win%"])
            )

        def _rows_for_city(df_in: pd.DataFrame, city_name: str) -> pd.DataFrame:
            d = df_in[df_in["city"] == city_name].copy()
            if d.empty:
                return d
            d = d.sort_values(["date", "strategy"], ascending=[False, True])

            # friendly percent columns
            if "yes_ask_prob" in d.columns:
                d["YES ask %"] = (pd.to_numeric(d["yes_ask_prob"], errors="coerce") * 100).round(1)
            if "model_prob" in d.columns:
                d["Forecast win %"] = (pd.to_numeric(d["model_prob"], errors="coerce") * 100).round(1)
            if "value_prob" in d.columns:
                d["Value %"] = (pd.to_numeric(d["value_prob"], errors="coerce") * 100).round(1)
            if "profit" in d.columns:
                d["Profit %"] = (pd.to_numeric(d["profit"], errors="coerce") * 100).round(2)

            cols = [
                c for c in [
                    "date", "strategy", "best_contract", "winning_contract",
                    "observed_high_f", "YES ask %", "Forecast win %", "Value %", "won", "Profit %"
                ]
                if c in d.columns
            ]
            return d[cols]

        def _render_city_panel(df_in: pd.DataFrame, label: str):
            if df_in.empty:
                st.info("No settled rows for this view yet.")
                return

            summ = _city_summary(df_in)
            st.dataframe(_style_city(summ), width="stretch", hide_index=True)

            # Pick a city and show its settled rows directly below
            cities = summ["City"].tolist()
            default_city = cities[0] if cities else None
            city_pick2 = st.selectbox(
                f"Show settled rows for a city ({label})",
                options=cities,
                index=0 if default_city else None,
                key=f"hist_city_pick_{label}",
            )

            if not city_pick2:
                return

            rows = _rows_for_city(df_in, city_pick2)
            if rows.empty:
                st.caption("No settled rows for this city yet in this view.")
                return

            # Add quick W/L indicators per row
            if "won" in rows.columns:
                def _bg_won(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return ""
                    return "background-color: rgba(34,197,94,0.18);" if float(v) >= 1.0 else "background-color: rgba(239,68,68,0.18);"

                st.dataframe(rows.style.applymap(_bg_won, subset=["won"]), width="stretch", hide_index=True)
            else:
                st.dataframe(rows, width="stretch", hide_index=True)

        with tabs[0]:
            _render_city_panel(done, "overall")

        with tabs[1]:
            _render_city_panel(done[done["strategy"] == "lock_0930"].copy(), "0930")

        with tabs[2]:
            _render_city_panel(done[done["strategy"] == "lock_1200"].copy(), "1200")