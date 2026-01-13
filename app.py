
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import altair as alt
import philly_edge as pe

# NOTE: Picks are now computed via pe.compute_pick_for_today()
# This includes strategy-aware sigma, EWMA bias correction,
# accuracy-gated + EV-aware selection, and no-bet handling.

from datetime import datetime, timezone
import time
from zoneinfo import ZoneInfo
import os
import json
import re
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

# Set Streamlit page config
st.set_page_config(page_title="Weather Edge", layout="wide")

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
# UI: Dashboard styling + helpers (UI-only)
# -----------------------
st.markdown(
    """
    <style>
      .we-wrap {max-width: 1220px; margin: 0 auto;}
      .we-hero {padding: 14px 16px; border-radius: 18px; border:1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.03);}
      .we-title {font-size: 38px; font-weight: 900; letter-spacing: -0.02em; line-height: 1.05; margin: 0;}
      .we-sub {opacity: 0.78; margin: 8px 0 0 0; font-size: 13px;}
      .we-row {display:flex; gap:12px; flex-wrap:wrap; margin-top: 12px;}
      .we-card {border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.03); border-radius: 16px; padding: 12px 14px;}
      .we-card h4 {margin:0 0 6px 0; font-size: 12px; opacity:0.78; font-weight:800; text-transform: uppercase; letter-spacing: 0.06em;}
      .we-metric {font-size: 24px; font-weight: 900; margin: 0;}
      .we-muted {opacity: 0.72; font-size: 12px; margin-top: 4px;}
      .we-pill {display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; border:1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.05);} 
      .we-pill.good {background: rgba(34,197,94,0.18);} 
      .we-pill.bad {background: rgba(239,68,68,0.18);} 
      .we-pill.warn {background: rgba(250,204,21,0.18);} 
      .we-pill.neutral {background: rgba(148,163,184,0.18);} 
      .we-grid {display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;}
      @media (max-width: 1050px) { .we-grid {grid-template-columns: repeat(2, minmax(0, 1fr));} }
      @media (max-width: 650px) { .we-grid {grid-template-columns: repeat(1, minmax(0, 1fr));} }
      .we-city {font-size: 18px; font-weight: 900; margin: 0;}
      .we-city-top {display:flex; justify-content:space-between; align-items:flex-start; gap:10px;}
      .we-kv {display:flex; gap:10px; flex-wrap:wrap; margin-top: 8px;}
      .we-kv div {font-size: 12px; opacity: 0.88;}
      .we-bar {height: 8px; width: 100%; border-radius: 999px; background: rgba(148,163,184,0.22); overflow:hidden; margin-top:10px;}
      .we-bar > span {display:block; height:100%; background: rgba(59,130,246,0.75); width: 0%;}
    </style>
    """,
    unsafe_allow_html=True,
)

def _fmt_pct(x, digits=1, signed=False):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        v = float(x)
        if signed:
            return f"{v:+.{digits}f}%"
        return f"{v:.{digits}f}%"
    except Exception:
        return "—"

def _pill_class(status: str) -> str:
    s = (status or "").lower()
    if "not viable" in s or "⛔" in s:
        return "bad"
    if "locked" in s or "🔒" in s:
        return "warn"
    if "live" in s:
        return "good"
    return "neutral"

def _conf_width(conf_pct) -> int:
    try:
        if conf_pct is None or (isinstance(conf_pct, float) and pd.isna(conf_pct)):
            return 0
        v = float(conf_pct)
        v = max(0.0, min(100.0, v))
        return int(round(v))
    except Exception:
        return 0

def render_hero(deployed_txt: str):
    now_txt = now_cst().strftime("%a %b %-d, %Y · %-I:%M %p CST")
    strat_txt = "12:00 strategy" if is_after_lock2_cst() else "09:30 strategy"
    st.markdown(
        f"""
        <div class="we-wrap">
          <div class="we-hero">
            <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;">
              <div>
                <div class="we-title">Weather Edge</div>
                <div class="we-sub">Daily high-temp contract picks for 7 cities · Accuracy-first ranking (80% model / 20% market) · {now_txt}</div>
              </div>
              <div style="text-align:right;">
                <div class="we-pill neutral">{strat_txt}</div>
                <div class="we-sub" style="margin-top:6px;">Deploy: <code>{DEPLOY_SHA}</code> · {deployed_txt}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_kpis(market_ok: int, market_total: int):
    lock1 = "✅ after lock" if is_after_lock_cst() else "⏳ before lock"
    lock2 = "✅ after lock" if is_after_lock2_cst() else "⏳ before lock"
    st.markdown(
        f"""
        <div class="we-wrap">
          <div class="we-row">
            <div class="we-card" style="flex:1; min-width:220px;">
              <h4>Market health</h4>
              <div class="we-metric">{market_ok}/{market_total}</div>
              <div class="we-muted">Cities with live bucket data</div>
            </div>
            <div class="we-card" style="flex:1; min-width:220px;">
              <h4>09:30 CST</h4>
              <div class="we-metric">{lock1}</div>
              <div class="we-muted">First daily lock</div>
            </div>
            <div class="we-card" style="flex:1; min-width:220px;">
              <h4>12:00 CST</h4>
              <div class="we-metric">{lock2}</div>
              <div class="we-muted">Second daily lock</div>
            </div>
            <div class="we-card" style="flex:1; min-width:220px;">
              <h4>Rules</h4>
              <div class="we-metric">No fabricated wins</div>
              <div class="we-muted">Pending days show ⏳ until NOAA/NWS highs settle</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_city_cards(lb_df: pd.DataFrame):
    if lb_df is None or lb_df.empty:
        st.info("No leaderboard rows to display.")
        return

    cards_html = []
    for _, r in lb_df.iterrows():
        city = str(r.get("City", ""))
        status = str(r.get("Status", ""))
        best = str(r.get("Best contract", "—"))
        final_rank = r.get("Final rank %")
        conf = r.get("Confidence %")
        value = r.get("Value %")
        mprob = r.get("Forecast win %")
        yask = r.get("YES ask %")
        odds = str(r.get("Odds", "") or "")
        sig = r.get("σ")

        pill = _pill_class(status)
        conf_w = _conf_width(conf)

        edge_cls = "neutral"
        try:
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                edge_cls = "good" if float(value) > 0 else "bad"
        except Exception:
            edge_cls = "neutral"

        cards_html.append(
            f"""
            <div class="we-card">
              <div class="we-city-top">
                <div>
                  <div class="we-city">{city}</div>
                  <div class="we-muted">Best contract: <b>{best}</b></div>
                </div>
                <div><span class="we-pill {pill}">{status}</span></div>
              </div>
              <div class="we-kv">
                <div><b>Final rank</b>: {_fmt_pct(final_rank, 1)}</div>
                <div><b>Confidence</b>: {_fmt_pct(conf, 0)}</div>
                <div><b>Edge</b>: <span class="we-pill {edge_cls}">{_fmt_pct(value, 1, signed=True)}</span></div>
                <div><b>Forecast</b>: {_fmt_pct(mprob, 1)}</div>
                <div><b>YES ask</b>: {_fmt_pct(yask, 1)}</div>
                <div><b>Odds</b>: {odds if odds else "—"}</div>
                <div><b>σ</b>: {"—" if (sig is None or (isinstance(sig, float) and pd.isna(sig))) else f"{float(sig):.2f}"}</div>
              </div>
              <div class="we-bar"><span style="width:{conf_w}%;"></span></div>
              <div class="we-muted">Confidence is informational only · Pending days never graded early</div>
            </div>
            """
        )

    st.markdown('<div class="we-wrap"><div class="we-grid">' + "".join(cards_html) + '</div></div>', unsafe_allow_html=True)
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

    # weights (accuracy boost): lean more on market consensus
    W_MODEL = 0.80   # primary: model win-probability
    W_MKT = 0.20     # secondary: market wisdom (YES ask)

    # Per-city tables currently use "Forecast win %" (older versions used "Model %")
    def _model_col(df_in: pd.DataFrame) -> Optional[str]:
        if df_in is None:
            return None
        if "Model %" in df_in.columns:
            return "Model %"
        if "Forecast win %" in df_in.columns:
            return "Forecast win %"
        return None

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
        mcol = _model_col(df)
        if (mcol is None) or ("YES ask %" not in df.columns):
            continue

        cand = df.dropna(subset=[mcol, "YES ask %"]).copy()
        if cand.empty:
            continue

        # probabilities 0..1
        cand["_model_p"] = pd.to_numeric(cand[mcol], errors="coerce") / 100.0
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
        st.markdown(
            """
            <div style=\"margin-top:8px;opacity:0.85;\">No viable overall bet right now — either market/model data is missing, the best prices are filtered (odds guardrails), or markets are effectively locked.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    city = best["city"]
    row = best["row"]

    contract = str(row.get("Contract", ""))
    yes_ask = row.get("YES ask %")
    # display the model column that exists
    _mcol_best = _model_col(pd.DataFrame([row]))
    model = row.get(_mcol_best) if _mcol_best else row.get("Forecast win %")
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
def compute_city_snapshot(city_name: str, strategy: str = "lock_0930", fast: bool = False):
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
    sigma_base = 2.0 if fast else get_city_sigma(city_name)
    sigma = pe.sigma_for_strategy(strategy, base_sigma_f=sigma_base) if hasattr(pe, "sigma_for_strategy") else sigma_base

    result = None  # engine output (candidates/confidence/market_bad)

    # Fetch markets (can fail if env vars missing / API issues)
    try:
        bucket_markets = pe.get_today_bucket_markets()
    except Exception as e:
        empty = pd.DataFrame(columns=["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"])
        return empty, None, sigma, [], str(e), None

    if not bucket_markets:
        empty = pd.DataFrame(columns=["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"])
        return empty, None, sigma, [], "No market data", None

    labels = [bm["label"] for bm in bucket_markets]
    bucket_bounds = [(bm["label"], bm["lo"], bm["hi"]) for bm in bucket_markets]

    # Model probabilities (best effort but should usually work)
    try:
        result = pe.compute_pick_for_today(strategy=strategy)  # UPDATED
        probs = result['probs']
        # old: pe.model_probs_for_buckets(bucket_bounds, sigma)
    except Exception as e:
        empty = pd.DataFrame(columns=["Contract", "YES ask %", "Odds", "Volume", "Value %", "Forecast win %"])
        return empty, None, sigma, labels, str(e), None

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
                cand["_acc"] = 0.80 * cand["_model_p"] + 0.20 * cand["_mkt_p"]
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

    return df, best, sigma, labels, "", result

# -----------------------
# Build leaderboard + logging
# -----------------------
leader_rows = []
snapshots = {}

# Prepare lock directory and lock file paths (two lock times)
# IMPORTANT: We do NOT create lock files until we successfully log at least one row.
# Otherwise a transient API failure (or best=None) can "burn" the day and prevent later runs (cron/track.py)
# from logging the missing rows.
lock_dir = os.path.join("data", "locks")
os.makedirs(lock_dir, exist_ok=True)

lock_file_0930 = os.path.join(lock_dir, f"LOCKED_0930_{lock_date_str_cst()}_CST.txt")
lock_file_1200 = os.path.join(lock_dir, f"LOCKED_1200_{lock_date_str_cst()}_CST.txt")

locked_0930 = os.path.isfile(lock_file_0930)
locked_1200 = os.path.isfile(lock_file_1200)

after_0930 = is_after_lock_cst()
after_1200 = is_after_lock2_cst()

# We intend to log if we're past the lock time and not already locked.
DO_LOCK_0930 = bool(after_0930 and not locked_0930)
DO_LOCK_1200 = bool(after_1200 and not locked_1200)

# Use noon strategy after 12:00 CST for live tables
DISPLAY_STRATEGY = "lock_1200" if after_1200 else "lock_0930"

# Track whether we actually logged anything; only then do we create the lock file.
logged_any_0930 = False
logged_any_1200 = False

for city_name in CITIES.keys():
    df, best, sigma, labels, err, result = compute_city_snapshot(city_name, strategy=DISPLAY_STRATEGY, fast=True)
    snapshots[city_name] = (df, sigma, labels, err)

    # Confidence % (informational only; does not affect picks)
    conf_pct = None
    try:
        if result is not None:
            _c = result.get("confidence")
            if _c is not None:
                conf_pct = float(_c) * 100.0
    except Exception:
        conf_pct = None

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
            # per-city tables now use Forecast win %
            _m = top_city.get("Forecast win %", None)
            model_prob = (float(_m) / 100.0) if (_m is not None and pd.notna(_m)) else None
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
    if DO_LOCK_0930 and hasattr(pe, "perf_log_snapshot"):
        df_l, best_l, sigma_l, labels_l, err_l = compute_city_snapshot(city_name, strategy="lock_0930", fast=True)
        if best_l is not None:
            try:
                pe.perf_log_snapshot(
                    date_s=pe._today_local_date_str(),
                    city=city_name,
                    station=CITIES[city_name]["station_obs"],
                    sigma_f=sigma_l,
                    labels=labels_l,
                    best_contract=best_l.get("Contract"),
                    yes_ask_prob=(best_l.get("YES ask %")/100.0 if best_l.get("YES ask %") is not None else None),
                    model_prob=(best_l.get("Forecast win %")/100.0 if best_l.get("Forecast win %") is not None else None),
                    value_prob=(best_l.get("Value %")/100.0 if best_l.get("Value %") is not None else None),
                    strategy="lock_0930",
                )
                logged_any_0930 = True
            except Exception:
                pass


    if DO_LOCK_1200 and hasattr(pe, "perf_log_snapshot"):
        df_l, best_l, sigma_l, labels_l, err_l = compute_city_snapshot(city_name, strategy="lock_1200", fast=True)
        if best_l is not None:
            try:
                pe.perf_log_snapshot(
                    date_s=pe._today_local_date_str(),
                    city=city_name,
                    station=CITIES[city_name]["station_obs"],
                    sigma_f=sigma_l,
                    labels=labels_l,
                    best_contract=best_l.get("Contract"),
                    yes_ask_prob=(best_l.get("YES ask %")/100.0 if best_l.get("YES ask %") is not None else None),
                    model_prob=(best_l.get("Forecast win %")/100.0 if best_l.get("Forecast win %") is not None else None),
                    value_prob=(best_l.get("Value %")/100.0 if best_l.get("Value %") is not None else None),
                    strategy="lock_1200",
                )
                logged_any_1200 = True
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
            "Confidence %": None,
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
            "Model %": best.get("Forecast win %"),
            "Confidence %": conf_pct,
            "Odds": best.get("Odds", ""),
            "σ": sigma,
        })
#
# Create lock files only if we successfully logged at least one row.
if DO_LOCK_0930 and (not locked_0930) and logged_any_0930:
    try:
        with open(lock_file_0930, "w") as f:
            f.write(f"Locked picks for {lock_date_str_cst()} 09:30 CST\n")
    except Exception:
        pass

if DO_LOCK_1200 and (not locked_1200) and logged_any_1200:
    try:
        with open(lock_file_1200, "w") as f:
            f.write(f"Locked picks for {lock_date_str_cst()} 12:00 CST\n")
    except Exception:
        pass

load_status.empty()

lb = pd.DataFrame(leader_rows)

# Rename leaderboard columns for clarity
lb = lb.rename(columns={"Acc score %": "Final rank %", "Model %": "Forecast win %"})

# Ensure numeric columns are real numbers (None -> NaN) so Styler formatters
# don't crash with "unsupported format string passed to NoneType".
for _col in ["Final rank %", "Confidence %", "Value %", "YES ask %", "Forecast win %", "σ"]:
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
    df0 = snapshots.get(city, (None, None, None, "", None))[0]
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
    "Confidence %",
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
          {"Final rank %": "{:.1f}%", "Confidence %": "{:.0f}%", "Value %": "{:+.1f}%", "YES ask %": "{:.1f}%", "Forecast win %": "{:.1f}%", "σ": "{:.2f}"},
          na_rep="—",
      )
      .map(value_color, subset=["Value %"])
)

# -----------------------
# NEW UI: Tabbed dashboard flow (UI-only)
# -----------------------

# Market health: count cities with a non-empty df snapshot and no error
try:
    market_ok = 0
    market_total = len(CITIES)
    for c in CITIES.keys():
        df0 = snapshots.get(c, (None, None, None, "", None))[0]
        err0 = snapshots.get(c, (None, None, None, "", None))[3] if c in snapshots else ""
        if df0 is not None and not getattr(df0, "empty", True) and not err0:
            market_ok += 1
except Exception:
    market_ok, market_total = 0, len(CITIES)

# Tabs
_tab_dash, _tab_city, _tab_hist, _tab_about = st.tabs(["📊 Dashboard", "🏙️ City", "📚 History", "ℹ️ About"])

with _tab_dash:
    render_hero(deployed_txt)
    render_kpis(market_ok, market_total)

    st.markdown("### Overall best bet")
    st.caption("Accuracy-first (80% model / 20% market). If a market is effectively locked, it's treated as no longer a bet.")
    snapshot_tables = {city: snapshots[city][0] for city in snapshots}
    render_overall_best_bet(snapshot_tables)

    st.markdown("### Today at a glance")
    st.caption("Cards summarize each city. Pending days show ⏳ and are never graded early.")
    render_city_cards(lb)

    st.markdown("### Leaderboard")
    with st.expander("Show full table", expanded=False):
        st.caption(
            "Legend: Forecast win% = model-only win chance. Final rank% = accuracy-first (80% model + 20% market). "
            "Value% = forecast − price (display-only; not the main ranking)."
        )
        st.caption(
            f"Odds guardrails: excluded heavy favorites (<= {ODDS_EXCLUDE_FAVORITE_AT_OR_BELOW}). "
            f"⚠️ warns longshots (>= +{ODDS_WARN_LONGSHOT_AT_OR_ABOVE})."
        )
        st.dataframe(styled_lb, width="stretch", hide_index=True)

    # Surface any non-fatal errors without breaking the dashboard
    errs = {c: snapshots[c][3] for c in snapshots if len(snapshots[c]) > 3 and snapshots[c][3]}
    if errs:
        st.warning(
            "Some live data calls failed (the app will still load):\n"
            + "\n".join([f"- {c}: {m}" for c, m in errs.items()])
        )

with _tab_city:
    st.markdown("## City deep dive")
    st.caption("Inspect the full market table, settlement station observations, and the 12h observed/forecast charts.")

    default_city = (
        lb.dropna(subset=["Value %"]).iloc[0]["City"]
        if (len(lb.dropna(subset=["Value %"])) > 0)
        else "Philadelphia"
    )
    city_pick = st.selectbox("Select a city", lb["City"].tolist(), index=list(lb["City"]).index(default_city))

    df_city, best_city, sigma_city, _labels_city, err_city, _res_city = compute_city_snapshot(city_pick, strategy=DISPLAY_STRATEGY, fast=False)
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

    # Charts: keep your exact existing chart logic (moved into this tab)
    try:
        apply_city(cfg)

        past = pe.nws_obs_past_hours_station(12)
        df_p = pd.DataFrame(past)
        if not df_p.empty:
            df_p = df_p.sort_values("time_local").rename(columns={"time_local": "time"})
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

        fut = pe.nws_hourly_forecast_next_hours(12)
        df_f = pd.DataFrame(fut)
        if not df_f.empty:
            df_f = df_f.sort_values("time_local").rename(columns={"time_local": "time"})
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

with _tab_hist:
    st.markdown("## Historical performance")
    st.caption("Settles using NOAA/NWS observed highs. Pending days remain ⏳ until official highs are available.")

    # Keep existing history logic exactly as before by executing it from here.
    # We re-run the same block by placing it behind a function boundary.

    # --- BEGIN: moved history block (unchanged) ---
    if hasattr(pe, "perf_load_df"):
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

        # The remainder of your historical section was below in the old layout.
        # It is intentionally omitted here to avoid duplicating hundreds of lines.
        # If you want the full history UI inside this tab too, we can move it next.
        st.caption("History loaded. (Next step: move the full history cards + drilldown into this tab.)")
    else:
        st.info("History functions not available in philly_edge.py")
    # --- END: moved history block (minimal) ---

with _tab_about:
    st.markdown("## How to read this")
    st.markdown(
        """
- **Final rank %**: accuracy-first ranking (**80% model / 20% market**). Primary metric.
- **Forecast win %**: model-only win probability.
- **YES ask %**: market-implied probability from the current ask.
- **Value %**: forecast − price (display-only; not the main ranking).
- **Confidence %**: informational (often rises late-day).
- **Pending**: we never grade early. Unsettled days remain **⏳** until NOAA/NWS observed highs are available.
        """
    )

# -----------------------
# LEGACY UI (disabled)
# -----------------------
if False:
    # Original linear layout kept for reference.
    st.caption(
        "Legend: Forecast win% = model-only win chance. Final rank% = accuracy-first (80% model + 20% market). Value% = forecast − price (not the main ranking)."
    )
    st.subheader("Best bet by city (ranked)")
    st.dataframe(styled_lb, width="stretch", hide_index=True)