
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

# --- Stable paths (works locally + on Streamlit Cloud) ---
from pathlib import Path
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
PERF_PATH = DATA_DIR / "performance.csv"
LOG_PATH = DATA_DIR / "cron_track.log"

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


# -----------------------
# Neon Terminal Theme (UI-only)
# -----------------------
NEON_CSS = """
<style>
:root {
  --bg0:#06070b;
  --bg1:#0b1020;
  --card:#0b1226;
  --stroke: rgba(255,255,255,0.10);
  --stroke2: rgba(255,255,255,0.14);
  --txt: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.64);
  --green:#22c55e;
  --red:#ef4444;
  --amber:#fbbf24;
  --cyan:#22d3ee;
  --pink:#fb7185;
  --violet:#a78bfa;
}

.stApp {
  background:
    radial-gradient(1200px 700px at 15% 10%, rgba(34,211,238,0.18), transparent 55%),
    radial-gradient(1000px 600px at 85% 20%, rgba(167,139,250,0.16), transparent 52%),
    radial-gradient(900px 600px at 60% 90%, rgba(251,113,133,0.12), transparent 55%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
}

.block-container { padding-top: 1.25rem; }
h1,h2,h3,h4,h5,h6,p,div,span { color: var(--txt); }

[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--stroke);
  background: rgba(11,18,38,0.55);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(11,16,32,0.92) 0%, rgba(8,10,18,0.92) 100%);
  border-right: 1px solid var(--stroke);
}

.stButton>button {
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: linear-gradient(180deg, rgba(34,211,238,0.22) 0%, rgba(167,139,250,0.18) 100%);
  color: var(--txt);
}
.stButton>button:hover { border-color: rgba(34,211,238,0.45); }

.we-chip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  border:1px solid var(--stroke);
  background: rgba(255,255,255,0.06);
  font-size:12px; color: var(--txt);
}
.we-chip b { color: var(--txt); }

.we-panel {
  padding:14px 16px; border-radius:16px;
  border:1px solid var(--stroke);
  background: linear-gradient(180deg, rgba(11,18,38,0.62) 0%, rgba(8,10,18,0.45) 100%);
}
.we-panel-title { font-size:13px; opacity:0.72; margin-bottom:8px; }

.we-title { font-size:44px; font-weight:900; letter-spacing:-0.03em; line-height:1; }
.we-sub { opacity:0.72; margin-top:6px; }

.we-badge {
  display:inline-flex; align-items:center; gap:10px;
  padding:10px 12px; border-radius:14px;
  border:1px solid var(--stroke2);
  background:
    radial-gradient(500px 100px at 10% 20%, rgba(34,211,238,0.24), transparent 55%),
    radial-gradient(500px 100px at 90% 20%, rgba(167,139,250,0.20), transparent 55%),
    rgba(255,255,255,0.05);
}

.we-kpi { display:flex; flex-direction:column; gap:2px; }
.we-kpi .k { font-size:12px; opacity:0.70; }
.we-kpi .v { font-size:26px; font-weight:850; }
.we-kpi .s { font-size:12px; opacity:0.70; }

.we-green { color: var(--green); }
.we-red { color: var(--red); }
.we-amber { color: var(--amber); }
.we-cyan { color: var(--cyan); }

/* Subtle motion + confidence ramp */
@keyframes weGlowPulse {
  0% { box-shadow: 0 0 0 rgba(34,211,238,0.0), 0 0 0 rgba(167,139,250,0.0); }
  50% { box-shadow: 0 0 18px rgba(34,211,238,0.14), 0 0 22px rgba(167,139,250,0.12); }
  100% { box-shadow: 0 0 0 rgba(34,211,238,0.0), 0 0 0 rgba(167,139,250,0.0); }
}

@keyframes weShimmer {
  0% { transform: translateX(-40%); opacity: 0.0; }
  20% { opacity: 0.35; }
  50% { opacity: 0.15; }
  100% { transform: translateX(140%); opacity: 0.0; }
}

.we-motion {
  transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
}
.we-motion:hover {
  transform: translateY(-2px);
  border-color: rgba(34,211,238,0.35);
}

.we-pulse {
  animation: weGlowPulse 4.2s ease-in-out infinite;
}

.we-ramp {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid var(--stroke2);
  background:
    radial-gradient(420px 90px at 10% 20%, rgba(34,211,238,0.18), transparent 55%),
    radial-gradient(420px 90px at 90% 20%, rgba(167,139,250,0.15), transparent 55%),
    rgba(255,255,255,0.05);
  overflow: hidden;
}

.we-ramp::after {
  content: "";
  position: absolute;
  top: -40%;
  left: -60%;
  width: 55%;
  height: 180%;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.10) 50%, transparent 100%);
  transform: translateX(-40%);
  animation: weShimmer 6s ease-in-out infinite;
  pointer-events: none;
}

.we-ramp-label { font-size: 12px; opacity: 0.75; }
.we-ramp-val { font-size: 14px; font-weight: 800; }
.we-ramp-track {
  width: 140px;
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
}
.we-ramp-fill {
  height: 100%;
  width: var(--pct);
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(34,211,238,0.55), rgba(167,139,250,0.55), rgba(251,113,133,0.35));
  transition: width 600ms ease;
}
</style>
"""

def _ui_inject_neon():
    st.markdown(NEON_CSS, unsafe_allow_html=True)

def chip(html_text: str) -> str:
    return f"<span class='we-chip we-motion'>{html_text}</span>"

def panel(title: str, inner_html: str) -> str:
    return f"<div class='we-panel we-motion'><div class='we-panel-title'>{title}</div>{inner_html}</div>"

def kpi(label: str, value: str, sub: str = "") -> str:
    return (
        "<div class='we-kpi'>"
        f"<div class='k'>{label}</div>"
        f"<div class='v'>{value}</div>"
        f"<div class='s'>{sub}</div>"
        "</div>"
    )

# -----------------------
# UI-only: Signal chips + helpers
# -----------------------

def _to_num(x):
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _is_bad_status(s: str) -> bool:
    s = (s or "").lower()
    return ("locked" in s) or ("⛔" in s) or ("🔒" in s) or ("not viable" in s)


def _signal_from_row(row: dict) -> tuple:
    """Return (signal_label, emoji, css_class) from existing metrics.
    UI-only: uses Final rank %, Status, Odds guardrails, and presence of data.
    """
    status = str(row.get("Status", "") or "")
    best = str(row.get("Best contract", "") or "")

    fr = _to_num(row.get("Final rank %"))  # percent units
    conf = _to_num(row.get("Confidence %")) if "Confidence %" in row else None

    # Missing data
    if (not best) or ("no market data" in best.lower()) or (fr is None):
        return ("NO DATA", "📡", "we-amber")

    # Locked / not viable
    if _is_bad_status(status):
        return ("NO PLAY", "🔒", "we-red")

    # Odds guardrail expensive favorites
    odds_s = str(row.get("Odds", "") or "")
    if is_odds_too_expensive(odds_s):
        return ("NO PLAY", "💸", "we-red")

    # Heuristic thresholds (UI-only)
    if fr >= 60.0 and (conf is None or conf >= 55.0):
        return ("BET", "🟢", "we-green")

    if fr >= 45.0:
        return ("WATCH", "🟡", "we-amber")

    return ("NO PLAY", "🔴", "we-red")


def signal_chip(label: str, emoji: str, css_class: str) -> str:
    return (
        "<span class='we-chip' style='border-color:rgba(255,255,255,0.18);'>"
        f"<span class='{css_class}' style='font-weight:800;'>{emoji} {label}</span>"
        "</span>"
    )

# Set Streamlit page config
st.set_page_config(page_title="Weather Edge", layout="wide")

_ui_inject_neon()

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


# -----------------------
# UI-only: Confidence ramp (early → mid → late)
# -----------------------

def confidence_ramp_pct(now_dt: Optional[datetime] = None) -> float:
    """0..100 ramp based on time of day relative to lock windows.
    UI-only heuristic: not a model change.
    """
    try:
        n = now_dt or now_cst()
        # Define a smooth ramp from 07:00 → 17:00 CST
        start = n.replace(hour=7, minute=0, second=0, microsecond=0)
        end = n.replace(hour=17, minute=0, second=0, microsecond=0)
        if n <= start:
            return 12.0
        if n >= end:
            return 100.0
        span = (end - start).total_seconds()
        t = (n - start).total_seconds() / span
        # ease-in-out curve
        eased = (1 - (1 - t) * (1 - t)) if t < 0.5 else (t * t * (3 - 2 * t))
        return max(0.0, min(100.0, 12.0 + 88.0 * eased))
    except Exception:
        return 50.0


def ramp_stage(pct: float) -> str:
    if pct >= 85:
        return "Late"
    if pct >= 45:
        return "Mid"
    return "Early"


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
# Neon Terminal — Header + Control Room (UI-only)
# -----------------------

now_txt = now_cst().strftime('%a %b %-d, %Y · %-I:%M %p CST')
strategy_live = "lock_1200" if is_after_lock2_cst() else "lock_0930"

# Compute ramp variables for UI badge
ramp_pct = confidence_ramp_pct()
ramp_lbl = ramp_stage(ramp_pct)
ramp_pct_str = f"{ramp_pct:.0f}%"

st.markdown(
    f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:14px;flex-wrap:wrap;">
  <div>
    <div class="we-title">Weather Edge</div>
    <div class="we-sub">🌌 Neon Terminal · 7-city daily highs · accuracy-first (80% model / 20% market)</div>
    <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
      {chip('🕒 Now: <b>' + now_txt + '</b>')}
      {chip('🧭 Live strategy: <b>' + strategy_live + '</b>')}
      {chip('🧾 Settlement: <b>NOAA/NWS observed highs</b>')}
      <span class="we-ramp we-motion" style="--pct:{ramp_pct_str};">
        <span class="we-ramp-label">📶 Confidence ramp</span>
        <span class="we-ramp-val">{ramp_lbl} · {ramp_pct_str}</span>
        <span class="we-ramp-track"><span class="we-ramp-fill"></span></span>
      </span>
    </div>
  </div>
  <div style="text-align:right;opacity:0.85;">
    <div style="font-size:12px;opacity:0.75;">Deploy</div>
    <div style="font-size:14px;font-weight:700;">{DEPLOY_SHA}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

deployed_txt = (
    DEPLOYED_AT_ET.strftime('%Y-%m-%d %I:%M %p %Z')
    if DEPLOYED_AT_ET is not None
    else "unknown"
)

with st.sidebar:
    st.markdown("## 🎛 Control Room")
    st.caption(f"Commit: `{DEPLOY_SHA}` · deployed {deployed_txt}")

    if st.button("🔄 Refresh markets"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### ⏱ Lock windows")
    st.write(f"🕤 09:30 CST: {'✅ after' if is_after_lock_cst() else '⏳ before'}")
    st.write(f"🕛 12:00 CST: {'✅ after' if is_after_lock2_cst() else '⏳ before'}")

    # UI-only: late-day stabilization indicator
    try:
        if is_after_lock2_cst():
            st.caption("✅ Late-day mode: highs often stabilized → confidence naturally rises")
        elif is_after_lock_cst():
            st.caption("🟡 Mid-day mode: confidence ramps as the day progresses")
        else:
            st.caption("🔵 Early mode: confidence reflects forecast uncertainty")
    except Exception:
        pass

    st.divider()
    st.markdown("### 🧪 Filters")
    show_only_viable = st.checkbox("Hide locked/not viable", value=False)
    show_debug = st.checkbox("Show diagnostics", value=False)

    # Repo-relative path so Streamlit Cloud can always find history
    perf_path = str(PERF_PATH)

#
# Global strategy used by snapshot/model calls
strategy = "lock_1200" if is_after_lock2_cst() else "lock_0930"

load_status = st.empty()
load_status.info("Connecting to markets…")



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
        result = pe.compute_pick_for_today(strategy=strategy)  # UPDATED
        probs = result['probs']
        # old: pe.model_probs_for_buckets(bucket_bounds, sigma)
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

    return df, best, sigma, labels, ""

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

# Track whether we actually logged anything; only then do we create the lock file.
logged_any_0930 = False
logged_any_1200 = False

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
                model_prob=(best.get("Forecast win %")/100.0 if best.get("Forecast win %") is not None else None),
                value_prob=(best.get("Value %")/100.0 if best.get("Value %") is not None else None),
                strategy="lock_0930",
            )
            logged_any_0930 = True
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
                model_prob=(best.get("Forecast win %")/100.0 if best.get("Forecast win %") is not None else None),
                value_prob=(best.get("Value %")/100.0 if best.get("Value %") is not None else None),
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
    "Signal",
    "Best contract",
    "Final rank %",
    "Forecast win %",
    "YES ask %",
    "Value %",
    "Odds",
    "σ",
]
lb = lb[[c for c in _cols if c in lb.columns]]

# UI-only: add Signal column
try:
    sigs = []
    for _, r in lb.iterrows():
        lab, emo, cls = _signal_from_row(r.to_dict())
        sigs.append(f"{emo} {lab}")
    if "Signal" not in lb.columns:
        lb.insert(2, "Signal", sigs)
except Exception:
    pass

# UI-only: subtle pulse around the best-bet area
best_bet_slot = st.container()
st.markdown("<div class='we-pulse'>", unsafe_allow_html=True)
snapshot_tables = {city: snapshots[city][0] for city in snapshots}
with best_bet_slot:
    render_overall_best_bet(snapshot_tables)
st.markdown("</div>", unsafe_allow_html=True)

# UI-only: Top 3 opportunities
try:
    st.markdown("### 🥇 Top 3 opportunities")
    top3 = lb.copy()
    if "Final rank %" in top3.columns:
        top3 = top3.sort_values("Final rank %", ascending=False)
    top3 = top3.head(3)

    cards = st.columns(3)
    for i, (_, r) in enumerate(top3.iterrows()):
        city = str(r.get("City", ""))
        contract = str(r.get("Best contract", ""))
        fr = r.get("Final rank %")
        v = r.get("Value %")
        stt = str(r.get("Status", ""))
        sig_lab, sig_emo, sig_cls = _signal_from_row(r.to_dict())

        fr_txt = "—" if pd.isna(fr) else f"{float(fr):.1f}%"
        v_txt = "—" if pd.isna(v) else f"{float(v):+.1f}%"

        inner = (
            f"<div style='display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;'>"
            f"<div style='font-size:16px;font-weight:850;'>{city}</div>"
            f"<div>{signal_chip(sig_lab, sig_emo, sig_cls)}</div>"
            f"</div>"
            f"<div style='margin-top:6px;opacity:0.78;'>Best: <b>{contract}</b></div>"
            f"<div style='margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;'>"
            f"{chip('🏁 Final rank: <b>' + fr_txt + '</b>')}"
            f"{chip('📈 Value: <b>' + v_txt + '</b>')}"
            f"</div>"
            f"<div style='margin-top:8px;opacity:0.70;'>Status: {stt}</div>"
        )
        cards[i].markdown(panel(f"#{i+1}", inner), unsafe_allow_html=True)
except Exception:
    pass

# Show any non-fatal data errors so the page doesn't look "blank" when an API call fails
errs = {c: snapshots[c][3] for c in snapshots if len(snapshots[c]) > 3 and snapshots[c][3]}
if errs:
    st.warning(
        "Some live data calls failed (the app will still load):\n"
        + "\n".join([f"- {c}: {m}" for c, m in errs.items()])
    )

# UI-only styling: emphasize rank/status/signal and warn on longshots

def _rank_color(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        v = float(v)
        if v >= 60:
            return "color:#22c55e;font-weight:700;"
        if v >= 45:
            return "color:#fbbf24;font-weight:700;"
        return "color:#ef4444;"
    except Exception:
        return ""


def _status_style(s):
    s2 = str(s or "")
    if "⛔" in s2 or "Not viable" in s2:
        return "color:#ef4444;font-weight:700;"
    if "🔒" in s2 or "Locked" in s2:
        return "color:#fbbf24;font-weight:700;"
    return "color:#22d3ee;"


def _signal_style(s):
    s2 = str(s or "")
    if "BET" in s2 or "🟢" in s2:
        return "color:#22c55e;font-weight:800;"
    if "WATCH" in s2 or "🟡" in s2:
        return "color:#fbbf24;font-weight:800;"
    if "NO DATA" in s2 or "📡" in s2:
        return "color:#fbbf24;font-weight:800;"
    return "color:#ef4444;font-weight:800;"


def _odds_warn(s):
    return "color:#fbbf24;font-weight:800;" if is_odds_longshot(str(s or "")) else ""

styled_lb = (
    lb.style
      .format(
          {"Final rank %": "{:.1f}%", "Value %": "{:+.1f}%", "YES ask %": "{:.1f}%", "Forecast win %": "{:.1f}%", "σ": "{:.2f}"},
          na_rep="—",
      )
      .map(value_color, subset=["Value %"])
      .map(_rank_color, subset=["Final rank %"])
      .map(_status_style, subset=["Status"])
      .map(_signal_style, subset=["Signal"])
      .map(_odds_warn, subset=["Odds"])
)

st.caption(
    "Legend: Forecast win% = model-only win chance. Final rank% = accuracy-first (80% model + 20% market). Value% = forecast − price (not the main ranking)."
)
st.subheader("Best bet by city (ranked)")
st.caption(f"Odds guardrails: excluded heavy favorites (<= {ODDS_EXCLUDE_FAVORITE_AT_OR_BELOW}). ⚠️ warns longshots (>= +{ODDS_WARN_LONGSHOT_AT_OR_ABOVE}) — consider avoiding unless you have a strong edge.")
st.dataframe(styled_lb, width="stretch", hide_index=True)

 # -----------------------
# City view + settlement station label + forecast graph
# -----------------------

# UI-only: City Compare tiles
try:
    st.markdown("### 🧩 City Compare")
    tiles = lb.copy()
    rows = [tiles.iloc[:4], tiles.iloc[4:7]]

    for row_df in rows:
        cols = st.columns(len(row_df))
        for c, (_, r) in zip(cols, row_df.iterrows()):
            city = str(r.get("City", ""))
            contract = str(r.get("Best contract", ""))
            fr = r.get("Final rank %")
            stt = str(r.get("Status", ""))

            sig_lab, sig_emo, sig_cls = _signal_from_row(r.to_dict())
            fr_txt = "—" if pd.isna(fr) else f"{float(fr):.0f}%"

            inner = (
                f"<div style='display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;'>"
                f"<div style='font-size:14px;font-weight:850;'>{city}</div>"
                f"<div>{signal_chip(sig_lab, sig_emo, sig_cls)}</div>"
                f"</div>"
                f"<div style='margin-top:6px;opacity:0.78; font-size:13px;'><b>{contract}</b></div>"
                f"<div style='margin-top:8px;display:flex;gap:10px;flex-wrap:wrap;'>"
                f"{chip('🏁 ' + fr_txt)}"
                f"</div>"
                f"<div style='margin-top:8px;opacity:0.70;font-size:12px;'>{stt}</div>"
            )
            c.markdown(panel("", inner), unsafe_allow_html=True)
except Exception:
    pass
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

# -----------------------
# CSV safety / repair helpers (local + deployed)
# -----------------------

import csv
import re

def _repair_perf_row(fields: list, n_expected: int) -> list:
    """Repair a row with the wrong field count by assuming the extra columns
    belong to `labels_json` (column index 4). This fixes cases where JSON wasn't
    properly quoted and got split by commas.

    Expected columns (n_expected) match PERF_COLUMNS in track.py:
    date, city, station, sigma_f, labels_json, best_contract, yes_ask_prob,
    model_prob, value_prob, observed_high_f, winning_contract, won, profit,
    computed_winning, computed_won, strategy
    """
    if fields is None:
        return [""] * n_expected

    # Too few fields -> pad
    if len(fields) < n_expected:
        return fields + ([""] * (n_expected - len(fields)))

    # Exact -> ok
    if len(fields) == n_expected:
        return fields

    # Too many -> merge the extra pieces into labels_json (index 4)
    # labels_json is the 5th column (0-based index 4)
    extras = len(fields) - n_expected
    # Merge fields[4 : 5+extras] into one comma-joined labels_json
    merged_labels = ",".join(fields[4:5 + extras])
    repaired = fields[:4] + [merged_labels] + fields[5 + extras:]

    # If still off, hard-trim/pad defensively
    if len(repaired) > n_expected:
        repaired = repaired[:n_expected]
    if len(repaired) < n_expected:
        repaired = repaired + ([""] * (n_expected - len(repaired)))
    return repaired


def safe_load_performance_csv(path: Path):
    """Load performance.csv even if 1+ lines are malformed.

    Returns: (df, info_dict)
      info_dict = {
        'expected_fields': int,
        'bad_line_count': int,
        'bad_lines': list of (lineno, field_count, preview)
        'repaired': bool
      }
    """
    info = {
        "expected_fields": 0,
        "bad_line_count": 0,
        "bad_lines": [],
        "repaired": False,
    }

    if path is None or (not Path(path).exists()):
        return pd.DataFrame(), info

    # First try: normal pandas read
    try:
        df0 = pd.read_csv(path)
        info["expected_fields"] = len(df0.columns)
        return df0, info
    except Exception:
        pass

    # Fallback: repair line-by-line
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return pd.DataFrame(), info

        header = rows[0]
        n_expected = len(header)
        info["expected_fields"] = n_expected

        fixed_rows = [header]
        for i, r in enumerate(rows[1:], start=2):  # 1-based line numbers; header is line 1
            if r is None:
                continue
            if len(r) != n_expected:
                info["bad_line_count"] += 1
                preview = ",".join(r)[:220]
                info["bad_lines"].append((i, len(r), preview))
                r = _repair_perf_row(r, n_expected)
                info["repaired"] = True
            fixed_rows.append(r)

        df = pd.DataFrame(fixed_rows[1:], columns=fixed_rows[0])
        return df, info
    except Exception as e:
        # Last resort: attempt python engine + skipping bad lines
        try:
            df1 = pd.read_csv(path, engine="python", on_bad_lines="skip")
            info["expected_fields"] = len(df1.columns)
            info["repaired"] = True
            return df1, info
        except Exception:
            st.error(f"Failed to load performance history: {e}")
            return pd.DataFrame(), info


def try_repair_performance_csv_in_place(path: Path) -> str:
    """Local-only helper: rewrite a cleaned CSV to disk (keeps a backup).
    Returns a status message.
    """
    try:
        p = Path(path)
        if not p.exists():
            return "No performance.csv found to repair."

        df, info = safe_load_performance_csv(p)
        if df.empty:
            return "Repair attempted, but resulting table is empty."

        backup = p.with_suffix(p.suffix + ".bak")
        # Avoid overwriting an existing backup repeatedly
        if not backup.exists():
            backup.write_bytes(p.read_bytes())

        df.to_csv(p, index=False)
        return f"Rewrote performance.csv (backup: {backup.name}). Bad lines detected: {info['bad_line_count']}"
    except Exception as e:
        return f"Repair failed: {e}"

if hasattr(pe, "perf_load_df"):
    st.subheader("Historical performance")
    # --- History file diagnostics (helps when live shows "no history") ---
    with st.expander("📁 History file status", expanded=False):
        st.write("PERF_PATH:", str(PERF_PATH))
        st.write("Exists:", PERF_PATH.exists())
        if PERF_PATH.exists():
            try:
                st.write("Size (bytes):", PERF_PATH.stat().st_size)
                # show header + first 3 rows for sanity
                _tmp = pd.read_csv(PERF_PATH).head(3)
                st.dataframe(_tmp, width="stretch", hide_index=True)
            except Exception as _e:
                st.write("Preview failed:", str(_e))

            # Also show a safe-load diagnostics report
            _df_safe, _info = safe_load_performance_csv(PERF_PATH)
            if _info.get("bad_line_count", 0) > 0:
                st.write("Malformed line(s) detected:", _info["bad_line_count"])
                st.write("First few bad lines (line_no, field_count, preview):")
                st.write(_info["bad_lines"][:5])

                if st.button("🛠 Repair performance.csv (local)"):
                    msg = try_repair_performance_csv_in_place(PERF_PATH)
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.write("CSV looks structurally OK (no malformed lines detected).")

    if not PERF_PATH.exists():
        st.info(
            "No performance.csv found in the deployed repo (`data/performance.csv`), so there’s no history to show. "
            "If GitHub Actions is green, make sure the workflow is committing `data/performance.csv` to `main`."
        )
        perf = pd.DataFrame()
    else:
        # Try the engine helper first (keeps schema consistent), but fall back
        # to a safe loader that can repair malformed CSV rows.
        try:
            perf = pe.perf_load_df()
        except Exception as e:
            perf, info = safe_load_performance_csv(PERF_PATH)
            if info.get("bad_line_count", 0) > 0:
                st.warning(
                    f"performance.csv had {info['bad_line_count']} malformed line(s). "
                    "Loaded with automatic repair so history can render. "
                    "Open the History file status expander to see details."
                )
            else:
                st.warning(f"perf_load_df() failed; loaded CSV directly: {e}")

        # If pe.perf_load_df() succeeded but pandas would fail later due to bad CSV,
        # we still keep a diagnostics view via the safe loader.
        if "perf" not in locals() or perf is None:
            perf, _ = safe_load_performance_csv(PERF_PATH)
    # Treat a row as "settled" if we have an observed high. Profit may legitimately be NaN
    # (e.g., price/fee not captured, or older rows), so don't drop rows on profit.
    perf = perf.copy()

    # --- Live settlement helper ---
    # Deployed Streamlit often can't persist writes back to data/performance.csv,
    # so we compute settlements in-memory for display using NOAA/NWS observed highs.

    @st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
    def _obs_high_cached(date_s: str, station_icao: str, city: str):
        try:
            return pe._fetch_observed_daily_high(date_s, station_icao=station_icao, city=city)
        except Exception:
            return None

    def _settle_perf_in_memory(perf_df: pd.DataFrame, max_days: int = 60) -> pd.DataFrame:
        """Fill observed_high_f / winning_contract / won / profit for past rows in-memory.
        Keeps Historical tab accurate even when the server can't persist file writes.
        """
        if perf_df is None or perf_df.empty:
            return perf_df

        df = perf_df.copy()

        # Ensure required columns exist
        for col in ["observed_high_f", "winning_contract", "won", "profit"]:
            if col not in df.columns:
                df[col] = pd.NA

        # Normalize date to string YYYY-MM-DD
        df["date"] = df["date"].astype(str)

        # Only attempt to settle recent days (keeps it fast)
        try:
            uniq_dates = sorted(df["date"].dropna().unique().tolist(), reverse=True)[:max_days]
            df = df[df["date"].isin(uniq_dates)].copy()
        except Exception:
            pass

        # Fill observed highs for rows missing them
        obs_series = pd.to_numeric(df["observed_high_f"], errors="coerce")
        missing_mask = obs_series.isna()

        if missing_mask.any():
            # Determine station per row (prefer explicit station column; else map from city)
            if "station" in df.columns:
                stations = df["station"].astype(str)
            else:
                stations = df["city"].map(lambda c: CITIES.get(str(c), {}).get("station_obs", ""))

            for idx in df[missing_mask].index:
                date_s = str(df.at[idx, "date"])
                city = str(df.at[idx, "city"]) if "city" in df.columns else ""
                station = str(stations.at[idx]) if idx in stations.index else ""
                if not station:
                    station = CITIES.get(city, {}).get("station_obs", "")
                if not station:
                    continue

                obs = _obs_high_cached(date_s, station, city)
                if obs is None:
                    continue

                try:
                    df.at[idx, "observed_high_f"] = float(obs)
                except Exception:
                    df.at[idx, "observed_high_f"] = pd.NA

        # Normalize observed highs to numeric
        df["observed_high_f"] = pd.to_numeric(df["observed_high_f"], errors="coerce")

        # Recompute winning contract when possible (using labels_json)
        if "labels_json" in df.columns:
            def _parse_bucket_label(lbl: str):
                if not isinstance(lbl, str):
                    return None
                s = lbl.strip().replace("º", "°")
                s = re.sub(r"\s+", " ", s)

                m = re.match(r"^(\-?\d+)\s*°\s*to\s*(\-?\d+)\s*°$", s)
                if m:
                    return (float(m.group(1)), float(m.group(2)), "range")

                m = re.match(r"^(\-?\d+)\s*°\s*or\s*below$", s)
                if m:
                    return (None, float(m.group(1)), "below")

                m = re.match(r"^(\-?\d+)\s*°\s*or\s*above$", s)
                if m:
                    return (float(m.group(1)), None, "above")

                return None

            def _winner_from_observed(obs_f: float, labels_json: str):
                if obs_f is None or (isinstance(obs_f, float) and pd.isna(obs_f)):
                    return None
                try:
                    labels = json.loads(labels_json) if isinstance(labels_json, str) else None
                except Exception:
                    labels = None
                if not isinstance(labels, list) or not labels:
                    return None

                x = float(obs_f)
                for lbl in labels:
                    spec = _parse_bucket_label(str(lbl))
                    if spec is None:
                        continue
                    lo, hi, kind = spec
                    if kind == "range" and (x >= lo) and (x <= hi):
                        return str(lbl)
                    if kind == "below" and (x <= hi):
                        return str(lbl)
                    if kind == "above" and (x >= lo):
                        return str(lbl)
                return None

            df["computed_winning_contract"] = [
                _winner_from_observed(o, lj)
                for o, lj in zip(df.get("observed_high_f"), df.get("labels_json"))
            ]

            comp = pd.Series(df["computed_winning_contract"], index=df.index)
            df["winning_contract"] = comp.where(comp.notna(), df.get("winning_contract"))

        # Recompute win flag from best_contract vs winner.
        # IMPORTANT: Only mark a bet as settled if we have an observed high AND we can identify the winning bucket.
        if "best_contract" in df.columns:
            # Only mark a bet as settled if we have an observed high AND we can identify the winning bucket.
            obs_ok = pd.to_numeric(df.get("observed_high_f"), errors="coerce").notna()
            wc_ok = df.get("winning_contract").notna()
            bc_ok = df.get("best_contract").notna()
            known = obs_ok & wc_ok & bc_ok

            df["won"] = pd.NA
            # Compare as strings but do NOT coerce NaNs to the literal string "nan".
            bc = df.loc[known, "best_contract"].astype(str)
            wc = df.loc[known, "winning_contract"].astype(str)
            df.loc[known, "won"] = (bc == wc).astype(float)

        # Recompute profit only for settled rows where we know won and price.
        # Profit per $1 YES contract: win => (1 - price), lose => (-price)
        if "yes_ask_prob" in df.columns:
            price = pd.to_numeric(df["yes_ask_prob"], errors="coerce")
            won_num = pd.to_numeric(df["won"], errors="coerce")
            df["profit"] = pd.NA
            m = won_num.notna() & price.notna()
            df.loc[m, "profit"] = won_num[m] * (1 - price[m]) + (1 - won_num[m]) * (-price[m])

        return df

    # Normalize strategy labels so 09:30 and 12:00 always group correctly
    if "strategy" not in perf.columns:
        perf["strategy"] = "lock_0930"
    perf["strategy"] = (
        perf["strategy"]
            .astype(str)
            .fillna("lock_0930")
            .str.strip()
            .str.lower()
    )
    perf["strategy"] = perf["strategy"].replace({
        "lock0930": "lock_0930",
        "lock-0930": "lock_0930",
        "0930": "lock_0930",
        "lock1200": "lock_1200",
        "lock-1200": "lock_1200",
        "1200": "lock_1200",
        "noon": "lock_1200",
    })

    # Debug visibility (kept compact): helps confirm whether 12:00 rows exist on the live server
    with st.expander("🧪 History diagnostics", expanded=False):
        show_debug_hist = st.checkbox("Show debug details", value=False)
    if show_debug_hist or show_debug:
        with st.expander("Debug: history rows by strategy (raw)"):
            try:
                st.write(perf["strategy"].value_counts(dropna=False))
                if "observed_high_f" in perf.columns:
                    st.write("Settled rows (observed_high_f present) by strategy")
                    st.write(
                        perf.loc[
                            pd.to_numeric(perf["observed_high_f"], errors="coerce").notna(),
                            "strategy",
                        ].value_counts(dropna=False)
                    )
            except Exception as _e:
                st.write(f"(debug failed: {_e})")

    # Compute settlements in-memory so the deployed site can still show true history
    perf = _settle_perf_in_memory(perf, max_days=60)

    # Use only rows that have observed highs
    if "observed_high_f" in perf.columns:
        _obs = pd.to_numeric(perf["observed_high_f"], errors="coerce")
    else:
        _obs = pd.Series([float("nan")] * len(perf), index=perf.index)

    settled_count = int(_obs.notna().sum()) if len(perf) else 0
    total_count = int(len(perf))

    st.caption(f"History rows present: {total_count} (settled: {settled_count})")

    done = perf[_obs.notna()].copy()

    if done.empty:
        st.info(
            "No settled history to display yet.\n\n"
            "If you *expect* settled rows (e.g., yesterday finished), the most common causes are:\n"
            "• NOAA/NWS observed high fetch failed for that station/date\n"
            "• the date is too recent and hasn't published final daily max yet\n\n"
            "Tip: refresh once, and check for a warning above about outcome-update failure."
        )
    else:
        # Normalize types
        done["profit"] = pd.to_numeric(done["profit"], errors="coerce")
        if "won" in done.columns:
            done["won"] = pd.to_numeric(done["won"], errors="coerce")
        # Strategy already normalized above; keep only the lock strategies we care about
        keep_strats = ["lock_0930", "lock_1200"]
        done = done[done["strategy"].isin(keep_strats)].copy()

        # Limit history for speed (last N dates with settled outcomes)
        MAX_HISTORY_DAYS = 30
        try:
            _recent_dates = sorted(done["date"].unique(), reverse=True)[:MAX_HISTORY_DAYS]
            done = done[done["date"].isin(_recent_dates)].copy()
        except Exception:
            pass

        # -------------------------
        # Simplified daily view
        # -------------------------
        st.markdown("### Daily results (7 cities) — 09:30 CST and 12:00 CST")
        st.caption(
            "Cards show wins out of 7 for each lock time. "
            "Use the drilldown below to see the exact pick vs. actual winning bucket (green = win, red = loss)."
        )

        # Dedupe settled rows so a city only counts once per date/strategy
        done_dedup = done.copy()
        if all(c in done_dedup.columns for c in ["date", "strategy", "city"]):
            # keep the last occurrence for a city/date/strategy
            done_dedup = (
                done_dedup
                .sort_values(["date", "strategy"], ascending=[False, True])
                .drop_duplicates(["date", "strategy", "city"], keep="last")
            )

        # Aggregate only on rows where the win/loss is known (prevents unknown winners from showing as 0/7).
        _won_num = pd.to_numeric(done_dedup.get("won"), errors="coerce")
        known_mask = _won_num.notna()
        daily = (
            done_dedup.loc[known_mask]
            .groupby(["date", "strategy"], as_index=False)
            .agg(bets=("city", "nunique"), wins=("won", "sum"))
        )

        # Include dates that have lock rows but are not settled yet (so we can show the ⏳ pending state)
        dates_done = done["date"].dropna().astype(str).unique().tolist()
        dates_all = (
            perf.loc[perf["strategy"].isin(keep_strats), "date"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
        )
        dates = sorted(set(dates_all) | set(dates_done), reverse=True)

        def _wl_tuple(date_s: str, strat: str):
            sub = daily[(daily["date"].astype(str) == str(date_s)) & (daily["strategy"] == strat)]
            if sub.empty:
                return (0, 0)
            b = int(float(sub.iloc[0]["bets"])) if pd.notna(sub.iloc[0]["bets"]) else 0
            w = int(float(sub.iloc[0]["wins"])) if pd.notna(sub.iloc[0]["wins"]) else 0
            return (w, b)

        def _lock_state(date_s: str, strat: str):
            """Return (has_lock_rows, all_settled, n_cities) for a date/strategy."""
            sub_all = perf[(perf["date"].astype(str) == str(date_s)) & (perf["strategy"] == strat)].copy()
            if sub_all.empty:
                return (False, False, 0)

            # Count unique cities for the bet count
            n_cities = int(sub_all["city"].nunique()) if "city" in sub_all.columns else int(len(sub_all))

            obs_all = pd.to_numeric(sub_all.get("observed_high_f"), errors="coerce")
            won_all = pd.to_numeric(sub_all.get("won"), errors="coerce")
            # Fully settled only if every row has an observed high AND a computed win/loss.
            all_settled = bool(len(sub_all) > 0 and obs_all.notna().all() and won_all.notna().all())
            return (True, all_settled, n_cities)

        for d in dates[:30]:
            w0930, b0930 = _wl_tuple(d, "lock_0930")
            w1200, b1200 = _wl_tuple(d, "lock_1200")
            has0930, all_settled0930, n0930 = _lock_state(d, "lock_0930")
            has1200, all_settled1200, n1200 = _lock_state(d, "lock_1200")

            p0930 = bool(has0930 and not all_settled0930)
            p1200 = bool(has1200 and not all_settled1200)

            # Use unique-city counts for the denominator when pending or when settled rows are sparse
            if has0930:
                b0930 = n0930 if n0930 else (b0930 if b0930 else 7)
            if has1200:
                b1200 = n1200 if n1200 else (b1200 if b1200 else 7)

            # If we truly have no rows, keep b=0 so the card can say "No records"

            st.markdown(f"#### {d}")
            c1, c2 = st.columns(2)

            def _card(col, title, w, b, pending: bool = False):
                if pending:
                    badge = "rgba(148,163,184,0.28)"  # stronger gray for pending
                elif w >= 6:
                    badge = "rgba(34,197,94,0.22)"    # green (6–7 wins)
                elif w >= 4:
                    badge = "rgba(250,204,21,0.22)"   # yellow (4–5 wins)
                else:
                    badge = "rgba(239,68,68,0.22)"    # red (0–3 wins)

                col.markdown(
                    f"""
                    <div style=\"padding:12px 14px;border-radius:14px;border:1px solid rgba(255,255,255,0.10);background:{badge};\">
                      <div style=\"font-size:13px;opacity:0.85;margin-bottom:6px;\">{title}</div>
                      <div style="font-size:28px;font-weight:800;line-height:1;">{('—' if pending else str(w))}/{(b if b else 7)}</div>
                      <div style="font-size:12px;opacity:0.8;margin-top:6px;">{('⏳ Waiting for settlement (highs/winner)' if pending else ('Settled' if b > 0 else 'No records'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            _card(c1, "🕤 09:30 CST", w0930, b0930, pending=p0930)
            _card(c2, "🕛 12:00 CST", w1200, b1200, pending=p1200)

        # Drilldown: show the 7 city picks for a chosen date/lock
        st.markdown("### What did it pick?")
        # Include pending days too (so you can see today's picks even before settlement)
        dates_any = (
            perf.loc[perf["strategy"].isin(keep_strats), "date"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
        )
        dates_any = sorted(dates_any, reverse=True)

        if perf.empty:
            st.info("No history rows loaded yet.")
            st.stop()
        if dates_any:
            drill_date = st.selectbox("Date", options=dates_any, index=0, key="drill_date")
            drill_lock = st.selectbox("Lock time", options=["09:30 CST", "12:00 CST"], index=0, key="drill_lock")
            drill_strat = "lock_0930" if drill_lock.startswith("09") else "lock_1200"

            # Pull from perf (all rows), and then display settlement if available
            ddf_all = perf[(perf["date"].astype(str) == str(drill_date)) & (perf["strategy"] == drill_strat)].copy()
            # Dedupe raw rows by city so the drilldown is always 1 row per city
            if not ddf_all.empty and all(c in ddf_all.columns for c in ["city", "date", "strategy"]):
                ddf_all = ddf_all.drop_duplicates(["date", "strategy", "city"], keep="last")

            if ddf_all.empty:
                st.info("No records for that date/lock yet.")
            else:
                # Prefer the settled/enriched version when available (done), but keep pending rows
                ddf_done = done[(done["date"].astype(str) == str(drill_date)) & (done["strategy"] == drill_strat)].copy()
                if not ddf_done.empty and all(c in ddf_done.columns for c in ["city", "date", "strategy"]):
                    ddf_done = ddf_done.drop_duplicates(["date", "strategy", "city"], keep="last")

                ddf = ddf_all.copy()
                if not ddf_done.empty:
                    # overlay settlement columns onto the base rows by city
                    for col in ["observed_high_f", "winning_contract", "won", "profit"]:
                        if col in ddf_done.columns:
                            # Build a SAFE city -> value mapping (dedupe cities to avoid pandas InvalidIndexError)
                            tmp = ddf_done[["city", col]].copy()
                            tmp["city"] = tmp["city"].astype(str).str.strip()
                            tmp = tmp.dropna(subset=["city"])
                            tmp = tmp.drop_duplicates(subset=["city"], keep="last")

                            m = dict(zip(tmp["city"], tmp[col]))

                            if col not in ddf.columns:
                                ddf[col] = pd.NA

                            ddf["_city_key"] = ddf["city"].astype(str).str.strip()
                            mapped = ddf["_city_key"].map(m)
                            ddf[col] = mapped.where(mapped.notna(), ddf[col])
                            ddf = ddf.drop(columns=["_city_key"])

                show = ddf[[c for c in ["city", "best_contract", "winning_contract", "observed_high_f", "won"] if c in ddf.columns]].copy()
                show = show.rename(columns={
                    "city": "City",
                    "best_contract": "Pick",
                    "winning_contract": "Winning contract",
                    "observed_high_f": "Observed high (°F)",
                    "won": "Won",
                })

                obs_num = pd.to_numeric(show.get("Observed high (°F)"), errors="coerce")

                if "Winning contract" in show.columns:
                    # If no observed high yet, we are waiting for settlement
                    show.loc[obs_num.isna(), "Winning contract"] = "⏳ Waiting for official highs"
                    # If we have an observed high but can't compute the bucket (missing labels_json), mark as unknown
                    unknown = obs_num.notna() & (
                        show["Winning contract"].isna()
                        | (show["Winning contract"].astype(str).str.strip().str.lower().isin(["nan", "none", ""]))
                    )
                    show.loc[unknown, "Winning contract"] = "⚠️ Unknown (missing labels)"

                # Normalize observed high numeric for display
                if "Observed high (°F)" in show.columns:
                    show["Observed high (°F)"] = pd.to_numeric(show["Observed high (°F)"], errors="coerce")

                # Style: green row if won, red if lost
                def _bg_row_won(row):
                    v = row.get("Won")
                    try:
                        # Unknown/unsettled rows: no color
                        if v is None or pd.isna(v):
                            return [""] * len(row)
                        return [
                            "background-color: rgba(34,197,94,0.14);" if float(v) >= 1.0 else "background-color: rgba(239,68,68,0.14);"
                        ] * len(row)
                    except Exception:
                        return [""] * len(row)

                styled_show = (
                    show.style
                        .format({"Observed high (°F)": "{:.1f}"}, na_rep="—")
                        .apply(_bg_row_won, axis=1)
                )
                st.dataframe(styled_show, width="stretch", hide_index=True)
        else:
            st.info("No history rows yet.")
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

            # Only show the minimal settled-row fields for legibility
            cols = [
                c for c in [
                    "date",
                    "strategy",
                    "best_contract",
                    "winning_contract",
                    "observed_high_f",
                    "won",
                ]
                if c in d.columns
            ]
            out = d[cols].copy()
            out = out.rename(columns={
                "winning_contract": "winning_contract",
                "observed_high_f": "observed_high_f",
            })
            return out

        def _render_city_panel(df_in: pd.DataFrame, label: str):
            if df_in.empty:
                st.info("No settled rows for this view yet.")

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

            # Color the observed high cell green/red depending on win/loss
            if "won" in rows.columns:
                rows2 = rows.copy()

                def _bg_obs_cell(_v, _won):
                    if _won is None or (isinstance(_won, float) and pd.isna(_won)):
                        return ""
                    return "background-color: rgba(34,197,94,0.22);" if float(_won) >= 1.0 else "background-color: rgba(239,68,68,0.22);"

                def _style_obs(s):
                    # s is a Series for the observed_high_f column
                    return [
                        _bg_obs_cell(v, w)
                        for v, w in zip(rows2.get("observed_high_f"), rows2.get("won"))
                    ]

                # Hide won from the table, but keep it for styling
                display_cols = [c for c in rows2.columns if c != "won"]
                st.dataframe(
                    rows2[display_cols].style.format({"observed_high_f": "{:.1f}"}, na_rep="—").apply(_style_obs, subset=["observed_high_f"]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.dataframe(rows, width="stretch", hide_index=True)

        with tabs[0]:
            _render_city_panel(done, "overall")

        with tabs[1]:
            _render_city_panel(done[done["strategy"] == "lock_0930"].copy(), "0930")

        with tabs[2]:
            _render_city_panel(done[done["strategy"] == "lock_1200"].copy(), "1200")