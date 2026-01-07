import os, time, math, base64
from datetime import datetime, timedelta, timezone

import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding


# -----------------------
# CONFIG
# -----------------------
SERIES_TICKER = "KXHIGHPHIL"
# If this base URL 404s, we’ll swap it to the correct Kalshi API host for your account.
BASE_URL = "https://api.elections.kalshi.com"

PATH_MARKETS = "/trade-api/v2/markets"
PATH_MARKET = "/trade-api/v2/markets/{ticker}"
PATH_EVENT = "/trade-api/v2/events/{event_ticker}"

STATION = "KPHL"
LAT = 39.872
LON = -75.241

# Philly in January (EST = UTC-5)
TZ = timezone(timedelta(hours=-5))

# Morning uncertainty knob (tune later)
SIGMA_F = 2.0

# Your exact Kalshi buckets
BUCKETS = [
    ("43° or below", None, 43),
    ("44° to 45°", 44, 45),
    ("46° to 47°", 46, 47),
    ("48° to 49°", 48, 49),
    ("50° to 51°", 50, 51),
    ("52° or above", 52, None),
]

NWS_HEADERS = {
    "User-Agent": "philly-edge/0.1 (contact: you@example.com)",
    "Accept": "application/geo+json",
}


# -----------------------
# NWS MODEL
# -----------------------
def nws_get_json(url: str) -> dict:
    r = requests.get(url, headers=NWS_HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def p_max_lt(x: float, means, sigma: float) -> float:
    prod = 1.0
    for m in means:
        prod *= normal_cdf((x - m) / sigma)
    return prod

def bucket_prob(lo, hi, means, sigma):
    # Convert integer buckets to half-degree edges:
    # 44–45 => [43.5, 45.5)
    if lo is None:
        a, b = -1e9, hi + 0.5
    elif hi is None:
        a, b = lo - 0.5, 1e9
    else:
        a, b = lo - 0.5, hi + 0.5
    return max(0.0, min(1.0, p_max_lt(b, means, sigma) - p_max_lt(a, means, sigma)))


def observed_high_so_far_today() -> float:
    # Pull recent observations and compute today's HIGH so far.
    # This fixes late-day behavior where the "max" already happened.
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    obs = nws_get_json(f"https://api.weather.gov/stations/{STATION}/observations")
    feats = obs.get("features", [])

    highs = []
    for f in feats:
        props = f.get("properties", {})
        ts = props.get("timestamp")
        if not ts:
            continue
        t_local = datetime.fromisoformat(ts).astimezone(TZ)
        if not (day_start <= t_local < day_end):
            continue
        temp_c = (props.get("temperature") or {}).get("value")
        if temp_c is None:
            continue
        temp_f = temp_c * 9/5 + 32
        highs.append(temp_f)

    return max(highs) if highs else float("nan")

def model_probs_today() -> dict:
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    points = nws_get_json(f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}")
    hourly_url = points["properties"]["forecastHourly"]
    hourly = nws_get_json(hourly_url)

    means = []
    for p in hourly["properties"]["periods"]:
        t_local = datetime.fromisoformat(p["startTime"]).astimezone(TZ)
        if day_start <= t_local < day_end:
            means.append(float(p["temperature"]))

    if len(means) < 1:
        raise RuntimeError("No NWS hourly points found for today. Try again in a bit.")
    if len(means) < 6:
        print(f"[WARN] Only {len(means)} forecast hours left in today; probabilities will be less reliable late in the day.\n")
    probs = {}
    total = 0.0
    for name, lo, hi in BUCKETS:
        p = bucket_prob(lo, hi, means, SIGMA_F)
        probs[name] = p
        total += p

    if total > 0:
        for k in probs:
            probs[k] /= total

        # Observed high so far today
    high_so_far = observed_high_so_far_today()
    # Attach it for later use
    probs['_HIGH_SO_FAR_F'] = high_so_far
    return probs


# -----------------------
# KALSHI AUTH (RSA-PSS)
# -----------------------
def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    path_wo_query = path.split("?")[0]
    msg = f"{timestamp_ms}{method.upper()}{path_wo_query}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

def kalshi_get(path: str, params=None) -> dict:
    key_id = os.environ["KALSHI_API_KEY_ID"].strip()
    key_path = os.environ["KALSHI_PRIVATE_KEY_PATH"].strip()
    private_key = load_private_key(key_path)

    ts = str(int(time.time() * 1000))
    sig = sign_request(private_key, ts, "GET", path)

    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }
    r = requests.get(BASE_URL + path, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# -----------------------
# KALSHI MARKET DISCOVERY
# -----------------------
def today_suffix() -> str:
    # '26jan06' style (lowercase), matching the web URL
    d = datetime.now(TZ)
    return d.strftime("%y%b%d").lower()

def get_today_anchor_market() -> dict:
    suffix = today_suffix()  # '26jan06'
    date_tag = "-" + suffix.upper() + "-"  # '-26JAN06-'

    data = kalshi_get(PATH_MARKETS, params={
        "series_ticker": SERIES_TICKER,
        "status": "open",
        "limit": 500
    })

    markets = data.get("markets", [])
    if not markets:
        raise RuntimeError("No open markets returned for series. Series ticker may be wrong.")

    todays = [m for m in markets if date_tag in (m.get("ticker") or "").upper()]
    if not todays:
        sample = [m.get("ticker") for m in markets[:15]]
        raise RuntimeError(f"No markets found for date tag {date_tag}. Sample tickers: {sample}")

    # Prefer a 'T' market as anchor (seems like a top-level contract for the day)
    t_markets = [m for m in todays if "-T" in (m.get("ticker") or "").upper()]
    anchor = t_markets[0] if t_markets else todays[0]
    return anchor

    # If not found, show a helpful hint
    sample = [m.get("ticker") for m in markets[:10]]
    raise RuntimeError(f"Couldn't find today's market ending with '{suffix}'. Sample tickers: {sample}")

def get_event_markets(event_ticker: str) -> list:
    data = kalshi_get(PATH_EVENT.format(event_ticker=event_ticker), params={"with_nested_markets": "true"})
    ev = data.get("event", data)
    # Some responses store markets at ev["markets"], others at data["markets"]
    return ev.get("markets") or data.get("markets") or []


def yes_ask_prob(mkt: dict):
    ya = mkt.get("yes_ask")
    return None if ya is None else ya / 100.0

def yes_bid_prob(mkt: dict):
    yb = mkt.get("yes_bid")
    return None if yb is None else yb / 100.0

def no_ask_prob(mkt: dict):
    na = mkt.get("no_ask")
    return None if na is None else na / 100.0

def no_bid_prob(mkt: dict):
    nb = mkt.get("no_bid")
    return None if nb is None else nb / 100.0

def best_yes_mid(mkt: dict) -> float:
    yb = mkt.get("yes_bid")
    ya = mkt.get("yes_ask")
    if yb is None or ya is None:
        return None
    return (yb + ya) / 2.0 / 100.0  # cents -> dollars prob

def label_for_market(mkt: dict) -> str:
    return (mkt.get("yes_sub_title") or mkt.get("title") or mkt.get("ticker") or "").strip()



def final_probs_with_obs(high_so_far_f: float, means_today, sigma: float) -> dict:
    # Combine observed high so far with remaining-max distribution.
    # If high_so_far is nan, fall back to forecast-only.
    if high_so_far_f != high_so_far_f:  # nan check
        out = {}
        total = 0.0
        for name, lo, hi in BUCKETS:
            p = bucket_prob(lo, hi, means_today, sigma)
            out[name] = p
            total += p
        if total > 0:
            for k in out:
                out[k] /= total
        return out

    H = int(round(high_so_far_f))

    # Helper: CDF of remaining max at x
    def F(x):
        return p_max_lt(x, means_today, sigma)

    out = {}
    total = 0.0
    for name, lo, hi in BUCKETS:
        # Convert bucket to half-step edges [a,b)
        if lo is None:
            a, b = -1e9, hi + 0.5
        elif hi is None:
            a, b = lo - 0.5, 1e9
        else:
            a, b = lo - 0.5, hi + 0.5

        # Final max = max(H, M_rem). So:
        # P(final in [a,b)) =
        #   0 if b <= H-0.5
        #   F(b)           if a <= H < b   (because final < b iff M_rem < b; and final >= a is guaranteed by H)
        #   F(b)-F(a)      if a > H        (same as forecast-only on M_rem)
        H_edge = H + 0.0
        if b <= H_edge - 0.5:
            p_bucket = 0.0
        elif a <= H_edge < b:
            p_bucket = F(b)
        else:
            p_bucket = max(0.0, F(b) - F(a))

        out[name] = max(0.0, p_bucket)
        total += out[name]

    if total > 0:
        for k in out:
            out[k] /= total
    return out

def main():
    print("\nPHILLY EDGE CHECK\n")

    # 1) Model probabilities
    probs = model_probs_today()
    high_so_far = probs.pop('_HIGH_SO_FAR_F', float('nan'))
    # Rebuild today's hourly means (same logic as model_probs_today) so we can combine with observed high
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    points = nws_get_json(f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}")
    hourly_url = points["properties"]["forecastHourly"]
    hourly = nws_get_json(hourly_url)
    means_today = []
    for pp in hourly["properties"]["periods"]:
        t_local = datetime.fromisoformat(pp["startTime"]).astimezone(TZ)
        if day_start <= t_local < day_end:
            means_today.append(float(pp["temperature"]))
    probs = final_probs_with_obs(high_so_far, means_today, SIGMA_F)
    if high_so_far == high_so_far:
        print(f"Observed HIGH so far today (KPHL): {high_so_far:.1f}F\n")


    # 2) Kalshi: find today's event via anchor market
    anchor_market = get_today_anchor_market()
    anchor_ticker = anchor_market["ticker"]
    event_ticker = anchor_market["event_ticker"]

    print(f"Kalshi anchor market: {anchor_ticker}")
    print(f"Kalshi event ticker:  {event_ticker}\n")

    markets = get_event_markets(event_ticker)

    # Build label -> mid price
    market_mid = {}
    for m in markets:
        label = label_for_market(m)
        if label:
            market_mid[label] = best_yes_mid(m)

    print(f"Sigma used: {SIGMA_F}°F\n")
    print(f"{'Bucket':<14}  {'Model%':>7}  {'MktMid%':>8}  {'Edge%':>7}  {'EV(YES)%':>9}")
    print("-" * 55)

    rows = []
    for name, _, _ in BUCKETS:
        model_pct = probs.get(name, 0.0) * 100.0
        mid = market_mid.get(name)

        if mid is None:
            mkt_pct = None
            edge = None
        else:
            mkt_pct = mid * 100.0
            edge = model_pct - mkt_pct

        ev_yes = None if mkt_pct is None else (model_pct - mkt_pct)
        rows.append((name, model_pct, mkt_pct, edge, ev_yes))

    # Print in your bucket order (clear), but also compute top pick by edge
    for name, model_pct, mkt_pct, edge, ev_yes in rows:
        if mkt_pct is None:
            print(f"{name:<14}  {model_pct:6.1f}%  {'(no data)':>8}  {'':>7}  {'':>9}")
        else:
            print(f"{name:<14}  {model_pct:6.1f}%  {mkt_pct:7.1f}%  {edge:6.1f}%  {ev_yes:8.1f}%")

    best_bucket, best_p = max(probs.items(), key=lambda kv: kv[1])
    print(f"\nMost likely (model): {best_bucket} ({best_p*100:.1f}%)")

    # Top pick = max positive edge (must have market data)
    candidates = [r for r in rows if (r[2] is not None and r[3] is not None)]
    if candidates:
        top = max(candidates, key=lambda r: r[4])  # biggest edge
        name, model_pct, mkt_pct, edge, ev_yes = top
        if edge > 0:
            print(f"Top action (by EV): {name}  | Model {model_pct:.1f}% vs Mkt {mkt_pct:.1f}%  => Edge +{edge:.1f}%")
        else:
            print(f"Top action (by EV): None positive (best is {name} at {edge:.1f}%)")


if __name__ == "__main__":
    main()

# -----------------------
# Dynamic bucket support (Kalshi-driven)
# -----------------------
import re
from typing import Optional, Tuple, List

def parse_bucket_label(label: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """
    Convert label like:
      '43° or below' -> (None, 43)
      '44° to 45°'   -> (44, 45)
      '52° or above' -> (52, None)
    Returns (lo, hi) in integer °F bounds, or None if it can't parse.
    """
    s = label.strip().replace("º", "°")

    m = re.search(r"(\d+)\s*°?\s*or\s*below", s, re.IGNORECASE)
    if m:
        return (None, int(m.group(1)))

    m = re.search(r"(\d+)\s*°?\s*or\s*above", s, re.IGNORECASE)
    if m:
        return (int(m.group(1)), None)

    m = re.search(r"(\d+)\s*°?\s*(?:to|-)\s*(\d+)\s*°?", s, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    return None

def today_date_tag() -> str:
    # '-26JAN06-'
    return "-" + today_suffix().upper() + "-"

def get_today_bucket_markets() -> List[dict]:
    """
    Returns list of dicts:
      { 'label': str, 'lo': int|None, 'hi': int|None, 'market': market_dict }
    pulled from Kalshi /markets list for today's date tag.
    """
    anchor = get_today_anchor_market()
    date_tag = today_date_tag()

    data = kalshi_get(PATH_MARKETS, params={
        "series_ticker": SERIES_TICKER,
        "status": "open",
        "limit": 500
    })
    markets = data.get("markets", [])

    todays = [m for m in markets if date_tag in (m.get("ticker") or "").upper()]

    out = []
    for m in todays:
        label = label_for_market(m)
        if not label:
            continue
        bounds = parse_bucket_label(label)
        if not bounds:
            continue
        lo, hi = bounds
        out.append({"label": label, "lo": lo, "hi": hi, "market": m})

    # sort by lower bound (None first)
    def key(x):
        lo = x["lo"]
        return -10_000 if lo is None else lo
    out.sort(key=key)
    return out

def model_probs_for_buckets(bucket_bounds: List[Tuple[str, Optional[int], Optional[int]]], sigma_f: float) -> dict:
    """
    Compute probabilities for arbitrary buckets (label, lo, hi).
    Uses observed high so far + remaining max distribution.
    """
    # Remaining hourly means for today
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    points = nws_get_json(f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}")
    hourly_url = points["properties"]["forecastHourly"]
    hourly = nws_get_json(hourly_url)

    means_today = []
    for pp in hourly["properties"]["periods"]:
        t_local = datetime.fromisoformat(pp["startTime"]).astimezone(TZ)
        if day_start <= t_local < day_end:
            means_today.append(float(pp["temperature"]))

    high_so_far = observed_high_so_far_today()

    # Build a local bucket list in the same structure as BUCKETS and re-use logic
    # We'll compute per-bucket using the same max-CDF approach used in final_probs_with_obs.
    if len(means_today) < 1:
        raise RuntimeError("No NWS hourly points found for today. Try again in a bit.")

    # CDF of remaining max at x
    def F(x):
        return p_max_lt(x, means_today, sigma_f)

    # If no obs, treat H as NaN
    H_is_nan = (high_so_far != high_so_far)
    H = int(round(high_so_far)) if not H_is_nan else None

    out = {}
    total = 0.0
    for label, lo, hi in bucket_bounds:
        # half-step edges [a,b)
        if lo is None:
            a, b = -1e9, hi + 0.5
        elif hi is None:
            a, b = lo - 0.5, 1e9
        else:
            a, b = lo - 0.5, hi + 0.5

        if H_is_nan:
            p_bucket = max(0.0, F(b) - F(a))
        else:
            # final max = max(H, M_rem)
            H_edge = float(H)
            if b <= H_edge - 0.5:
                p_bucket = 0.0
            elif a <= H_edge < b:
                p_bucket = F(b)
            else:
                p_bucket = max(0.0, F(b) - F(a))

        out[label] = float(max(0.0, p_bucket))
        total += out[label]

    if total > 0:
        for k in out:
            out[k] /= total
    return out

# -----------------------
# Self-calibrating sigma (local logging)
# -----------------------
import csv
import json
from pathlib import Path

LOG_DIR = Path.home() / ".kalshi_weather_edge"
LOG_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_LOG = LOG_DIR / "forecast_log.jsonl"  # one line per run/day
SIGMA_LOG = LOG_DIR / "sigma_history.csv"

def _as_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _nws_hourly_means_for_local_day(target_day_local: datetime) -> list:
    """
    Return NWS hourly forecast temps (°F) for the given local day.
    This is the *current* forecast for that day (not historical).
    """
    day_start = target_day_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    points = nws_get_json(f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}")
    hourly_url = points["properties"]["forecastHourly"]
    hourly = nws_get_json(hourly_url)

    means = []
    for pp in hourly["properties"]["periods"]:
        t_local = datetime.fromisoformat(pp["startTime"]).astimezone(TZ)
        if day_start <= t_local < day_end:
            means.append(float(pp["temperature"]))
    return means

def log_forecast_snapshot():
    """
    Logs today's forecast snapshot (hourly means) + predicted expected max.
    Run this once in the morning (or each run—duplicates are deduped by date).
    """
    now_local = datetime.now(TZ)
    day = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    date_s = _as_date_str(day)

    # Avoid duplicate entries for same date
    if FORECAST_LOG.exists():
        with FORECAST_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("date") == date_s:
                        return
                except Exception:
                    continue

    means = _nws_hourly_means_for_local_day(day)
    if not means:
        return

    # expected value of max isn't trivial; use a practical proxy:
    # expected max approx = max(mean) (works well for daily highs)
    expected_max = float(max(means))

    rec = {
        "date": date_s,
        "station": STATION,
        "series": SERIES_TICKER,
        "expected_max": expected_max,
        "hourly_means": means,
    }
    with FORECAST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

def _fetch_observed_daily_high(date_s: str) -> Optional[float]:
    """
    Fetch observed high (°F) for STATION on date_s using IEM ASOS JSON.
    We compute max(tmpf) for that UTC date window; good enough for daily high.
    """
    # IEM ASOS request. We already use requests in this file.
    y, m, d = date_s.split("-")
    params = {
        "station": STATION,
        "data": "tmpf",
        "year1": int(y), "month1": int(m), "day1": int(d),
        "year2": int(y), "month2": int(m), "day2": int(d),
        "tz": "America/New_York",
        "format": "json",
    }
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    temps = []
    for row in j.get("data", []):
        t = row.get("tmpf")
        try:
            if t is not None:
                temps.append(float(t))
        except Exception:
            pass
    if not temps:
        return None
    return float(max(temps))

def _robust_std(vals: list[float]) -> Optional[float]:
    """Robust sigma estimate using MAD -> std (scaled)."""
    vals = [v for v in vals if v == v]  # remove NaN
    if len(vals) < 5:
        return None
    med = sorted(vals)[len(vals)//2]
    abs_dev = sorted([abs(v - med) for v in vals])
    mad = abs_dev[len(abs_dev)//2]
    # For normal dist, std ≈ 1.4826 * MAD
    return 1.4826 * mad

def calibrate_sigma(days_back: int = 14, default: float = 2.0, min_sigma: float = 1.0, max_sigma: float = 5.0) -> float:
    """
    Calibrate sigma from your own logged forecast snapshots vs observed daily highs.
    - Uses rolling last `days_back` completed days
    - Returns sigma in °F
    """
    # Make sure we have at least today's snapshot logged
    try:
        log_forecast_snapshot()
    except Exception:
        pass

    if not FORECAST_LOG.exists():
        return default

    # Read log and compute errors for days with observation available
    recs = []
    with FORECAST_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue

    # Only keep last N days
    recs = sorted(recs, key=lambda r: r.get("date", ""))[-(days_back+3):]

    errors = []
    for rec in recs:
        date_s = rec.get("date")
        if not date_s:
            continue
        # Don't score today (still in progress)
        if date_s == _as_date_str(datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)):
            continue

        try:
            obs_high = _fetch_observed_daily_high(date_s)
        except Exception:
            continue
        if obs_high is None:
            continue

        expected_max = rec.get("expected_max")
        if expected_max is None:
            continue

        errors.append(float(obs_high) - float(expected_max))

    sig = _robust_std(errors)
    if sig is None:
        sig = default

    sig = float(max(min_sigma, min(max_sigma, sig)))

    # Write sigma history (optional but nice)
    try:
        write_header = not SIGMA_LOG.exists()
        with SIGMA_LOG.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp_local", "station", "days_used", "sigma_f"])
            w.writerow([datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), STATION, len(errors), f"{sig:.3f}"])
    except Exception:
        pass

    return sig

# -----------------------
# Historical performance tracking (per-city, per-day)
# -----------------------
from typing import Optional
import json
from pathlib import Path
import pandas as pd

PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)
PERF_CSV = PROJECT_DATA_DIR / "performance.csv"
def _today_local_date_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")

def _parse_bucket_label(label: str):
    """
    Returns (lo, hi) integer bounds in °F, where hi can be None for +inf and lo can be None for -inf.
    Supports labels like:
      "43° or below"
      "52° or above"
      "44° to 45°"
    """
    s = label.replace("°", "").strip()

    if "or below" in s:
        n = int("".join([c for c in s.split("or below")[0] if c.isdigit() or c == "-"]).strip())
        return (None, n)

    if "or above" in s:
        n = int("".join([c for c in s.split("or above")[0] if c.isdigit() or c == "-"]).strip())
        return (n, None)

    if "to" in s:
        a, b = s.split("to")
        lo = int("".join([c for c in a if c.isdigit() or c == "-"]).strip())
        hi = int("".join([c for c in b if c.isdigit() or c == "-"]).strip())
        return (lo, hi)

    return (None, None)

def _label_matches_temp(label: str, temp_f: float) -> bool:
    lo, hi = _parse_bucket_label(label)
    t = float(temp_f)
    if lo is None and hi is None:
        return False
    if lo is None:
        return t <= hi
    if hi is None:
        return t >= lo
    return (lo <= t <= hi)

def _resolve_winning_label(labels: list, observed_high_f: float) -> Optional[str]:
    for lab in labels:
        try:
            if _label_matches_temp(lab, observed_high_f):
                return lab
        except Exception:
            continue
    return None

def _ensure_perf_header():
    if PERF_CSV.exists():
        return
    PERF_CSV.write_text(
        "date,city,station,sigma_f,labels_json,best_contract,yes_ask_prob,model_prob,value_prob,"
        "observed_high_f,winning_contract,won,profit\n",
        encoding="utf-8"
    )

def perf_log_snapshot(date_s: str, city: str, station: str, sigma_f: float,
                      labels: list, best_contract: str,
                      yes_ask_prob: Optional[float], model_prob: Optional[float], value_prob: Optional[float]):
    """
    Writes ONE row per city per day (deduped by date+city).
    yes_ask_prob/model_prob/value_prob are in 0..1 (not percent).
    """
    _ensure_perf_header()

    # Deduplicate
    if PERF_CSV.exists():
        existing = PERF_CSV.read_text(encoding="utf-8").splitlines()
        for line in existing[1:]:
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2 and parts[0] == date_s and parts[1] == city:
                return

    labels_json = json.dumps(labels, separators=(",", ":"))

    def fnum(x):
        return "" if x is None else f"{float(x):.6f}"

    row = (
        f"{date_s},{city},{station},{float(sigma_f):.6f},{json.dumps(labels_json)},"
        f"{best_contract},{fnum(yes_ask_prob)},{fnum(model_prob)},{fnum(value_prob)},"
        f",,,,\n"
    )
    with PERF_CSV.open("a", encoding="utf-8") as f:
        f.write(row)

def perf_update_outcomes():
    """
    For rows with blank outcomes and date < today, fetch observed daily high and compute results.
    """
    if not PERF_CSV.exists():
        return

    lines = PERF_CSV.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return

    header = lines[0]
    out = [header]
    today = _today_local_date_str()

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        # Expected columns:
        # 0 date,1 city,2 station,3 sigma_f,4 labels_json,5 best_contract,6 yes_ask_prob,7 model_prob,8 value_prob,
        # 9 observed_high_f,10 winning_contract,11 won,12 profit
        # If we already have observed_high_f filled, keep line
        if len(parts) < 13:
            out.append(line)
            continue

        date_s = parts[0]
        station = parts[2]

        observed = parts[9].strip()
        if observed or date_s >= today:
            out.append(line)
            continue

        # Load labels list back
        try:
            labels_json_escaped = parts[4]
            labels_json = json.loads(labels_json_escaped)   # this returns a JSON string
            labels = json.loads(labels_json)               # this returns the list
        except Exception:
            labels = []

        # Fetch observed high for that station/date
        try:
            # temporarily swap station for fetch
            old_station = globals().get("STATION", station)
            globals()["STATION"] = station
            obs_high = _fetch_observed_daily_high(date_s)
            globals()["STATION"] = old_station
        except Exception:
            obs_high = None

        if obs_high is None:
            out.append(line)
            continue

        winning = _resolve_winning_label(labels, obs_high) or ""

        best_contract = parts[5]
        yes_ask_prob = parts[6].strip()
        try:
            price = float(yes_ask_prob) if yes_ask_prob else None
        except Exception:
            price = None

        won = "1" if (winning and best_contract == winning) else "0"

        profit = ""
        if price is not None:
            # Simple YES at ask: + (1 - price) if win else -price
            profit_val = (1.0 - price) if won == "1" else (-price)
            profit = f"{profit_val:.6f}"

        parts[9] = f"{float(obs_high):.1f}"
        parts[10] = winning
        parts[11] = won
        parts[12] = profit

        out.append(",".join(parts))

    PERF_CSV.write_text("\n".join(out) + "\n", encoding="utf-8")

def perf_load_df():
    if not PERF_CSV.exists():
        _ensure_perf_header()
        return pd.read_csv(PERF_CSV)
    return pd.read_csv(PERF_CSV)

# -----------------------
# Hourly forecast (NWS) for plotting
# -----------------------
def nws_hourly_forecast_for_today():
    """
    Returns list of dicts:
      [{"time_local": datetime, "temp_f": float, "short": str}, ...]
    for the local calendar day in TZ.
    """
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    points = nws_get_json(f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}")
    hourly_url = points["properties"]["forecastHourly"]
    hourly = nws_get_json(hourly_url)

    out = []
    for pp in hourly["properties"]["periods"]:
        t_local = datetime.fromisoformat(pp["startTime"]).astimezone(TZ)
        if day_start <= t_local < day_end:
            out.append({
                "time_local": t_local,
                "temp_f": float(pp["temperature"]),
                "short": pp.get("shortForecast", ""),
            })
    return out

# -----------------------
# Observations: latest temp + high so far today (station)
# -----------------------
def obs_latest_and_high_today():
    """
    Returns dict:
      {"latest_time_local": "YYYY-mm-dd HH:MM", "latest_temp_f": float,
       "high_so_far_f": float}
    Uses IEM ASOS JSON for the configured STATION in TZ.
    """
    now_local = datetime.now(TZ)
    date_s = now_local.strftime("%Y-%m-%d")
    y, m, d = date_s.split("-")

    params = {
        "station": STATION,
        "data": "tmpf",
        "year1": int(y), "month1": int(m), "day1": int(d),
        "year2": int(y), "month2": int(m), "day2": int(d),
        "tz": "America/New_York",
        "format": "json",
    }
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    latest = None
    latest_ts = None
    high = None

    for row in j.get("data", []):
        t = row.get("tmpf")
        ts = row.get("local_valid") or row.get("valid")  # local_valid is typical with tz param
        if t is None or ts is None:
            continue
        try:
            tf = float(t)
        except Exception:
            continue

        # high so far
        high = tf if high is None else max(high, tf)

        # latest (string compare works for ISO-ish timestamps; keep safest by just taking last seen)
        latest = tf
        latest_ts = ts

    if latest is None or high is None:
        return None

    # ts often like "2026-01-07 13:53" already local
    return {
        "latest_time_local": str(latest_ts),
        "latest_temp_f": float(latest),
        "high_so_far_f": float(high),
    }

# -----------------------
# NWS station observations (api.weather.gov) — official source
# -----------------------
def _c_to_f(c: float) -> float:
    return (float(c) * 9.0 / 5.0) + 32.0

def nws_obs_latest_and_high_today_station():
    """
    Uses: https://api.weather.gov/stations/{STATION}/observations/latest
    and https://api.weather.gov/stations/{STATION}/observations?start=... to compute high so far today.
    Returns dict:
      {"latest_time_local": "...", "latest_temp_f": float, "high_so_far_f": float}
    """
    # Latest
    latest = nws_get_json(f"https://api.weather.gov/stations/{STATION}/observations/latest")
    props = latest["properties"]
    t_c = props["temperature"]["value"]
    if t_c is None:
        return None
    latest_time = datetime.fromisoformat(props["timestamp"].replace("Z", "+00:00")).astimezone(TZ)
    latest_f = _c_to_f(t_c)

    # High so far today (local day)
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_iso = day_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    obs = nws_get_json(f"https://api.weather.gov/stations/{STATION}/observations?start={start_iso}")
    high_f = None
    for feat in obs.get("features", []):
        p = feat.get("properties", {})
        tc = (p.get("temperature") or {}).get("value")
        if tc is None:
            continue
        tf = _c_to_f(tc)
        high_f = tf if high_f is None else max(high_f, tf)

    if high_f is None:
        high_f = latest_f

    return {
        "latest_time_local": latest_time.strftime("%Y-%m-%d %I:%M %p"),
        "latest_temp_f": float(latest_f),
        "high_so_far_f": float(high_f),
    }

def nws_obs_past_hours_station(hours: int = 24):
    """
    Returns list of dicts: [{"time_local": datetime, "temp_f": float}, ...] for past N hours.
    Uses: https://api.weather.gov/stations/{STATION}/observations?start=...
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=int(hours))
    start_iso = start_utc.isoformat().replace("+00:00", "Z")

    obs = nws_get_json(f"https://api.weather.gov/stations/{STATION}/observations?start={start_iso}")
    out = []
    for feat in obs.get("features", []):
        p = feat.get("properties", {})
        ts = p.get("timestamp")
        tc = (p.get("temperature") or {}).get("value")
        if ts is None or tc is None:
            continue
        t_local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(TZ)
        out.append({"time_local": t_local, "temp_f": _c_to_f(tc)})

    out.sort(key=lambda x: x["time_local"])
    return out

def nws_hourly_forecast_next_hours(hours: int = 24):
    """
    Returns list of dicts for next N hours:
      [{"time_local": datetime, "temp_f": float}, ...]
    Uses NWS hourly forecast for LAT/LON.
    """
    now_local = datetime.now(TZ)
    end_local = now_local + timedelta(hours=int(hours))

    points = nws_get_json(f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}")
    hourly_url = points["properties"]["forecastHourly"]
    hourly = nws_get_json(hourly_url)

    out = []
    for pp in hourly["properties"]["periods"]:
        t_local = datetime.fromisoformat(pp["startTime"]).astimezone(TZ)
        if now_local <= t_local <= end_local:
            out.append({"time_local": t_local, "temp_f": float(pp["temperature"])})
    return out
