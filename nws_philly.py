import requests
from datetime import datetime, timedelta, timezone

STATION = "KPHL"
LAT = 39.872
LON = -75.241

# January in Philly is EST (UTC-5)
TZ = timezone(timedelta(hours=-5))

HEADERS = {
    "User-Agent": "philly-temp-script/0.4 (contact: you@example.com)",
    "Accept": "application/geo+json",
}

# Kalshi outcomes you pasted:
# 43° or below
# 44° to 45°
# 46° to 47°
# 48° to 49°
# 50° to 51°
# 52° or above

def kalshi_bucket(temp_f: float) -> str:
    # Use "nearest whole degree" since Kalshi outcomes are whole-degree buckets.
    t = int(round(temp_f))

    if t <= 43:
        return "43° or below"
    if 44 <= t <= 45:
        return "44° to 45°"
    if 46 <= t <= 47:
        return "46° to 47°"
    if 48 <= t <= 49:
        return "48° to 49°"
    if 50 <= t <= 51:
        return "50° to 51°"
    return "52° or above"

def get_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)

def f_from_c(c: float) -> float:
    return c * 9/5 + 32

def main():
    now_local = datetime.now(TZ)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    print(f"Now (Philly local): {now_local.strftime('%Y-%m-%d %H:%M')}")
    print(f"Today's window:    {day_start.strftime('%Y-%m-%d %H:%M')} → {day_end.strftime('%Y-%m-%d %H:%M')}\n")

    # -----------------------
    # Forecast hourly (today)
    # -----------------------
    points_url = f"https://api.weather.gov/points/{LAT:.4f},{LON:.4f}"
    points = get_json(points_url)
    hourly_url = points["properties"]["forecastHourly"]

    hourly = get_json(hourly_url)
    periods = hourly["properties"]["periods"]

    print("Forecast (next 12 hours):")
    for p in periods[:12]:
        print(f'{p["startTime"]}  {p["temperature"]}{p["temperatureUnit"]}  {p["shortForecast"]}')

    # Forecast high remaining TODAY (only hours between now and midnight)
    forecast_remaining = []
    for p in periods:
        t_local = parse_iso(p["startTime"]).astimezone(TZ)
        if now_local <= t_local < day_end:
            forecast_remaining.append(p["temperature"])

    forecast_high_remaining = max(forecast_remaining) if forecast_remaining else None

    # -----------------------
    # Observations (today)
    # -----------------------
    obs_url = f"https://api.weather.gov/stations/{STATION}/observations"
    obs = get_json(obs_url)
    features = obs.get("features", [])

    temps_today = []
    latest_obs_time = None
    latest_obs_temp_f = None

    for feat in features:
        props = feat.get("properties", {})
        ts = props.get("timestamp")
        if not ts:
            continue

        t_local = parse_iso(ts).astimezone(TZ)
        if not (day_start <= t_local < day_end):
            continue

        temp_c = (props.get("temperature") or {}).get("value")
        if temp_c is None:
            continue

        temp_f = f_from_c(temp_c)
        temps_today.append(temp_f)

        if latest_obs_time is None or t_local > latest_obs_time:
            latest_obs_time = t_local
            latest_obs_temp_f = temp_f

    high_so_far = max(temps_today) if temps_today else None

    print(f"\nObserved (station {STATION})")
    if latest_obs_time is not None:
        print(f"  latest time: {latest_obs_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  latest temp: {latest_obs_temp_f:.1f}F")
    else:
        print("  latest: N/A")

    if high_so_far is not None:
        print(f"  HIGH so far today: {high_so_far:.1f}F")
    else:
        print("  HIGH so far today: N/A")

    # -----------------------
    # Best guess final high
    # -----------------------
    candidates = [x for x in [high_so_far, forecast_high_remaining] if x is not None]
    best_guess = max(candidates) if candidates else None

    print("\nModel summary")
    print(f"  forecast HIGH remaining today: {forecast_high_remaining if forecast_high_remaining is not None else 'N/A'}F")

    if best_guess is not None:
        print(f"  best guess FINAL high today:   {best_guess:.1f}F")
        print(f"  Kalshi bucket:                 {kalshi_bucket(best_guess)}")
    else:
        print("  best guess FINAL high today:   N/A")
        print("  Kalshi bucket:                 N/A")

if __name__ == "__main__":
    main()
