from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import philly_edge as pe

# NOTE:
# - This script is meant to run via GitHub Actions / cron.
# - It logs ONE snapshot per strategy per day into data/performance.csv
# - It also back-fills outcomes for previous days.

# City config must live here so this script can run without the Streamlit app.
# station_obs should match the settlement/official station used for verification.
CITIES = {
    "Philadelphia": {"series": "KXHIGHPHIL", "station_obs": "KPHL", "lat": 39.872, "lon": -75.241},
    "Los Angeles": {"series": "KXHIGHLAX", "station_obs": "KLAX", "lat": 33.9425, "lon": -118.4081},
    "Denver": {"series": "KXHIGHDEN", "station_obs": "KDEN", "lat": 39.8561, "lon": -104.6737},
    "Miami": {"series": "KXHIGHMIA", "station_obs": "KMIA", "lat": 25.7959, "lon": -80.2870},
    "NYC": {"series": "KXHIGHNY", "station_obs": "KNYC", "lat": 40.7790, "lon": -73.96925},  # Central Park
    "Chicago": {"series": "KXHIGHCHI", "station_obs": "KMDW", "lat": 41.7868, "lon": -87.7522},
    "Austin": {"series": "KXHIGHAUS", "station_obs": "KAUS", "lat": 30.1945, "lon": -97.6699},
}

CST = ZoneInfo("America/Chicago")
LOCK_DIR = Path("data/locks")
LOCK_DIR.mkdir(parents=True, exist_ok=True)


def apply_city(cfg: dict) -> None:
    pe.SERIES_TICKER = cfg["series"]
    pe.STATION = cfg["station_obs"]
    pe.LAT = cfg["lat"]
    pe.LON = cfg["lon"]


def lock_file(strategy: str, date_s: str) -> Path:
    # Keep the same naming convention you already have in data/locks
    # Example: data/locks/LOCKED_0930_2026-01-07_CST.txt
    suffix = "0930" if strategy == "lock_0930" else "1200" if strategy == "lock_1200" else strategy
    return LOCK_DIR / f"LOCKED_{suffix}_{date_s}_CST.txt"


def should_run_lock(strategy: str, date_s: str) -> bool:
    return not lock_file(strategy, date_s).exists()


def mark_lock_ran(strategy: str, date_s: str) -> None:
    lock_file(strategy, date_s).write_text("ok\n", encoding="utf-8")


def log_today_for_all_cities(strategy: str) -> int:
    """Log one snapshot row per city for the given strategy.

    Returns number of rows successfully logged.
    """
    today_s = pe._today_local_date_str()
    wrote = 0

    for city, cfg in CITIES.items():
        try:
            apply_city(cfg)

            sigma = pe.calibrate_sigma(days_back=14)

            bucket_markets = pe.get_today_bucket_markets()
            if not bucket_markets:
                continue

            labels = [bm["label"] for bm in bucket_markets]
            bucket_bounds = [(bm["label"], bm["lo"], bm["hi"]) for bm in bucket_markets]
            probs = pe.model_probs_for_buckets(bucket_bounds, sigma)

            # Find best value bucket (Model - YES ask)
            best = None
            best_value = None

            for bm in bucket_markets:
                label = bm["label"]
                m = bm["market"]

                p_model = float(probs.get(label, 0.0))
                yes_ask = pe.yes_ask_prob(m)  # 0..1 or None
                if yes_ask is None:
                    continue
                value = p_model - yes_ask

                if (best_value is None) or (value > best_value):
                    best_value = value
                    best = (label, yes_ask, p_model, value)

            if best is None:
                continue

            label, yes_ask, p_model, value = best

            # IMPORTANT: pass strategy through so lock_1200 rows are distinct
            pe.perf_log_snapshot(
                date_s=today_s,
                city=city,
                station=cfg["station_obs"],
                sigma_f=sigma,
                labels=labels,
                best_contract=label,
                yes_ask_prob=yes_ask,
                model_prob=p_model,
                value_prob=value,
                strategy=strategy,
            )
            wrote += 1

        except Exception as e:
            print(f"[TRACK][{strategy}][{city}] ERROR: {e}")
            continue

    return wrote


def maybe_run_lock(strategy: str, hour: int, minute: int) -> None:
    today_s = pe._today_local_date_str()
    now_cst = datetime.now(CST)

    # Only run after the target time
    if (now_cst.hour, now_cst.minute) < (hour, minute):
        return

    # Only run once per day
    if not should_run_lock(strategy, today_s):
        return

    print(f"[TRACK] running {strategy} for {today_s}")

    # Write rows FIRST; only create lock file if we successfully wrote at least one row
    wrote = log_today_for_all_cities(strategy)
    if wrote > 0:
        mark_lock_ran(strategy, today_s)
    else:
        print(f"[TRACK] {strategy}: wrote 0 rows, not creating lock file")


if __name__ == "__main__":
    # Snapshot at 09:30 CST
    maybe_run_lock("lock_0930", 9, 30)

    # Snapshot at 12:00 CST
    maybe_run_lock("lock_1200", 12, 0)

    # Fill in outcomes for past days that have settled
    pe.perf_update_outcomes()

    print("✅ Tracker complete: snapshots logged + outcomes updated")
