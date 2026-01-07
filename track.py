import philly_edge as pe

# City config must live here so this script can run without the Streamlit app.
# NOTE: station_obs should match Kalshi settlement station. NYC = Central Park.
CITIES = {
    "Philadelphia": {"series": "KXHIGHPHIL", "station_obs": "KPHL", "lat": 39.872,  "lon": -75.241},
    "Los Angeles":  {"series": "KXHIGHLAX",  "station_obs": "KLAX", "lat": 33.9425, "lon": -118.4081},
    "Denver":       {"series": "KXHIGHDEN",  "station_obs": "KDEN", "lat": 39.8561, "lon": -104.6737},
    "Miami":        {"series": "KXHIGHMIA",  "station_obs": "KMIA", "lat": 25.7959, "lon": -80.2870},
    "NYC":          {"series": "KXHIGHNY",   "station_obs": "KNYC", "lat": 40.7790, "lon": -73.96925},  # Central Park
    "Chicago":      {"series": "KXHIGHCHI",  "station_obs": "KMDW", "lat": 41.7868, "lon": -87.7522},
    "Austin":       {"series": "KXHIGHAUS",  "station_obs": "KAUS", "lat": 30.1945, "lon": -97.6699},
}

def apply_city(cfg):
    pe.SERIES_TICKER = cfg["series"]
    pe.STATION = cfg["station_obs"]
    pe.LAT = cfg["lat"]
    pe.LON = cfg["lon"]

def log_today_for_all_cities():
    today_s = pe._today_local_date_str()

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
            )

        except Exception:
            # If one city fails, keep going
            continue

if __name__ == "__main__":
    # 1) Ensure today's picks are logged
    log_today_for_all_cities()

    # 2) Fill in outcomes for past days that have settled
    pe.perf_update_outcomes()

    print("✅ Tracker complete: snapshots logged + outcomes updated")
