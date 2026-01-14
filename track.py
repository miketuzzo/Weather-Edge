#!/usr/bin/env python3
"""
Weather Edge Tracker (GitHub Actions)

Goals:
- Run lock_0930 and lock_1200 picks (idempotent via lock files)
- Backfill yesterday (CST) in case a scheduled run failed/missed
- Run settlement sweep (update observed highs + wins) if available
- Never crash the workflow (log errors; exit 0)
- Support Kalshi auth via either:
    - KALSHI_PRIVATE_KEY_PEM (multiline)
    - KALSHI_PRIVATE_KEY_B64 (single-line base64)
    - KALSHI_PRIVATE_KEY_PATH (file path)
"""

import os
import sys
import csv
import base64
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# -------------------------
# Paths / constants
# -------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PERF_PATH = DATA_DIR / "performance.csv"
LOG_PATH = DATA_DIR / "cron_track.log"
LOCK_DIR = DATA_DIR / "locks"
LOCK_DIR.mkdir(exist_ok=True)

PERF_COLUMNS = [
    "date",
    "city",
    "station",
    "sigma_f",
    "labels_json",
    "best_contract",
    "yes_ask_prob",
    "model_prob",
    "value_prob",
    "observed_high_f",
    "winning_contract",
    "won",
    "profit",
    "computed_winning",
    "computed_won",
    "strategy",
]

# -------------------------
# Logging helpers
# -------------------------
def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def day_str_cst(offset_days: int = 0) -> str:
    """
    Use America/Chicago as the app's "business day".
    Falls back to UTC if ZoneInfo isn't available (should be available on GH Actions).
    """
    if ZoneInfo is None:
        d = datetime.now(timezone.utc) + timedelta(days=offset_days)
        return d.strftime("%Y-%m-%d")
    tz = ZoneInfo("America/Chicago")
    d = datetime.now(tz) + timedelta(days=offset_days)
    return d.strftime("%Y-%m-%d")

def lock_file(strategy: str, day: str) -> Path:
    return LOCK_DIR / f"{day}_{strategy}.lock"

def is_locked(strategy: str, day: str) -> bool:
    return lock_file(strategy, day).exists()

def set_locked(strategy: str, day: str, detail: str = "") -> None:
    try:
        lock_file(strategy, day).write_text(detail or "ok", encoding="utf-8")
    except Exception:
        pass

# -------------------------
# Secrets handling
# -------------------------
def ensure_kalshi_key_material() -> None:
    """
    Ensures Kalshi key is available for philly_edge.py.
    Supports:
      - KALSHI_PRIVATE_KEY_PEM
      - KALSHI_PRIVATE_KEY_B64  (preferred in Actions)
      - KALSHI_PRIVATE_KEY_PATH
    """
    key_id = (os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID") or "").strip()
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    b64 = (os.getenv("KALSHI_PRIVATE_KEY_B64") or "").strip()
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or os.getenv("KALSHI_API_PRIVATE_KEY_PATH") or "").strip()

    log(f"[ENV] KEY_ID_SET={bool(key_id)} PEM_LEN={len(pem.strip())} B64_LEN={len(b64)} PATH_SET={bool(key_path)}")

    if key_path:
        return
    if pem.strip():
        return

    if b64:
        tmp_key = Path("/tmp/kalshi.key")
        try:
            raw = base64.b64decode(b64.encode("utf-8"))
            tmp_key.write_bytes(raw)
            os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(tmp_key)
            log(f"[ENV] Decoded KALSHI_PRIVATE_KEY_B64 -> {tmp_key} ({tmp_key.stat().st_size} bytes)")
        except Exception as e:
            log(f"[ENV][ERROR] Failed to decode KALSHI_PRIVATE_KEY_B64: {e}")
        return

    log("[ENV][WARN] No Kalshi private key material found (PEM/B64/PATH). Market calls may fail.")

# -------------------------
# CSV helpers
# -------------------------
def write_rows_append(new_rows: list) -> int:
    """
    Append rows to performance.csv, creating it with PERF_COLUMNS if missing.
    Filters out bad keys (None) and unknown keys (keeps only PERF_COLUMNS).
    """
    if not new_rows:
        return 0

    cleaned = []
    for r in new_rows:
        if not isinstance(r, dict):
            continue
        r = {k: v for k, v in r.items() if k is not None}
        out = {c: r.get(c, "") for c in PERF_COLUMNS}
        cleaned.append(out)

    if not cleaned:
        return 0

    file_exists = PERF_PATH.exists()
    try:
        with PERF_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PERF_COLUMNS)
            if not file_exists:
                writer.writeheader()
            for row in cleaned:
                writer.writerow(row)
        return len(cleaned)
    except Exception as e:
        log(f"[CSV][ERROR] Append failed: {e}")
        return 0

# -------------------------
# Main tracking logic
# -------------------------
def run_lock(pe, strategy: str, day: str) -> int:
    if is_locked(strategy, day):
        log(f"[TRACK] {strategy} already ran for {day} (lock exists). Skipping.")
        return 0

    log(f"[TRACK] running {strategy} for {day}")

    candidate_funcs = [
        "track_lock_for_day",
        "run_lock_for_day",
        "perf_run_lock_for_day",
        "perf_track_lock",
    ]

    new_rows = []

    for fname in candidate_funcs:
        fn = getattr(pe, fname, None)
        if callable(fn):
            try:
                out = fn(day=day, strategy=strategy)
            except TypeError:
                out = fn(day, strategy)

            if isinstance(out, list):
                new_rows = out
            elif isinstance(out, tuple) and out and isinstance(out[0], list):
                new_rows = out[0]
            else:
                new_rows = []
            break

    if not new_rows:
        cities = getattr(pe, "CITIES", None) or ["Philadelphia", "Los Angeles", "Denver", "Miami", "NYC", "Chicago", "Austin"]
        compute_fn = getattr(pe, "compute_pick_for_today", None) or getattr(pe, "compute_pick", None)

        if not callable(compute_fn):
            log(f"[TRACK][{strategy}] No compute function found in philly_edge.py (expected compute_pick_for_today).")
        else:
            for city in cities:
                try:
                    try:
                        result = compute_fn(city=city, strategy=strategy)
                    except TypeError:
                        result = compute_fn(city, strategy)

                    row = None
                    if isinstance(result, dict):
                        row = result
                    elif isinstance(result, tuple):
                        for item in result:
                            if isinstance(item, dict):
                                row = item
                                break

                    if row:
                        row.setdefault("strategy", strategy)
                        row.setdefault("date", day)
                        new_rows.append(row)
                except Exception as e:
                    log(f"[TRACK][{strategy}][{city}] ERROR: {e}")

    wrote = write_rows_append(new_rows)

    if wrote > 0:
        set_locked(strategy, day, detail=f"wrote={wrote}")
        log(f"[TRACK] {strategy}: wrote {wrote} rows and set lock.")
    else:
        log(f"[TRACK] {strategy}: wrote 0 rows, not creating lock file")

    return wrote

def run_settlement(pe) -> None:
    fn = getattr(pe, "perf_update_outcomes", None) or getattr(pe, "update_outcomes", None)
    if not callable(fn):
        log("[SETTLE] No settlement function found (perf_update_outcomes). Skipping.")
        return
    try:
        log("[SETTLE] Starting settlement sweep...")
        fn()
        log("[SETTLE] Settlement sweep complete.")
    except Exception as e:
        log(f"[SETTLE][ERROR] Settlement sweep failed (non-fatal): {e}")

def main() -> int:
    ensure_kalshi_key_material()

    try:
        import philly_edge as pe
    except Exception as e:
        log(f"[FATAL] Could not import philly_edge.py: {e}")
        log(traceback.format_exc())
        return 0

    # Always attempt yesterday + today in CST (backfill safety)
    days = [day_str_cst(-1), day_str_cst(0)]
    strategies = ("lock_0930", "lock_1200")

    total_written = 0
    for day in days:
        for strat in strategies:
            try:
                total_written += run_lock(pe, strat, day)
            except Exception as e:
                log(f"[TRACK][{strat}][{day}][ERROR] Non-fatal: {e}")
                log(traceback.format_exc())

    run_settlement(pe)

    log(f"[DONE] total_rows_written={total_written} days={days}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
