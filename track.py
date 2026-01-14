#!/usr/bin/env python3
"""
Weather Edge Tracker (GitHub Actions)

Goals:
- Run lock_0930 and lock_1200 picks (idempotent via lock files)
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
import json
import time
import base64
import traceback
from datetime import datetime, timezone
from pathlib import Path

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

# Canonical columns (match your current performance.csv shape)
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
        # logging must never fail the run
        pass


def today_str_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def lock_file(strategy: str, day: str) -> Path:
    return LOCK_DIR / f"{day}_{strategy}.lock"


def is_locked(strategy: str, day: str) -> bool:
    return lock_file(strategy, day).exists()


def set_locked(strategy: str, day: str, detail: str = "") -> None:
    lf = lock_file(strategy, day)
    try:
        lf.write_text(detail or "ok", encoding="utf-8")
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
    key_id = os.getenv("KALSHI_KEY_ID", "").strip() or os.getenv("KALSHI_API_KEY_ID", "").strip()
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    b64 = os.getenv("KALSHI_PRIVATE_KEY_B64", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip() or os.getenv("KALSHI_API_PRIVATE_KEY_PATH", "").strip()

    log(f"[ENV] KEY_ID_SET={bool(key_id)} PEM_LEN={len(pem.strip())} B64_LEN={len(b64)} PATH_SET={bool(key_path)}")

    # If path already provided, nothing to do
    if key_path:
        return

    # If PEM is present, leave it to philly_edge (it reads *_PEM)
    if pem.strip():
        return

    # If B64 is present, decode to a temp file and set *_PATH
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

    # If none of the above exist, we just proceed; picks will log missing-key per city.
    log("[ENV][WARN] No Kalshi private key material found (PEM/B64/PATH). Market calls may fail.")


# -------------------------
# CSV helpers
# -------------------------
def read_existing_rows() -> list:
    if not PERF_PATH.exists():
        return []
    try:
        with PERF_PATH.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        log(f"[CSV][ERROR] Failed reading performance.csv: {e}")
        return []


def write_rows_append(new_rows: list) -> int:
    """
    Append rows to performance.csv, creating it with PERF_COLUMNS if missing.
    Filters out bad keys (None) and unknown keys (keeps only PERF_COLUMNS).
    """
    if not new_rows:
        return 0

    # Normalize / filter rows
    cleaned = []
    for r in new_rows:
        if not isinstance(r, dict):
            continue
        # remove None keys to avoid the exact crash you saw
        r = {k: v for k, v in r.items() if k is not None}
        # keep only known columns
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
    """
    Run one lock strategy (lock_0930 or lock_1200).
    Tries multiple function names to match your evolving philly_edge.py.
    Returns number of rows appended.
    """
    if is_locked(strategy, day):
        log(f"[TRACK] {strategy} already ran for {day} (lock exists). Skipping.")
        return 0

    log(f"[TRACK] running {strategy} for {day}")

    # Preferred: philly_edge provides a high-level tracking function
    # We try in order of likely names; fall back to per-city compute.
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
                out = fn(day=day, strategy=strategy)  # might accept named args
                if isinstance(out, list):
                    new_rows = out
                elif isinstance(out, tuple) and out and isinstance(out[0], list):
                    new_rows = out[0]
                else:
                    # could be None; rows might already be written inside pe
                    new_rows = []
                break
            except TypeError:
                # try positional
                try:
                    out = fn(day, strategy)
                    if isinstance(out, list):
                        new_rows = out
                    elif isinstance(out, tuple) and out and isinstance(out[0], list):
                        new_rows = out[0]
                    else:
                        new_rows = []
                    break
                except Exception as e:
                    log(f"[TRACK][{strategy}] {fname} failed: {e}")
            except Exception as e:
                log(f"[TRACK][{strategy}] {fname} failed: {e}")

    # Fallback: compute per-city if no high-level fn exists
    if not new_rows:
        cities = getattr(pe, "CITIES", None)
        if not cities:
            # hard fallback
            cities = ["Philadelphia", "Los Angeles", "Denver", "Miami", "NYC", "Chicago", "Austin"]

        compute_fn = getattr(pe, "compute_pick_for_today", None) or getattr(pe, "compute_pick", None)
        if not callable(compute_fn):
            log(f"[TRACK][{strategy}] No compute function found in philly_edge.py (expected compute_pick_for_today).")
        else:
            for city in cities:
                try:
                    # Most compatible call shape
                    result = compute_fn(city=city, strategy=strategy)
                    # We expect either:
                    # - dict row
                    # - tuple containing a dict row
                    row = None
                    if isinstance(result, dict):
                        row = result
                    elif isinstance(result, tuple):
                        # pick first dict-like
                        for item in result:
                            if isinstance(item, dict):
                                row = item
                                break
                    if row:
                        # ensure strategy/date
                        row.setdefault("strategy", strategy)
                        row.setdefault("date", day)
                        new_rows.append(row)
                except Exception as e:
                    log(f"[TRACK][{strategy}][{city}] ERROR: {e}")

    # Append rows ourselves (safe) if any
    wrote = write_rows_append(new_rows)

    # Only create lock file if something meaningful happened
    if wrote > 0:
        set_locked(strategy, day, detail=f"wrote={wrote}")
        log(f"[TRACK] {strategy}: wrote {wrote} rows and set lock.")
    else:
        log(f"[TRACK] {strategy}: wrote 0 rows, not creating lock file")

    return wrote


def run_settlement(pe) -> None:
    """
    Settlement sweep: update outcomes using NOAA/NWS observed highs.
    We hard-wrap to ensure workflow never fails.
    """
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
    # Secrets / key material
    ensure_kalshi_key_material()

    # Import engine
    try:
        import philly_edge as pe
    except Exception as e:
        log(f"[FATAL] Could not import philly_edge.py: {e}")
        log(traceback.format_exc())
        return 0  # never fail Actions

    day = today_str_utc()

    total_written = 0
    for strat in ("lock_0930", "lock_1200"):
        try:
            total_written += run_lock(pe, strat, day)
        except Exception as e:
            log(f"[TRACK][{strat}][ERROR] Non-fatal: {e}")
            log(traceback.format_exc())

    # Settlement sweep (always attempt; never fatal)
    run_settlement(pe)

    log(f"[DONE] total_rows_written={total_written}")
    return 0  # Always succeed so schedule keeps running


if __name__ == "__main__":
    sys.exit(main())
