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

Notes:
- In GitHub Actions, secrets passed via env are strings; base64 is usually safest.
- This script will decode B64 -> /tmp/kalshi.key and set KALSHI_PRIVATE_KEY_PATH for philly_edge.py
"""

import os
import sys
import csv
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
def _clean_b64(s: str) -> str:
    """Remove whitespace/newlines so base64 decoding works even if pasted with line breaks."""
    return "".join((s or "").split())


def _looks_like_pem(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("-----BEGIN") and "PRIVATE KEY" in t


def has_kalshi_key_material() -> bool:
    """Used to decide if settlement is safe to run."""
    if os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip():
        return True
    if os.getenv("KALSHI_PRIVATE_KEY_PEM", "").strip():
        return True
    if os.getenv("KALSHI_PRIVATE_KEY_B64", "").strip():
        return True
    # Some people store PEM-as-base64 in KALSHI_PRIVATE_KEY_PEM by mistake
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    if pem and _clean_b64(pem) and not _looks_like_pem(pem):
        return True
    return False


def ensure_kalshi_key_material() -> None:
    """
    Ensures Kalshi key is available for philly_edge.py.
    Supports:
      - KALSHI_PRIVATE_KEY_PEM (raw multiline PEM)
      - KALSHI_PRIVATE_KEY_B64 (base64 of PEM bytes)  <-- recommended in Actions
      - KALSHI_PRIVATE_KEY_PATH (file path)
    Also handles the common mistake: KALSHI_PRIVATE_KEY_PEM accidentally contains base64.
    """
    key_id = (os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID") or "").strip()
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "") or ""
    b64 = os.getenv("KALSHI_PRIVATE_KEY_B64", "") or ""
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or os.getenv("KALSHI_API_PRIVATE_KEY_PATH") or "").strip()

    pem_stripped = pem.strip()
    b64_clean = _clean_b64(b64)

    log(
        f"[ENV] KEY_ID_SET={bool(key_id)} "
        f"PEM_LEN={len(pem_stripped)} B64_LEN={len(b64_clean)} PATH_SET={bool(key_path)}"
    )

    # If path already provided, nothing to do
    if key_path:
        return

    # If PEM looks valid, leave it to philly_edge (it reads *_PEM)
    if pem_stripped and _looks_like_pem(pem_stripped):
        return

    # If B64 exists, decode it into a file and set *_PATH
    if b64_clean:
        tmp_key = Path("/tmp/kalshi.key")
        try:
            raw = base64.b64decode(b64_clean.encode("utf-8"))
            tmp_key.write_bytes(raw)
            os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(tmp_key)
            log(f"[ENV] Decoded KALSHI_PRIVATE_KEY_B64 -> {tmp_key} ({tmp_key.stat().st_size} bytes)")
        except Exception as e:
            log(f"[ENV][ERROR] Failed to decode KALSHI_PRIVATE_KEY_B64: {e}")
        return

    # Common mistake: PEM env var contains base64 instead of PEM text
    pem_b64_clean = _clean_b64(pem_stripped)
    if pem_b64_clean and not _looks_like_pem(pem_stripped):
        tmp_key = Path("/tmp/kalshi.key")
        try:
            raw = base64.b64decode(pem_b64_clean.encode("utf-8"))
            tmp_key.write_bytes(raw)
            os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(tmp_key)
            # Clear the PEM var so philly_edge doesn't try to parse base64 as PEM
            os.environ.pop("KALSHI_PRIVATE_KEY_PEM", None)
            log(f"[ENV] Decoded (PEM-as-B64) -> {tmp_key} ({tmp_key.stat().st_size} bytes)")
        except Exception as e:
            log(f"[ENV][ERROR] Failed to decode KALSHI_PRIVATE_KEY_PEM as base64: {e}")
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
        # remove None keys to avoid csv.DictWriter crash
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
    """
    Run one lock strategy (lock_0930 or lock_1200).
    Tries multiple function names to match evolving philly_edge.py.
    Returns number of rows appended.
    """
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
                if isinstance(out, list):
                    new_rows = out
                elif isinstance(out, tuple) and out and isinstance(out[0], list):
                    new_rows = out[0]
                else:
                    new_rows = []
                break
            except TypeError:
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
        cities = getattr(pe, "CITIES", None) or ["Philadelphia", "Los Angeles", "Denver", "Miami", "NYC", "Chicago", "Austin"]

        compute_fn = getattr(pe, "compute_pick_for_today", None) or getattr(pe, "compute_pick", None)
        if not callable(compute_fn):
            log(f"[TRACK][{strategy}] No compute function found in philly_edge.py (expected compute_pick_for_today).")
        else:
            for city in cities:
                try:
                    result = compute_fn(city=city, strategy=strategy)
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
    """
    Settlement sweep: update outcomes using NOAA/NWS observed highs.
    Hard-wrapped so workflow never fails.
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
        log(traceback.format_exc())


def main() -> int:
    # Ensure key material is wired for philly_edge
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

    # IMPORTANT: only run settlement if we actually have key material.
    # This avoids crashes inside pe.perf_update_outcomes() during "missing key" runs.
    if has_kalshi_key_material():
        run_settlement(pe)
    else:
        log("[SETTLE] Skipping settlement sweep (no Kalshi key material present).")

    log(f"[DONE] total_rows_written={total_written}")
    return 0  # Always succeed so schedule keeps running


if __name__ == "__main__":
    sys.exit(main())
