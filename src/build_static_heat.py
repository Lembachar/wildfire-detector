"""Build a world map of permanent heat sources (gas flares, factories, volcanoes) from NASA's FIRMS archive.

NASA's standard-processing VIIRS archive marks each heat detection with a type: 0 = presumed vegetation fire,
1 = active volcano, 2 = other static land source (industry, flares), 3 = offshore. Types 1-3 are places that
are hot over and over, not fires. This downloads one year of the archive for the whole world ONCE, keeps the
type 1-3 detections, and grids them into a small file that ships with the website (no download per check).

Stage 1 (download, resumable): data/static_heat/<source>_<date>.csv with only type 1-3 rows.
Stage 2 (grid): data/static_heat.npz: cell centre lat/lon (0.004 deg, ~400 m), detections, distinct days, type.

    python src/build_static_heat.py
"""
import csv
import io
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "static_heat"
SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]
START, END = date(2025, 8, 1), date(2026, 7, 31)     # latest full year in the archive
DAYS = 5                                              # longest range the area API allows
CELL = 0.004                                          # deg, about one 375 m VIIRS pixel
COST = 200                                            # transactions one world-wide 5-day call uses (measured 180)
API = "https://firms.modaps.eosdis.nasa.gov"


def wait_for_quota(key: str) -> None:
    while True:
        try:
            s = requests.get(f"{API}/mapserver/mapkey_status/?MAP_KEY={key}", timeout=30).json()
            if int(s["current_transactions"]) + COST <= int(s["transaction_limit"]):
                return
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(30)


def download(key: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = [START + timedelta(days=i) for i in range(0, (END - START).days + 1, DAYS)]
    todo = [(s, d) for s in SOURCES for d in days if not (OUT / f"{s}_{d}.csv").exists()]
    print(f"{len(todo)} of {len(SOURCES) * len(days)} downloads left", flush=True)
    for i, (src, d) in enumerate(todo):
        n = min(DAYS, (END - d).days + 1)
        for attempt in range(6):
            wait_for_quota(key)
            try:
                r = requests.get(f"{API}/api/area/csv/{key}/{src}/world/{n}/{d}", timeout=900)
                r.raise_for_status()
                if not r.text.startswith("latitude"):
                    raise ValueError(r.text[:100])
                break
            except (requests.RequestException, ValueError) as e:
                print(f"  {src} {d}: retry {attempt + 1} ({type(e).__name__})", flush=True)
                time.sleep(60)
        else:
            print(f"  {src} {d}: FAILED, rerun to retry", flush=True)
            continue
        rows = list(csv.DictReader(io.StringIO(r.text)))
        keep = [x for x in rows if x.get("type") in ("1", "2", "3")]
        tmp = OUT / f"{src}_{d}.tmp"
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["latitude"])
            w.writeheader()
            w.writerows(keep)
        tmp.replace(OUT / f"{src}_{d}.csv")
        print(f"[{i + 1}/{len(todo)}] {src} {d}: {len(rows):,} detections, {len(keep):,} permanent heat",
              flush=True)


def grid() -> None:
    det, days, typ = defaultdict(int), defaultdict(set), {}
    for f in sorted(OUT.glob("*.csv")):
        with f.open() as fh:
            for x in csv.DictReader(fh):
                k = (round(float(x["latitude"]) / CELL), round(float(x["longitude"]) / CELL))
                det[k] += 1
                days[k].add(x["acq_date"])
                typ[k] = max(typ.get(k, 0), int(x["type"]))
    keys = sorted(det)
    np.savez_compressed(ROOT / "data" / "static_heat.npz",
                        lat=np.array([k[0] * CELL for k in keys], np.float32),
                        lon=np.array([k[1] * CELL for k in keys], np.float32),
                        detections=np.array([det[k] for k in keys], np.int32),
                        days=np.array([len(days[k]) for k in keys], np.int16),
                        type=np.array([typ[k] for k in keys], np.int8),
                        cell_deg=CELL, period=f"{START}/{END}", sources=",".join(SOURCES))
    print(f"saved data/static_heat.npz: {len(keys):,} cells", flush=True)


if __name__ == "__main__":
    download((ROOT / "firms_key.txt").read_text().strip())
    missing = [(s, d) for s in SOURCES for d in
               (START + timedelta(days=i) for i in range(0, (END - START).days + 1, DAYS))
               if not (OUT / f"{s}_{d}.csv").exists()]
    if missing:
        print(f"{len(missing)} downloads still missing: not gridding an incomplete year", flush=True)
    else:
        grid()
