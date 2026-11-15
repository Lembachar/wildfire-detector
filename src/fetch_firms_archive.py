"""Step 1b: past fires from the FIRMS archive, to cover fire types the last week doesn't have.

Needs a free FIRMS MAP_KEY in firms_key.txt (project root, one line, never committed to git).
Get one at https://firms.modaps.eosdis.nasa.gov/api/map_key/

Regions are chosen for variety: forests, fires reaching towns, volcanoes over houses, industry,
and seasons / continents the 7-day feed lacked. Output, in the same format as fetch_firms.py:
  data/firms_archive.csv   detections (with a 'region' column)
  data/sites_archive.csv   candidate (site, date) pairs, at most PER_REGION per region
"""
import io
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fetch_firms import CELL_DEG, MIN_DETECTIONS

ROOT = Path(__file__).resolve().parents[1]
API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{start}"
SOURCES = ["VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"]
PER_REGION = 120

# name: (west, south, east, north), [start dates of 5-day windows], kind
REGIONS = {
    "california":        ((-124.5, 32.5, -114.0, 42.0), ["2024-07-25", "2024-09-08", "2025-01-08"], "forest/urban"),
    "canada_boreal":     ((-125.0, 49.0, -60.0, 62.0), ["2023-06-05", "2023-07-10", "2025-06-01"], "forest"),
    "mediterranean":     ((-10.0, 35.0, 30.0, 45.0), ["2023-07-18", "2023-08-20", "2024-08-11", "2025-08-12"], "forest/urban"),
    "amazon_cerrado":    ((-75.0, -20.0, -40.0, 0.0), ["2024-08-20", "2024-09-10"], "forest/savanna"),
    "siberia":           ((100.0, 50.0, 140.0, 70.0), ["2021-07-15", "2024-05-05"], "forest"),
    "indonesia":         ((95.0, -6.0, 120.0, 6.0), ["2019-09-15", "2023-09-20"], "peat/forest"),
    "australia_forest":  ((145.0, -39.0, 154.0, -27.0), ["2019-12-30", "2020-01-04"], "forest"),
    "iceland_volcano":   ((-23.0, 63.5, -21.5, 64.3), ["2024-01-14", "2024-08-22", "2024-11-20"], "volcano/urban"),
    "hawaii_volcano":    ((-156.2, 19.0, -154.8, 20.3), ["2023-09-10", "2024-12-23"], "volcano"),
    "lapalma_volcano":   ((-18.1, 28.4, -17.7, 28.9), ["2021-09-20", "2021-10-10"], "volcano/urban"),
    "persian_gulf_industry": ((47.0, 24.0, 57.0, 31.0), ["2024-06-01"], "industrial"),
    "niger_delta_industry":  ((5.0, 4.0, 8.0, 6.0), ["2024-01-10"], "industrial"),
}


def query(key: str, source: str, bbox: tuple, start: str) -> pd.DataFrame:
    url = API.format(key=key, source=source, area=",".join(str(v) for v in bbox), days=5, start=start)
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200 and r.text.startswith("latitude"):
                return pd.read_csv(io.StringIO(r.text))
            if "Invalid" in r.text or r.status_code in (400, 403):
                print(f"    {source} {start}: {r.text.strip()[:120]}")
                return pd.DataFrame()
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    key_file = ROOT / "firms_key.txt"
    if not key_file.exists():
        raise SystemExit("firms_key.txt not found: get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/")
    key = key_file.read_text().strip()
    frames = []
    for region, (bbox, starts, kind) in REGIONS.items():
        n0 = sum(len(f) for f in frames)
        for start in starts:
            for source in SOURCES:
                df = query(key, source, bbox, start)
                if len(df):
                    df["region"], df["kind"], df["sat"] = region, kind, source
                    frames.append(df)
                time.sleep(1)
        print(f"{region:24s} {sum(len(f) for f in frames) - n0:7,} detections")
    df = pd.concat(frames, ignore_index=True)

    conf = df["confidence"].astype(str).str.lower()
    df = df[(df["daynight"] == "D") & conf.isin(["n", "h", "nominal", "high"])].copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    df["cell_x"] = np.floor(df["longitude"] / CELL_DEG).astype(int)
    df["cell_y"] = np.floor(df["latitude"] / CELL_DEG).astype(int)
    df.to_csv(ROOT / "data" / "firms_archive.csv", index=False)

    g = df.groupby(["region", "cell_x", "cell_y", "acq_date"])
    sites = g.agg(n=("latitude", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                  frp=("frp", "sum")).reset_index()
    sites = sites[sites["n"] >= MIN_DETECTIONS]
    sites["site_id"] = "a" + sites["cell_x"].astype(str) + "_" + sites["cell_y"].astype(str)
    # Per region: spread over places first, then dates; small fires stay in (MIN_DETECTIONS is low).
    sites = sites.sample(frac=1, random_state=0)
    sites["rank_in_site"] = sites.groupby("site_id").cumcount()
    sites = (sites.sort_values(["region", "rank_in_site"]).groupby("region").head(PER_REGION))
    sites.to_csv(ROOT / "data" / "sites_archive.csv", index=False)
    print(f"{len(df):,} daytime detections -> {len(sites)} candidate (site, date) pairs")
    print(sites.groupby("region").size().to_string())


if __name__ == "__main__":
    main()
