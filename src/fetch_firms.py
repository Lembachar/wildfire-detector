"""Step 1: download recent VIIRS active-fire detections from NASA FIRMS and pick fire sites.

Uses the public 7-day global CSV feeds, so no FIRMS MAP_KEY is needed.
Output:
  data/raw/firms_*.csv      cached downloads
  data/firms.csv            daytime, nominal/high-confidence detections from all satellites
  data/sites.csv            one row per (site, date): the places worth fetching imagery for
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://firms.modaps.eosdis.nasa.gov/data/active_fire"
FEEDS = {
    "SNPP": f"{BASE}/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv",
    "NOAA20": f"{BASE}/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv",
    "NOAA21": f"{BASE}/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_7d.csv",
}
CELL_DEG = 0.2          # sites are 0.2 degree grid cells (~22 km)
MIN_DETECTIONS = 3      # low on purpose: small fires (a few detections) must be in the data too
MAX_SITES = 2000        # cap on (site, date) candidates passed to the imagery step


def download(name: str, url: str) -> Path:
    out = RAW / f"firms_{name}.csv"
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"downloading {name} ...")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(out)
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, url in FEEDS.items():
        try:
            df = pd.read_csv(download(name, url))
        except Exception as e:  # one feed missing should not stop the run
            print(f"skipping {name}: {e}")
            continue
        df["sat"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Sentinel-2 only images in daylight, and low-confidence VIIRS points are mostly noise.
    # The feeds spell confidence out ("nominal"/"high"); archive files use "n"/"h".
    df = df[(df["daynight"] == "D") & (df["confidence"].isin(["n", "h", "nominal", "high"]))].copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    df["cell_x"] = np.floor(df["longitude"] / CELL_DEG).astype(int)
    df["cell_y"] = np.floor(df["latitude"] / CELL_DEG).astype(int)
    df.to_csv(ROOT / "data" / "firms.csv", index=False)
    print(f"{len(df):,} daytime detections, {df['acq_date'].min()} to {df['acq_date'].max()}")

    # Centre each (cell, date) on the detections themselves, not the cell corner.
    g = df.groupby(["cell_x", "cell_y", "acq_date"])
    sites = g.agg(n=("latitude", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                  frp=("frp", "sum")).reset_index()
    sites = sites[sites["n"] >= MIN_DETECTIONS]
    sites["site_id"] = sites["cell_x"].astype(str) + "_" + sites["cell_y"].astype(str)
    # Spread candidates over many places instead of many dates of the same fire.
    sites = sites.sort_values("n", ascending=False)
    sites["rank_in_site"] = sites.groupby("site_id").cumcount()
    sites = sites.sort_values(["rank_in_site", "n"], ascending=[True, False]).head(MAX_SITES)
    sites.to_csv(ROOT / "data" / "sites.csv", index=False)
    print(f"{len(sites)} candidate (site, date) pairs across {sites['site_id'].nunique()} sites")


if __name__ == "__main__":
    main()
