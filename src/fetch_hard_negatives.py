"""Fire-free "hard" examples the models barely saw: cities (roofs, shadows, winter sun) and cloud edges.

The first website test showed false alarms in big cities and next to clouds; the training's fire-free
chips were all countryside next to fires. Each chip here is checked against NASA VIIRS (no detection within
5 km of the window, +-1 day). Saved as background chips: data/chips/h<kind>_<place>_<date>_bg0.npz,
site id h_<place>, so they spread over train / val / test and count in the false-alarm score.
"""
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

from fetch_s2 import CHIPS, ROOT, STAC, read_window, save_chip

FIRMS = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{area}/3/{day}"
CLEAR_KM = 5 + 5.12
# (name, lat, lon): big cities on every continent and latitude, incl. industrial and desert cities.
CITIES = [
    ("tokyo", 35.68, 139.76), ("osaka", 34.69, 135.50), ("seoul", 37.57, 126.98), ("beijing", 39.91, 116.40),
    ("shanghai", 31.23, 121.47), ("guangzhou", 23.13, 113.26), ("chengdu", 30.66, 104.06), ("wuhan", 30.59, 114.30),
    ("hongkong", 22.32, 114.17), ("taipei", 25.03, 121.56), ("manila", 14.60, 120.98), ("bangkok", 13.75, 100.50),
    ("hanoi", 21.03, 105.85), ("jakarta", -6.21, 106.85), ("singapore", 1.35, 103.82), ("kualalumpur", 3.14, 101.69),
    ("delhi", 28.61, 77.21), ("mumbai", 19.08, 72.88), ("kolkata", 22.57, 88.36), ("bangalore", 12.97, 77.59),
    ("karachi", 24.86, 67.01), ("lahore", 31.55, 74.34), ("dhaka", 23.81, 90.41), ("kathmandu", 27.72, 85.32),
    ("tehran", 35.69, 51.39), ("zanjan", 36.67, 48.48), ("riyadh", 24.71, 46.68), ("dubai", 25.20, 55.27),
    ("baghdad", 33.31, 44.37), ("istanbul", 41.01, 28.98), ("ankara", 39.93, 32.86), ("telaviv", 32.08, 34.78),
    ("cairo", 30.04, 31.24), ("casablanca", 33.57, -7.59), ("agadir", 30.42, -9.58), ("algiers", 36.75, 3.06),
    ("lagos", 6.52, 3.38), ("accra", 5.60, -0.19), ("kinshasa", -4.44, 15.27), ("nairobi", -1.29, 36.82),
    ("addisababa", 9.03, 38.74), ("johannesburg", -26.20, 28.05), ("capetown", -33.92, 18.42), ("luanda", -8.84, 13.23),
    ("khartoum", 15.50, 32.56), ("dakar", 14.72, -17.47), ("london", 51.51, -0.13), ("paris", 48.86, 2.35),
    ("berlin", 52.52, 13.40), ("madrid", 40.42, -3.70), ("rome", 41.90, 12.50), ("milan", 45.46, 9.19),
    ("warsaw", 52.23, 21.01), ("moscow", 55.76, 37.62), ("stpetersburg", 59.93, 30.36), ("kyiv", 50.45, 30.52),
    ("stockholm", 59.33, 18.07), ("oslo", 59.91, 10.75), ("amsterdam", 52.37, 4.90), ("brussels", 50.85, 4.35),
    ("vienna", 48.21, 16.37), ("athens", 37.98, 23.73), ("lisbon", 38.72, -9.14), ("ruhr", 51.46, 7.01),
    ("newyork", 40.71, -74.01), ("losangeles", 34.05, -118.24), ("chicago", 41.88, -87.63), ("houston", 29.76, -95.37),
    ("phoenix", 33.45, -112.07), ("denver", 39.74, -104.99), ("seattle", 47.61, -122.33), ("miami", 25.76, -80.19),
    ("atlanta", 33.75, -84.39), ("dallas", 32.78, -96.80), ("toronto", 43.65, -79.38), ("montreal", 45.50, -73.57),
    ("vancouver", 49.28, -123.12), ("calgary", 51.05, -114.07), ("mexicocity", 19.43, -99.13), ("monterrey", 25.69, -100.32),
    ("bogota", 4.71, -74.07), ("lima", -12.05, -77.04), ("santiago", -33.45, -70.67), ("buenosaires", -34.60, -58.38),
    ("saopaulo", -23.55, -46.63), ("riodejaneiro", -22.91, -43.17), ("brasilia", -15.79, -47.88), ("caracas", 10.48, -66.90),
    ("sydney", -33.87, 151.21), ("melbourne", -37.81, 144.96), ("brisbane", -27.47, 153.03), ("perth", -31.95, 115.86),
    ("auckland", -36.85, 174.76), ("anchorage", 61.22, -149.90), ("reykjavik", 64.15, -21.94), ("helsinki", 60.17, 24.94),
    ("novosibirsk", 55.01, 82.93), ("almaty", 43.24, 76.89), ("tashkent", 41.30, 69.24), ("ulaanbaatar", 47.89, 106.91),
    ("harbin", 45.80, 126.53), ("vladivostok", 43.12, 131.89), ("lasvegas", 36.17, -115.14), ("saltlake", 40.76, -111.89),
]
SEASONS = [("2026-01-05", "2026-02-20"), ("2026-06-20", "2026-08-10")]
CLOUD_EDGE_OFFSET_KM = 70             # cloud-edge chips: countryside 70 km from each city


def key() -> str:
    return (ROOT / "firms_key.txt").read_text().strip()


def search(lon: float, lat: float, start: str, end: str, cloud_lo: float, cloud_hi: float) -> dict | None:
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 60,
            "query": {"eo:cloud_cover": {"gte": cloud_lo, "lte": cloud_hi}}}
    for attempt in range(4):
        try:
            feats = requests.post(STAC, json=body, timeout=60).json().get("features", [])
            break
        except (requests.RequestException, ValueError):
            time.sleep(3 * (attempt + 1))
    else:
        return None
    if not feats:
        return None
    mid = (cloud_lo + cloud_hi) / 2
    return min(feats, key=lambda f: abs(f["properties"]["eo:cloud_cover"] - mid))


def viirs(lon: float, lat: float, day: str) -> pd.DataFrame:
    """VIIRS detections near the point, from the day before to the day after."""
    dlat = CLEAR_KM / 111.0
    dlon = CLEAR_KM / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    area = f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}"
    start = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    frames = []
    for src in ("VIIRS_NOAA20_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"):
        for attempt in range(3):
            try:
                r = requests.get(FIRMS.format(key=key(), src=src, area=area, day=start), timeout=60)
                if r.ok and r.text.startswith("latitude"):
                    frames.append(pd.read_csv(io.StringIO(r.text)))
                break
            except requests.RequestException:
                time.sleep(3)
    cols = ["latitude", "longitude", "acq_date", "acq_time", "frp"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames)
    return df[cols].drop_duplicates()


def make(kind: str, name: str, lat: float, lon: float, start: str, end: str) -> str:
    cloud = (0, 25) if kind == "city" else (25, 60)
    it = search(lon, lat, start, end, *cloud)
    if it is None:
        return f"{kind} {name}: no photo"
    day = it["properties"]["datetime"][:10]
    chip_id = f"h{kind}_{name}_{day}_bg0"
    out = CHIPS / f"{chip_id}.npz"
    if out.exists():
        return f"{chip_id}: cached"
    fires = viirs(lon, lat, day)
    if len(fires):
        return f"{chip_id}: skipped ({len(fires)} VIIRS detections nearby)"
    w = read_window(it, lon, lat)
    if w is None:
        return f"{chip_id}: too close to photo edge"
    return save_chip(out, chip_id, it, w, f"h_{name}", day, lat, lon, fires)


def main() -> None:
    jobs = []
    for name, lat, lon in CITIES:
        for start, end in SEASONS:
            jobs.append(("city", name, lat, lon, start, end))
        # cloud edges: countryside 70 km east of the city, one season
        dlon = CLOUD_EDGE_OFFSET_KM / (111.0 * max(math.cos(math.radians(lat)), 0.1))
        jobs.append(("cloud", name, lat, lon + dlon, *SEASONS[len(name) % 2]))
    saved = 0
    with ThreadPoolExecutor(6) as pool:
        futs = [pool.submit(make, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:
                msg = f"error {type(e).__name__}: {e}"
            saved += "saved" in msg
            print(f"[{i}/{len(jobs)} saved {saved}] {msg}", flush=True)
    print(f"done: {saved} new hard fire-free chips")


if __name__ == "__main__":
    main()
