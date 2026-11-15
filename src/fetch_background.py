"""Step 2b: fire-free "background" chips, to measure false alarms on ordinary land.

A global alert system scans mostly fire-free ground, so its most important number is how often it
cries wolf there. For every fire chip, this cuts up to PER_CHIP more chips from the SAME Sentinel-2
image, 25-50 km away, where no VIIRS detection from +-2 days lies within 5 km of the window.
Same image = same date, sun angle and region, so the only thing that differs is the fire.

Background chips keep their fire chip's site id, so they land in the same train / val / test split.
Saved as data/chips/<fire chip id>_bg<k>.npz in the same format as fire chips.
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fetch_s2 import CHIPS, ROOT, load_fires, read_window, save_chip

ITEM_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{}"
PER_CHIP = 2
OFFSETS_KM = [25, 40]
CLEAR_KM = 5 + 5.12          # no VIIRS point within 5 km of the 10.24 km window (centre + half width)


def get_item(scene_id: str) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(ITEM_URL.format(scene_id), timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {scene_id}")


def fire_free(fires: pd.DataFrame, lat: float, lon: float, date: str) -> bool:
    dlat = CLEAR_KM / 111.0
    dlon = CLEAR_KM / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    near = fires[fires["latitude"].between(lat - dlat, lat + dlat)
                 & fires["longitude"].between(lon - dlon, lon + dlon)]
    days = (pd.to_datetime(near["acq_date"]) - pd.Timestamp(date)).dt.days.abs()
    return not (days <= 2).any()


def background_for(path: Path, fires: pd.DataFrame) -> list[str]:
    meta = json.loads(str(np.load(path)["meta"]))
    msgs, done = [], 0
    item = None
    for km in OFFSETS_KM:
        for bearing in (0, 90, 180, 270, 45, 135, 225, 315):
            if done >= PER_CHIP:
                return msgs
            chip_id = f"{meta['chip_id']}_bg{done}"
            out = CHIPS / f"{chip_id}.npz"
            if out.exists():
                done += 1
                continue
            lat = meta["lat"] + km / 111.0 * math.cos(math.radians(bearing))
            lon = meta["lon"] + km / (111.0 * max(math.cos(math.radians(meta["lat"])), 0.1)) * math.sin(math.radians(bearing))
            if not fire_free(fires, lat, lon, meta["date"]):
                continue
            item = item or get_item(meta["scene"])
            w = read_window(item, lon, lat)
            if w is None:
                continue
            msg = save_chip(out, chip_id, item, w, meta["site_id"], meta["date"], lat, lon, fires)
            if "saved" in msg:
                done += 1
                msgs.append(msg)
    if done < PER_CHIP:
        msgs.append(f"{meta['chip_id']}: only {done} background chip(s) found")
    return msgs


def main() -> None:
    fires = load_fires()
    fire_chips = [p for p in sorted(CHIPS.glob("*.npz")) if "_bg" not in p.stem]
    print(f"{len(fire_chips)} fire chips -> up to {PER_CHIP * len(fire_chips)} background chips")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(background_for, p, fires): p for p in fire_chips}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                for m in fut.result():
                    print(f"[{i}/{len(fire_chips)}] {m}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(fire_chips)}] {futures[fut].stem}: error {type(e).__name__}: {e}", flush=True)
    print(f"done: {len(list(CHIPS.glob('*_bg*.npz')))} background chips on disk")


if __name__ == "__main__":
    main()
