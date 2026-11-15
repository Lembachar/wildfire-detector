""""Before" picture for every chip: a clear-pixel composite of the photos from the 15 days before it.

Used two ways: (1) to tune and score the website's "was it already there?" check on the real test set,
(2) as the extra input for a model that sees before + now (temporal model).

For each chip, earlier Sentinel-2 photos on the same 20 m grid (same UTM zone: MGRS tiles share it) are
read over exactly the chip's pixels. Cloud, cloud shadow, defective and missing pixels are dropped (SCL),
then per pixel:
  before  float16 (2, H, W)  median B12, B11 over the clear earlier photos (nan = never clear)
  heat_max float16 (H, W)     highest B12 - B11 seen before (catches heat that was there only sometimes)
  n_clear uint8   (H, W)     how many earlier photos saw that pixel clearly
saved to data/history/<chip>.npz, with the dates used. Safe to re-run: done chips are skipped.

    python src/fetch_history.py
"""
import json
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import rasterio
import requests
from rasterio.windows import Window

from fetch_s2 import CHIPS, GDAL_ENV, ROOT, STAC, scale_offset

OUT = ROOT / "data" / "history"
LOOKBACK_DAYS, MAX_PHOTOS, CHIP = 15, 3, 512
UNCLEAR_SCL = [0, 1, 3, 8, 9, 10]
KEYS = ["swir22", "swir16"]          # B12, B11: all the check needs (the connection is the bottleneck)
ENV = dict(GDAL_ENV, GDAL_INGESTED_BYTES_AT_OPEN="32768", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2")
_proxy = urllib.request.getproxies().get("https") or urllib.request.getproxies().get("http")
if _proxy:                                     # GDAL ignores the Windows system proxy: very slow without it
    ENV.update(GDAL_HTTP_PROXY=_proxy.split("://")[-1].rstrip("/"))


def earlier_items(meta: dict) -> list[dict]:
    t = datetime.fromisoformat(meta["s2_datetime"].replace("Z", "+00:00"))
    f = "%Y-%m-%dT%H:%M:%SZ"
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [meta["lon"], meta["lat"]]},
            "datetime": f"{(t - timedelta(days=LOOKBACK_DAYS)).strftime(f)}/{(t - timedelta(hours=1)).strftime(f)}",
            "limit": 40, "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    for attempt in range(4):
        try:
            return requests.post(STAC, json=body, timeout=60).json().get("features", [])
        except (requests.RequestException, ValueError):
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("STAC search failed")


def window_in(item: dict, meta: dict) -> Window | None:
    """The chip's pixels in this photo's grid, or None if it's another grid (other UTM zone) or misaligned."""
    a = item["assets"]["swir22"]
    epsg = item["properties"].get("proj:epsg") or a.get("proj:epsg")
    tf = a.get("proj:transform")
    if not tf or f"EPSG:{epsg}" != meta["crs"]:
        return None
    col = (meta["transform"][2] - tf[2]) / tf[0]
    row = (meta["transform"][5] - tf[5]) / tf[4]
    if abs(col - round(col)) > 0.01 or abs(row - round(row)) > 0.01:
        return None
    return Window(round(col), round(row), CHIP, CHIP)


def read(asset: dict, win: Window, props: dict | None, scl: bool = False) -> np.ndarray:
    for attempt in range(3):
        try:
            with rasterio.Env(**ENV), rasterio.open(asset["href"]) as src:
                dn = src.read(1, window=win, boundless=True, fill_value=0)
            break
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(2)
    if scl:
        return dn
    s, o = scale_offset(asset, props)
    v = dn.astype(np.float32) * s + o
    v[dn == 0] = np.nan
    return v


def one(path) -> str:
    out = OUT / path.name
    if out.exists():
        return "cached"
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    photos, seen = [], set()
    for it in earlier_items(meta):
        key = it["properties"]["datetime"][:13]
        if key in seen or it["properties"].get("eo:cloud_cover", 100) > 95:
            continue
        win = window_in(it, meta)
        if win is not None:
            seen.add(key)
            photos.append((it, win))
        if len(photos) == MAX_PHOTOS:
            break
    stack, dates = [], []
    for it, win in photos:
        a, pr = it["assets"], it["properties"]
        scl = read(a["scl"], win, None, scl=True)
        bad = np.isin(scl, UNCLEAR_SCL)
        if bad.mean() > 0.97:                                      # fully cloudy over the chip: skip 3 reads
            continue
        bands = np.stack([read(a[k], win, pr) for k in KEYS])
        bands[:, bad] = np.nan
        stack.append(bands)
        dates.append(pr["datetime"])
    if stack:
        s = np.stack(stack)                                       # (photos, 3, H, W)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            before = np.nanmedian(s, axis=0)
            heat_max = np.nanmax(s[:, 0] - s[:, 1], axis=0)
        n_clear = np.isfinite(s[:, 0]).sum(axis=0).astype(np.uint8)
    else:
        before = np.full((2, CHIP, CHIP), np.nan, np.float32)
        heat_max = np.full((CHIP, CHIP), np.nan, np.float32)
        n_clear = np.zeros((CHIP, CHIP), np.uint8)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, before=before.astype(np.float16), heat_max=heat_max.astype(np.float16), n_clear=n_clear,
                        meta=json.dumps(dict(chip=path.stem, dates=dates, now=meta["s2_datetime"])))
    tmp.replace(out)
    return f"{len(dates)} clear photos, {float((n_clear > 0).mean()):.0%} of pixels seen"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted(CHIPS.glob("*.npz")) if ".tmp" not in p.name]
    # val / test photos first (needed to score the check), training photos after (for the temporal model)
    prep = ROOT / "data" / "prepared"
    if (prep / "tiles.npz").exists():
        t = np.load(prep / "tiles.npz")
        names = json.loads((prep / "chips.json").read_text())["chips"]
        held_out = {names[c] for c in np.unique(t["chip"][t["split"] > 0])}
        paths.sort(key=lambda p: p.stem not in held_out)
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(one, p): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:
                msg = f"error {type(e).__name__}: {e}"
            print(f"[{i}/{len(paths)} {time.time() - t0:5.0f}s] {futs[f].stem}: {msg}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
