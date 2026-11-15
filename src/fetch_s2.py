"""Step 2: for each fire site, find a Sentinel-2 image from the SAME day and cut a 10 km chip.

Imagery: Sentinel-2 L2A Cloud-Optimized GeoTIFFs on AWS, found through the public Earth Search
STAC API (no account needed). Only the chip window is downloaded, not the whole scene.

Each chip (512 x 512 pixels at 20 m = 10.24 km) is saved to data/chips/<id>.npz with:
  swir   float16 (3, H, W)  reflectance of B12, B11, B8A  (the bands where active fire glows)
  vis    float16 (4, H, W)  reflectance of B4, B3, B2, B8 (red, green, blue, near infrared)
  rgb    uint8   (H, W, 3)  true colour, for looking at
  scl    uint8   (H, W)     Sentinel-2 scene classification (clouds, water, nodata)
  hot_*  arrays             VIIRS detections inside the chip (pixel row/col, day offset, FRP, minutes
                            between the VIIRS pass and the Sentinel-2 pass)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
CHIPS = ROOT / "data" / "chips"
STAC = "https://earth-search.aws.element84.com/v1/search"
CHIP = 512                # pixels at 20 m
MAX_CLOUD_SCENE = 60      # % cloud for the whole scene
MAX_CLOUD_CHIP = 0.4      # fraction of the chip that may be cloud
MAX_NODATA_CHIP = 0.1
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="4",
                GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")
SWIR_KEYS = ["swir22", "swir16", "nir08"]   # B12, B11, B8A
VIS_KEYS = ["red", "green", "blue", "nir"]   # B4, B3, B2, B8 (10 m, averaged to 20 m)
CLOUD_SCL = [8, 9, 10]                      # cloud medium, cloud high, cirrus


def find_scene(lon: float, lat: float, date: str) -> dict | None:
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z", "limit": 10}
    for attempt in range(5):           # the STAC server drops connections under load
        try:
            feats = requests.post(STAC, json=body, timeout=60).json().get("features", [])
            break
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    feats = [f for f in feats if f["properties"].get("eo:cloud_cover", 100) <= MAX_CLOUD_SCENE]
    return min(feats, key=lambda f: f["properties"]["eo:cloud_cover"]) if feats else None


def scale_offset(asset: dict, props: dict | None = None) -> tuple[float, float]:
    """Digital number -> reflectance. Earth Search lists offset -0.1 for processing baseline >= 04.00, but
    when the item says earthsearch:boa_offset_applied the stored numbers are ALREADY corrected (open water
    reads ~0 without the offset, -0.1 with it). So the offset is only used when it hasn't been applied."""
    band = (asset.get("raster:bands") or [{}])[0]
    offset = band.get("offset", 0.0)
    if props is not None and props.get("earthsearch:boa_offset_applied"):
        offset = 0.0
    return band.get("scale", 1e-4), offset


def read_window(item: dict, lon: float, lat: float, size: int = CHIP) -> dict | None:
    """Download only a size x size (20 m pixels) window centred on lon/lat from one scene.

    Returns None if the point is too close to the scene edge.
    """
    assets = item["assets"]
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(assets["swir22"]["href"]) as ref:
            crs, tf = ref.crs, ref.transform
            (x,), (y,) = warp_transform("EPSG:4326", crs, [lon], [lat])
            row, col = ref.index(x, y)
            r0, c0 = row - size // 2, col - size // 2
            if r0 < 0 or c0 < 0 or r0 + size > ref.height or c0 + size > ref.width:
                return None
            win = Window(c0, r0, size, size)
            win_tf = rasterio.windows.transform(win, tf)
            bounds = rasterio.windows.bounds(win, tf)

        bands = []
        for key in SWIR_KEYS:
            s, o = scale_offset(assets[key], item["properties"])
            with rasterio.open(assets[key]["href"]) as src:
                dn = src.read(1, window=win).astype(np.float32)
            refl = dn * s + o
            refl[dn == 0] = np.nan
            bands.append(refl)
        with rasterio.open(assets["scl"]["href"]) as src:
            scl = src.read(1, window=win)
        with rasterio.open(assets["visual"]["href"]) as src:
            vwin = from_bounds(*bounds, transform=src.transform)
            rgb = src.read([1, 2, 3], window=vwin, out_shape=(3, size, size),
                           resampling=Resampling.average).transpose(1, 2, 0)
        vis = read_visible(assets, bounds, size, item["properties"])
    return dict(swir=np.stack(bands), vis=vis, scl=scl, rgb=rgb, crs=crs, transform=win_tf)


def read_visible(assets: dict, bounds: tuple, size: int, props: dict | None = None) -> np.ndarray:
    """Reflectance of B4, B3, B2 (red, green, blue) and B8 (near infrared), averaged from 10 m to 20 m.

    These help tell fire from look-alikes: shiny roofs and sun glint are bright in visible light,
    flames mostly aren't, and smoke shows up blue-white. Must run inside rasterio.Env(**GDAL_ENV).
    """
    out = []
    for key in VIS_KEYS:
        s, o = scale_offset(assets[key], props)
        with rasterio.open(assets[key]["href"]) as src:
            win = from_bounds(*bounds, transform=src.transform)
            dn = src.read(1, window=win, out_shape=(size, size), resampling=Resampling.average).astype(np.float32)
        refl = dn * s + o
        refl[dn == 0] = np.nan
        out.append(refl)
    return np.stack(out)


def make_chip(site: pd.Series, fires: pd.DataFrame) -> str:
    chip_id = f"{site.site_id}_{site.acq_date}"
    out = CHIPS / f"{chip_id}.npz"
    if out.exists():
        return f"{chip_id}: cached"
    item = find_scene(site.lon, site.lat, site.acq_date)
    if item is None:
        return f"{chip_id}: no clear same-day scene"
    w = read_window(item, site.lon, site.lat)
    if w is None:
        return f"{chip_id}: fire too close to scene edge"
    return save_chip(out, chip_id, item, w, site.site_id, site.acq_date, site.lat, site.lon, fires)


def save_chip(out: Path, chip_id: str, item: dict, w: dict, site_id: str, date: str,
              lat: float, lon: float, fires: pd.DataFrame) -> str:
    """Quality-check a downloaded window, attach nearby VIIRS detections, and save it."""
    swir, scl, rgb, crs, win_tf = w["swir"], w["scl"], w["rgb"], w["crs"], w["transform"]
    nodata = float((scl == 0).mean())
    cloud = float(np.isin(scl, CLOUD_SCL).mean())
    if nodata > MAX_NODATA_CHIP or cloud > MAX_CLOUD_CHIP:
        return f"{chip_id}: rejected (cloud {cloud:.0%}, nodata {nodata:.0%})"

    # VIIRS detections within +-2 days that fall inside the chip.
    s2_time = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
    day = datetime.strptime(date, "%Y-%m-%d")
    near = fires[(fires["latitude"].between(lat - 0.15, lat + 0.15))
                 & (fires["longitude"].between(lon - 0.15, lon + 0.15))]
    near = near[pd.to_datetime(near["acq_date"]).sub(day).abs() <= timedelta(days=2)]
    hx, hy = warp_transform("EPSG:4326", crs, near["longitude"].tolist(), near["latitude"].tolist())
    cols, rows = ~win_tf * (np.array(hx), np.array(hy))
    inside = (rows >= 0) & (rows < CHIP) & (cols >= 0) & (cols < CHIP)
    near, rows, cols = near[inside], rows[inside], cols[inside]
    day_off = (pd.to_datetime(near["acq_date"]) - day).dt.days.to_numpy()
    viirs_t = pd.to_datetime(near["acq_date"] + " " + near["acq_time"].astype(str).str.zfill(4),
                             format="%Y-%m-%d %H%M", utc=True)
    gap_min = ((viirs_t - pd.Timestamp(s2_time)).dt.total_seconds() / 60).to_numpy()

    meta = dict(chip_id=chip_id, site_id=site_id, date=date, scene=item["id"],
                s2_datetime=item["properties"]["datetime"], cloud_scene=item["properties"]["eo:cloud_cover"],
                cloud_chip=cloud, crs=str(crs), transform=list(win_tf)[:6], lat=lat, lon=lon,
                offset_fix=0.0)          # read with the corrected scale_offset: fix_brightness.py must skip it
    np.savez_compressed(out, swir=swir.astype(np.float16), vis=w["vis"].astype(np.float16), rgb=rgb, scl=scl,
                        hot_row=rows.astype(np.float32), hot_col=cols.astype(np.float32),
                        hot_day=day_off.astype(np.int8), hot_frp=near["frp"].to_numpy(np.float32),
                        hot_gap_min=gap_min.astype(np.float32), meta=json.dumps(meta))
    same = int((day_off == 0).sum())
    return f"{chip_id}: saved ({same} same-day detections, cloud {cloud:.0%})"


def load_fires() -> pd.DataFrame:
    """Last week's detections plus the archive ones, if fetched."""
    frames = [pd.read_csv(ROOT / "data" / "firms.csv")]
    if (ROOT / "data" / "firms_archive.csv").exists():
        frames.append(pd.read_csv(ROOT / "data" / "firms_archive.csv"))
    cols = ["latitude", "longitude", "acq_date", "acq_time", "frp"]
    return pd.concat([f[cols] for f in frames], ignore_index=True)


def main() -> None:
    """python fetch_s2.py [target] [sites file in data/]"""
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    sites_file = sys.argv[2] if len(sys.argv) > 2 else "sites.csv"      # e.g. sites_archive.csv
    CHIPS.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(ROOT / "data" / sites_file, dtype={"site_id": str})
    fires = load_fires()
    saved = len([p for p in CHIPS.glob("*.npz") if "_bg" not in p.stem])   # fire chips only
    target += saved if sites_file != "sites.csv" else 0                      # archive: N more chips
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(make_chip, s, fires): s for s in sites.itertuples(index=False)}
        for fut in as_completed(futures):
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futures[fut].site_id}_{futures[fut].acq_date}: error {type(e).__name__}: {e}"
            if "saved" in msg:
                saved += 1
            print(f"[{saved}] {msg}", flush=True)
            if saved >= target:
                print("target reached, cancelling the rest")
                for f in futures:
                    f.cancel()
                break


if __name__ == "__main__":
    main()
