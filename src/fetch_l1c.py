"""Top-of-atmosphere (Level-1C) version of every TEST photo, for testing the published fire methods on the input
they were designed for (Schroeder, Kumar-Roy and Murphy used Landsat-8 TOA reflectance; Hu et al. used
Sentinel-2 L1C TOA).

Source: Google Cloud public Sentinel-2 L1C archive (free, no account), the same satellite pass and tile as our
L2A photo. Only the chip's 10 km window is read from the JPEG2000 files (GDAL range reads).
Saved per chip: data/l1c/<chip>.npz with
  toa      float32 (7, 512, 512)  B01, B02, B03, B04, B8A, B11, B12 TOA reflectance on the chip's 20 m grid
                                  (10 m bands averaged, 60 m B01 nearest), nan = no data
  sun_zen  mean sun zenith angle (deg) of the tile, from MTD_TL.xml
  meta     source product name, processing baseline, offset used
Reflectance = (DN + offset) / 10000, offset -1000 for processing baseline >= 04.00 (ESA rule), else 0.

    python src/fetch_l1c.py
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as win_bounds, from_bounds

from fetch_history import ENV
from train_v2 import DATA, ROOT

OUT = ROOT / "data" / "l1c"
GCS = "https://storage.googleapis.com"
API = f"{GCS}/storage/v1/b/gcp-public-data-sentinel-2/o"
BANDS = ["B01", "B02", "B03", "B04", "B8A", "B11", "B12"]
JP2_ENV = dict(ENV, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2,.xml")
X_L2A = np.load(DATA / "X.npy", mmap_mode="r")
CHIP_INDEX = {n: i for i, n in enumerate(json.loads((DATA / "chips.json").read_text())["chips"])}


def get(url: str, **kw) -> requests.Response:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.ok:
                return r
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


class MissingData(Exception):
    """This product has no data over our window (a tile can be split into two products from the same pass)."""


def find_products(meta: dict) -> list[str]:
    """SAFE prefixes of the L1C products for the same pass and tile, best guess first: closest pass in time,
    then the newest processing baseline. One pass can be split into two products covering different parts
    of the tile, so the caller tries the next one if the first has no data over the window."""
    sat, tile, day = re.match(r"(S2[ABC])_(\d{1,2}[A-Z]{3})_(\d{8})", meta["scene"]).groups()
    tile = tile.zfill(5)
    base = f"tiles/{int(tile[:2])}/{tile[2]}/{tile[3:]}/"
    items = get(API, params=dict(prefix=f"{base}{sat}_MSIL1C_{day}", delimiter="/")).json().get("prefixes", [])
    if not items:
        raise RuntimeError("no L1C product")
    t_now = datetime.fromisoformat(meta["s2_datetime"].replace("Z", "+00:00")).replace(tzinfo=None)

    def key(p):
        sense = datetime.strptime(p.split("_")[2], "%Y%m%dT%H%M%S")
        return abs((sense - t_now).total_seconds()) // 3600, -int(p.split("_")[3][1:])
    return sorted(items, key=key)


def granule(safe: str) -> str:
    """IMG_DATA file prefix of the product's granule."""
    files = get(API, params=dict(prefix=f"{safe}GRANULE/", fields="items(name)")).json()["items"]
    img = [f["name"] for f in files if f["name"].endswith("_B12.jp2")]
    if not img:
        raise RuntimeError("no B12 file")
    return img[0][:-len("B12.jp2")]


def one(name: str, meta: dict) -> str:
    out = OUT / f"{name}.npz"
    if out.exists():
        return "cached"
    bounds = win_bounds(Window(0, 0, 512, 512), rasterio.Affine(*meta["transform"]))
    # Where our L2A photo has data, the L1C photo (same pass) must too. A cut-off JPEG2000 read ("Stream too
    # short") can come back as zeros WITHOUT raising, so a band with missing data there is read again.
    l2a_valid = np.asarray(X_L2A[CHIP_INDEX[name], 0, 16:528, 16:528]) > 0

    def read(band, img, offset):
        rs = Resampling.nearest if band == "B01" else Resampling.average
        n_missing = 0
        for attempt in range(6):
            try:
                with rasterio.Env(**JP2_ENV), rasterio.open(f"/vsicurl/{GCS}/gcp-public-data-sentinel-2/{img}{band}.jp2") as src:
                    if f"EPSG:{src.crs.to_epsg()}" != meta["crs"]:
                        raise RuntimeError("different grid")
                    w = from_bounds(*bounds, transform=src.transform)
                    dn = src.read(1, window=w, out_shape=(512, 512), resampling=rs, boundless=True, fill_value=0)
                missing = ((dn == 0) & l2a_valid).mean()
                if missing > 0.01:
                    n_missing += 1
                    if n_missing >= 3:                         # not a cut-off read: this product misses the window
                        raise MissingData(f"{band}: {missing:.0%} missing where L2A has data")
                    raise rasterio.errors.RasterioIOError(f"{band}: {missing:.0%} missing where L2A has data")
                v = (dn.astype(np.float32) + offset) / 10000.0
                v[dn == 0] = np.nan
                return v
            except rasterio.errors.RasterioIOError:
                if attempt == 5:
                    raise
                time.sleep(4 * (attempt + 1))
    errors = []
    for safe in find_products(meta):
        img = granule(safe)
        baseline = int(safe.split("_")[3][1:])
        offset = -1000.0 if baseline >= 400 else 0.0
        try:
            with ThreadPoolExecutor(len(BANDS)) as pool:
                toa = np.stack(list(pool.map(lambda b: read(b, img, offset), BANDS)))
            break
        except MissingData as e:
            errors.append(f"{safe.split('/')[-2][:60]}: {e}")
    else:
        raise RuntimeError("no product covers the window: " + " | ".join(errors))
    tl = get(f"{GCS}/gcp-public-data-sentinel-2/{img.split('IMG_DATA')[0]}MTD_TL.xml").text
    zen = float(re.search(r"<Mean_Sun_Angle>\s*<ZENITH_ANGLE[^>]*>([\d.]+)<", tl).group(1))
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, toa=toa, sun_zen=zen,
                        meta=json.dumps(dict(product=safe.rstrip("/").split("/")[-1], baseline=baseline, offset=offset)))
    tmp.replace(out)
    return f"{safe.split('/')[-2][:60]} zen {zen:.1f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.load(DATA / "tiles.npz")
    info = json.loads((DATA / "chips.json").read_text())
    test = sorted(set(np.unique(t["chip"][t["split"] == 2]).tolist()))
    t0, failed = time.time(), []
    with ThreadPoolExecutor(3) as pool:                       # 4 x 7 parallel reads cut connections off
        futs = {pool.submit(one, info["chips"][c], info["metas"][c]): info["chips"][c] for c in test}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:
                msg = f"FAILED {type(e).__name__}: {e}"
                failed.append(futs[f])
            print(f"[{i}/{len(test)} {time.time() - t0:5.0f}s] {futs[f]}: {msg}", flush=True)
    print(f"done; failed {len(failed)}: {failed}")


if __name__ == "__main__":
    main()
