"""Live fire check for any place: find the latest clear Sentinel-2 photo, download a 10 km square,
run the three models (combined), and return pictures + detections.

Speed: the 8 image layers are downloaded at the same time (threads); models stay loaded on the GPU.
"""
import base64
import io
import json
import sys
import threading
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
import torch
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fetch_s2 import CLOUD_SCL, GDAL_ENV, STAC, scale_offset  # noqa: E402
from train_v2 import DEVICE, PreNet, TileNet, UNet  # noqa: E402
import fetch_history as FH  # noqa: E402  same earlier-photo code that built the training data

CHIP, PAD, TILE, WIN = 512, 16, 32, 64
PIXEL_HA = 0.04                      # one 20 m pixel = 400 m2
LOOKBACK_DAYS = 15
EDGE = 128                           # px (2.56 km): closer than this to a block side -> centre the square
HISTORY_MAX = 4                      # earlier photos compared against ("was it already there?")
UNCLEAR_SCL = [0, 1, 3, 8, 9, 10]    # no data, defective, cloud shadow, clouds, cirrus: don't trust B12 there
HEAT_RISE = 0.3                      # B12 - B11 must rise this much vs before
# OFF: scored on val/test (src/eval_history.py) the rule also drops real fires that were already burning in
# the earlier photo (recall 86% -> 57% at 0.3). Kept for the before+now model, which learns this instead.
HISTORY_CHECK = False
SWIR_KEYS = ["swir22", "swir16", "nir08"]     # B12, B11, B8A (20 m)
VIS_KEYS = ["red", "green", "blue", "nir"]    # B4, B3, B2, B8 (10 m, averaged to 20 m)
ENSEMBLE = "after avg tile(0.1)+unet(0.2)+pretrained(0.7)"
WEIGHTS = {"tile": 0.1, "unet": 0.2, "pretrained": 0.7}
FIRMS = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/1/{day}"
FIRMS_SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"]
# Day AND night watch: NASA's thermal satellites see fires at night too (Sentinel-2 only works in daylight).
# VIIRS 375 m on 3 satellites + MODIS 1 km, about 6 passes a day, alerts within ~3 h of the pass.
RECENT_HOURS = 48
RECENT_KM = 7                        # half side of the box around the click (the square is ~10 km)
RECENT_FIRMS = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{day}"
RECENT_NRT = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"]
RECENT_SP = ["VIIRS_NOAA20_SP", "VIIRS_SNPP_SP", "MODIS_SP"]          # older dates: the archive versions
SAT_NAMES = {"N20": "NOAA-20 VIIRS", "N21": "NOAA-21 VIIRS", "N": "Suomi NPP VIIRS",
             "Terra": "Terra MODIS", "Aqua": "Aqua MODIS", "T": "Terra MODIS", "A": "Aqua MODIS"}
# Fewer round trips per file: read more header bytes at open, merge ranges, reuse HTTP/2 connections.
FAST_ENV = dict(GDAL_ENV, GDAL_INGESTED_BYTES_AT_OPEN="32768", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2")
# GDAL ignores the Windows system proxy (VPN apps set it); without it downloads can be ~25x slower.
_proxy = urllib.request.getproxies().get("https") or urllib.request.getproxies().get("http")
if _proxy:
    FAST_ENV.update(GDAL_HTTP_PROXY=_proxy.split("://")[-1].rstrip("/"))


class Detector:
    def __init__(self) -> None:
        self.models, self.norm = {}, {}
        for kind, cls in [("tile", TileNet), ("unet", UNet), ("pretrained", PreNet)]:
            ck = torch.load(ROOT / "outputs" / f"model_{kind}.pt", map_location=DEVICE, weights_only=False)
            m = cls().to(DEVICE).eval()
            m.load_state_dict(ck["state"])
            self.models[kind] = m
            self.norm[kind] = (ck["mean"].to(DEVICE), ck["std"].to(DEVICE))
        ens = ROOT / "outputs" / "ensemble.json"                # written by src/compare_models.py
        if ens.exists():
            cfg = json.loads(ens.read_text())
            self.weights, self.threshold = cfg["weights"], cfg["threshold"]
        else:
            self.weights = WEIGHTS
            self.threshold = json.loads((ROOT / "outputs" / "compare_v4.json").read_text())[ENSEMBLE]["threshold"]
        # Before + now models (10 bands: the 7 of the photo + B12 before, B11 before, seen before).
        # Used for every check when present; the photo-only models above are the fallback if they're missing.
        self.hmodels, self.hnorm = {}, {}
        hens = ROOT / "outputs" / "ensemble_hist.json"
        if hens.exists() and all((ROOT / "outputs" / f"model_{k}_hist.pt").exists() for k in self.models):
            cfg = json.loads(hens.read_text())
            self.hweights = {k: w for k, w in cfg["weights"].items() if w > 0}
            self.hthreshold = cfg["threshold"]
            for kind, cls in [("tile", TileNet), ("unet", UNet), ("pretrained", PreNet)]:
                if kind not in self.hweights and kind != "unet":         # weight 0: not needed
                    continue
                ck = torch.load(ROOT / "outputs" / f"model_{kind}_hist.pt", map_location=DEVICE, weights_only=False)
                m = cls(ck["bands"]).to(DEVICE).eval()
                m.load_state_dict(ck["state"])
                self.hmodels[kind] = m
                self.hnorm[kind] = (ck["mean"].to(DEVICE), ck["std"].to(DEVICE))
        self.gpu = threading.Lock()
        self.cache: dict = {}
        key_file = ROOT / "firms_key.txt"
        self.firms_key = key_file.read_text().strip() if key_file.exists() else None
        self.http = requests.Session()
        self.bg = ThreadPoolExecutor(8)                         # NASA day/night lookups, alongside the photo
        self.analyze(None, None, warmup=True)

    # ------------------------------------------------------------ satellite photo
    def find_items(self, lon: float, lat: float, until: date) -> list[dict]:
        start = until - timedelta(days=LOOKBACK_DAYS)
        body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{start}T00:00:00Z/{until}T23:59:59Z", "limit": 30,
                "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
        for attempt in range(3):
            try:
                feats = self.http.post(STAC, json=body, timeout=30).json().get("features", [])
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1)
        feats = [f for f in feats if f["properties"].get("eo:cloud_cover", 100) <= 95]
        return sorted(feats, key=lambda f: f["properties"]["datetime"], reverse=True)

    @staticmethod
    def geometry(item: dict, lon: float, lat: float):
        """Pixel window (20 m grid) of the 10 km square centred on lon/lat, without an extra download."""
        a = item["assets"]["swir22"]
        epsg = item["properties"].get("proj:epsg") or a.get("proj:epsg")
        tf, shape = a.get("proj:transform"), a.get("proj:shape")
        if not (epsg and tf and shape):
            with rasterio.Env(**GDAL_ENV), rasterio.open(a["href"]) as src:
                crs, transform, shape = src.crs, src.transform, (src.height, src.width)
        else:
            crs, transform = rasterio.crs.CRS.from_epsg(epsg), rasterio.Affine(*tf[:6])
        (x,), (y,) = warp_transform("EPSG:4326", crs, [lon], [lat])
        row, col = rasterio.transform.rowcol(transform, x, y)
        # Snap to the file's own 512 px (10.24 km) blocks: one block per layer instead of up to four,
        # so ~4x less to download. But if the click is within EDGE px of a block side, centre the square
        # on it along that axis (2 blocks on that axis: slower), so the spot isn't cut off at the corner.
        def start(v: int) -> int:
            s = (v // CHIP) * CHIP
            return v - CHIP // 2 if (v - s < EDGE or s + CHIP - v <= EDGE) else s
        r0, c0 = start(row), start(col)
        inside = r0 >= 0 and c0 >= 0 and r0 + CHIP <= shape[0] and c0 + CHIP <= shape[1]
        win = Window(c0, r0, CHIP, CHIP)
        return crs, win, rasterio.windows.transform(win, transform), rasterio.windows.bounds(win, transform), inside

    @staticmethod
    def read(asset: dict, win=None, bounds=None, scl=False, props=None) -> np.ndarray:
        with rasterio.Env(**FAST_ENV), rasterio.open(asset["href"]) as src:
            if bounds is not None:            # 10 m band: average down to the 20 m grid
                w = from_bounds(*bounds, transform=src.transform)
                dn = src.read(1, window=w, out_shape=(CHIP, CHIP), resampling=Resampling.average, boundless=True)
            else:
                dn = src.read(1, window=win, boundless=True)
        if scl:
            return dn
        s, o = scale_offset(asset, props)
        refl = dn.astype(np.float32) * s + o
        refl[dn == 0] = np.nan
        return refl

    def photo(self, lon: float, lat: float, until: date):
        items = self.find_items(lon, lat, until)
        if not items:
            return None, f"No Sentinel-2 photo of this place in the {LOOKBACK_DAYS} days up to {until}."
        # Prefer photos that cover the whole square; check clouds on our square only (cheap: 1 layer).
        cands = []
        for it in items:
            g = self.geometry(it, lon, lat)
            cands.append((not g[4], it, g))
        cands.sort(key=lambda c: c[0])                        # stable: newest whole-square photos first
        # Earlier photos for the before+now model: start now, alongside the main download (bet on the newest).
        pre = {cands[0][1]["id"]: self.bg.submit(self.before, cands[0][1], cands[0][2][2], lon, lat)} \
            if self.hmodels else {}

        def layers(pool, it, g):                               # all 8 layers of one photo at once
            a, pr = it["assets"], it["properties"]
            return ([pool.submit(self.read, a[k], win=g[1], props=pr) for k in SWIR_KEYS],
                    [pool.submit(self.read, a[k], bounds=g[3], props=pr) for k in VIS_KEYS])

        def clear(scl):
            return float(np.isin(scl, CLOUD_SCL).mean()), float((scl == 0).mean())

        with ThreadPoolExecutor(9) as pool:
            # Bet on the newest photo: download its cloud map and all layers together.
            _, it, g = cands[0]
            scl_f, (sw, vi) = pool.submit(self.read, it["assets"]["scl"], win=g[1], scl=True), layers(pool, it, g)
            scl = scl_f.result()
            cloud, nodata = clear(scl)
            best = (it, g, scl, cloud, nodata, sw, vi)
            if not (cloud <= 0.3 and nodata <= 0.2):           # too cloudy: check older photos' cloud maps
                for _, it2, g2 in cands[1:5]:
                    scl2 = self.read(it2["assets"]["scl"], win=g2[1], scl=True)
                    c2, n2 = clear(scl2)
                    if c2 + n2 < best[3] + best[4]:
                        best = (it2, g2, scl2, c2, n2, None, None)
                    if c2 <= 0.3 and n2 <= 0.2:
                        break
            it, g, scl, cloud, nodata, sw, vi = best
            if cloud + nodata > 0.85:
                return None, f"Every photo of this place in the last {LOOKBACK_DAYS} days is covered by cloud."
            if sw is None:
                sw, vi = layers(pool, it, g)
            swir, vis = np.stack([f.result() for f in sw]), np.stack([f.result() for f in vi])
        crs, _, win_tf, _, _ = g
        if self.hmodels and it["id"] not in pre:              # an older photo was chosen: its own "before"
            pre[it["id"]] = self.bg.submit(self.before, it, win_tf, lon, lat)
        return dict(item=it, crs=crs, transform=win_tf, win=g[1], swir=swir, vis=vis, scl=scl, cloud=cloud,
                    nodata=nodata, before=pre.get(it["id"])), None

    # ------------------------------------------------------------ "before" picture for the before+now model
    def before(self, item: dict, win_tf, lon: float, lat: float):
        """Median B12, B11 over up to 3 clear earlier photos of exactly these pixels (15 days before this one).

        Same steps as src/fetch_history.py, which built the training data: same search, same choice of photos
        (the first 3 on the same pixel grid, then any fully cloudy one dropped), same cloud mask, same median.
        Only difference: all layers are downloaded at once instead of one by one (same result, faster).
        Returns (before (2, 512, 512) with nan where never seen clearly, dates used).
        """
        a = item["assets"]["swir22"]
        epsg = item["properties"].get("proj:epsg") or a.get("proj:epsg")
        meta = dict(s2_datetime=item["properties"]["datetime"], lon=lon, lat=lat, crs=f"EPSG:{epsg}",
                    transform=list(tuple(win_tf)[:6]))
        photos, seen = [], set()
        for f in FH.earlier_items(meta):
            key = f["properties"]["datetime"][:13]
            if key in seen or f["properties"].get("eo:cloud_cover", 100) > 95:
                continue
            w = FH.window_in(f, meta)
            if w is not None:
                seen.add(key)
                photos.append((f, w))
            if len(photos) == FH.MAX_PHOTOS:
                break
        with ThreadPoolExecutor(max(1, 3 * len(photos))) as pool:
            futs = [(f, pool.submit(FH.read, f["assets"]["scl"], w, None, True),
                     [pool.submit(FH.read, f["assets"][k], w, f["properties"]) for k in FH.KEYS]) for f, w in photos]
            stack, dates = [], []
            for f, fs, fb in futs:
                bad = np.isin(fs.result(), FH.UNCLEAR_SCL)
                if bad.mean() > 0.97:                          # fully cloudy over the square: not used
                    continue
                bands = np.stack([x.result() for x in fb])
                bands[:, bad] = np.nan
                stack.append(bands)
                dates.append(f["properties"]["datetime"])
        if not stack:
            return np.full((2, CHIP, CHIP), np.nan, np.float32), []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bef = np.nanmedian(np.stack(stack), axis=0)
        return bef.astype(np.float16).astype(np.float32), dates     # stored as float16 in training too

    @staticmethod
    def history_bands(swir: np.ndarray, bef: np.ndarray) -> np.ndarray:
        """The 3 extra bands exactly as src/prepare_history.py makes them: B12/B11 before where a clear earlier
        photo saw the pixel (else the current value), and 1/0 "seen before"."""
        now = np.clip(np.nan_to_num(swir[:2]), 0, 2).astype(np.float16).astype(np.float32)
        ok = np.isfinite(bef).all(axis=0)
        h = np.empty((3, CHIP, CHIP), np.float32)
        h[:2] = np.where(ok, bef, now)
        h[2] = ok
        return h.astype(np.float16).astype(np.float32)

    # ------------------------------------------------------------ "was it already there?"
    def earlier(self, item: dict, lon: float, lat: float, win_tf) -> list[dict]:
        """Photos of the same tile (same pixel grid) from the 15 days before this one, newest first."""
        t = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{(t - timedelta(days=LOOKBACK_DAYS)).strftime(fmt)}/{(t - timedelta(hours=1)).strftime(fmt)}",
                "limit": 30, "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
        feats = self.http.post(STAC, json=body, timeout=30).json().get("features", [])
        out, seen = [], set()
        for f in feats:
            day = f["properties"]["datetime"][:13]                 # same pass listed twice (reprocessing)
            if f["properties"].get("eo:cloud_cover", 100) > 95 or day in seen:
                continue
            g = self.geometry(f, lon, lat)
            if g[4] and np.allclose(tuple(g[2])[:6], tuple(win_tf)[:6]):   # exactly the same pixels
                seen.add(day)
                out.append(f)
        return out[:HISTORY_MAX]

    def history(self, p: dict, lon: float, lat: float, groups: list) -> tuple[list, list]:
        """For each detection: is the heat NEW, or did the same pixels look the same in earlier clear photos?

        Roofs, factories, flares and hot rocks look the same every day; a fire isn't there before it starts.
        The comparison uses the heat signature B12 - B11, not brightness: cars, containers, building work and
        sun glints change brightness between photos but raise B11 about as much as B12, flames don't.
        Per pixel, "before" is the median B12 - B11 over the earlier photos where that pixel was clear (no
        cloud, shadow or missing data), widened by 1 px for small misalignment between photos. A pixel is new
        heat if it now looks like heat and B12 - B11 rose by HEAT_RISE. Returns a status per detection and
        the dates of the photos compared.
        """
        try:
            items = self.earlier(p["item"], lon, lat, p["transform"])
        except (requests.RequestException, ValueError):
            items = []
        if not items:
            return [dict(status="unknown", before=[]) for _ in groups], []
        with ThreadPoolExecutor(3 * len(items)) as pool:
            futs = [(it, pool.submit(self.read, it["assets"]["scl"], win=p["win"], scl=True),
                     [pool.submit(self.read, it["assets"][k], win=p["win"], props=it["properties"])
                      for k in ("swir22", "swir16")]) for it in items]
            stack, dates = [], []
            for it, fs, fb in futs:
                try:
                    scl, b12, b11 = fs.result(), fb[0].result(), fb[1].result()
                except Exception:                                  # one failed download: compare without it
                    continue
                heat = b12 - b11                                   # heat signature: flames make B12 >> B11
                heat[np.isin(scl, UNCLEAR_SCL)] = np.nan
                stack.append(heat)
                dates.append(it["properties"]["datetime"][:10])
        if not stack:
            return [dict(status="unknown", before=[]) for _ in groups], []
        with warnings.catch_warnings():                        # all-cloud pixels: median of nothing -> nan
            warnings.simplefilter("ignore")
            base = np.nanmedian(np.stack(stack), axis=0)
        padded = np.pad(base, 1, constant_values=np.nan)
        base3 = np.full_like(base, np.nan)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                base3 = np.fmax(base3, padded[dr:dr + CHIP, dc:dc + CHIP])
        b12, b11, b8a = np.nan_to_num(p["swir"])
        heat_like = (b12 > b11) & (b12 > b8a) & (b12 >= 0.12)
        known = np.isfinite(base3)
        rise = known & ((b12 - b11) - np.nan_to_num(base3) >= HEAT_RISE)
        stack = np.stack(stack)
        out = []
        for region, fire in groups:
            cand = region & heat_like
            if not cand.any():
                cand = fire
            new = cand & rise
            # The detection's strongest pixels must have a clear "before" to call it not new.
            used = [d for d, s in zip(dates, stack) if np.isfinite(s[fire]).any()]
            if new.any():
                status = "new"
            elif known[fire].all():
                status = "not_new"
            else:
                status = "unknown"
            out.append(dict(status=status, before=used))
        return out, dates

    # ------------------------------------------------------------ models
    @torch.no_grad()
    def run_models(self, swir: np.ndarray, vis: np.ndarray, hist: np.ndarray | None = None):
        """Score per 32 px tile and pixel map. With hist (3 "before" bands) the before+now models are used.
        Returns (score, pix, threshold)."""
        x = np.clip(np.nan_to_num(np.concatenate([swir, vis])), 0, 2).astype(np.float16).astype(np.float32)
        if hist is not None:
            x = np.concatenate([x, hist])
            models, norm, weights, thr = self.hmodels, self.hnorm, self.hweights, self.hthreshold
        else:
            models, norm, weights, thr = self.models, self.norm, self.weights, self.threshold
        nb = x.shape[0]
        xp = np.zeros((nb, CHIP + 2 * PAD, CHIP + 2 * PAD), np.float32)
        xp[:, PAD:PAD + CHIP, PAD:PAD + CHIP] = x
        n = CHIP // TILE
        with self.gpu:
            t = torch.from_numpy(xp).to(DEVICE)
            crops = t.unfold(1, WIN, TILE).unfold(2, WIN, TILE)          # (bands, 16, 16, 64, 64)
            crops = crops.permute(1, 2, 0, 3, 4).reshape(n * n, nb, WIN, WIN).contiguous()
            out, pix = {}, None
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                for kind, m in models.items():
                    mean, std = norm[kind]
                    inp = crops if kind == "pretrained" else (crops - mean) / std
                    p = torch.sigmoid(m(inp).float())
                    if kind == "unet":
                        centre = p[:, 16:48, 16:48]
                        pix = centre.reshape(n, n, TILE, TILE).permute(0, 2, 1, 3).reshape(CHIP, CHIP).cpu().numpy()
                        p = centre.amax(dim=(1, 2))
                    out[kind] = p.reshape(n, n).cpu().numpy()
        score = sum(weights[k] * out[k] for k in weights)
        return score, pix, thr

    # ------------------------------------------------------------ main entry
    def analyze(self, lon, lat, when: date | None = None, warmup: bool = False) -> dict:
        if warmup:                                             # first GPU run is slow: do it at start-up
            z3, z4 = np.zeros((3, CHIP, CHIP), np.float32), np.zeros((4, CHIP, CHIP), np.float32)
            self.run_models(z3, z4)
            if self.hmodels:
                self.run_models(z3, z4, z3)
            return {}
        t0 = time.time()
        until = when or date.today()
        recent_f = self.bg.submit(self.recent, lon, lat, when)   # NASA day + night, fetched meanwhile
        p, err = self.photo(lon, lat, until)
        t_photo = time.time() - t0
        if err:                                                # no usable photo: NASA may still have seen heat
            return dict(ok=False, message=err, recent=recent_f.result())
        # Before+now: wait for the earlier photos (downloaded meanwhile). If that download fails, the model gets
        # "nothing seen before", the same input it was trained on for photos without a usable earlier photo.
        t_wait, compared, mode = time.time(), [], "photo only"
        hist = None
        if p["before"] is not None:
            try:
                bef, used = p["before"].result()
            except Exception:
                bef, used = np.full((2, CHIP, CHIP), np.nan, np.float32), []
            hist = self.history_bands(p["swir"], bef)
            compared = [u[:10] for u in used]
            mode = "before+now"
        t_wait = time.time() - t_wait
        score, pix, threshold = self.run_models(p["swir"], p["vis"], hist)
        it = p["item"]
        swir, scl = np.nan_to_num(p["swir"]), p["scl"]
        b12, b11, b8a = swir
        hot = (b12 > 0.25) & (b12 > b11) & (b12 > b8a)                 # same test as the training labels
        flame = (b12 >= 0.3) & (b12 >= 1.6 * b8a) & (b12 > b11)         # strict: real flames only
        blind = np.isin(scl, CLOUD_SCL) | (scl == 0)
        n = CHIP // TILE
        tile_blind = blind.reshape(n, TILE, n, TILE).mean(axis=(1, 3))
        tile_flame = flame.reshape(n, TILE, n, TILE).any(axis=(1, 3))
        # Cloud edges are a known look-alike: next to cloud (any cloud in the tile or the 16 px around it)
        # a tile only counts as fire if it also has a pixel passing the strict flame test.
        cloud = np.pad(np.isin(scl, CLOUD_SCL), 16)
        near_cloud = np.array([[cloud[r * TILE:r * TILE + 64, c * TILE:c * TILE + 64].mean() > 0.01
                                for c in range(n)] for r in range(n)])
        flagged = (score >= threshold) & (tile_blind < 0.5) & (~near_cloud | tile_flame)

        # Group touching flagged tiles into one detection each.
        seen, dets, groups = np.zeros_like(flagged), [], []
        for r0, c0 in zip(*np.nonzero(flagged)):
            if seen[r0, c0]:
                continue
            stack, tiles = [(r0, c0)], []
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                tiles.append((int(r), int(c)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < n and 0 <= cc < n and flagged[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
            region = np.zeros((CHIP, CHIP), bool)
            for r, c in tiles:
                region[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = True
            # Burning pixels (for the outline and the size): a flame test on true brightness. The pixel
            # model and the loose hot-pixel rule also mark fresh burn scars, which inflates the size.
            fire = region & flame
            if not fire.any():
                fire = region & hot
            if not fire.any():                                # model is sure but no pixel stands out
                fire = region & (pix >= pix[region].max() - 1e-6)
            groups.append((region, fire))
            rows, cols = np.nonzero(fire)
            xs, ys = rasterio.transform.xy(p["transform"], rows.mean(), cols.mean())
            (lo,), (la,) = warp_transform(p["crs"], "EPSG:4326", [xs], [ys])
            dets.append(dict(confidence=round(float(max(score[r, c] for r, c in tiles)), 3),
                             hectares=round(float(fire.sum() * PIXEL_HA), 2), pixels=int(fire.sum()),
                             lat=round(la, 5), lon=round(lo, 5), tiles=tiles,
                             bbox=[int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())]))
        # Compare with earlier photos (only when something was flagged, so fire-free places stay fast).
        t_hist = time.time()
        for d in dets:
            d["history"] = dict(status="unknown", before=[])
        if dets and HISTORY_CHECK:
            rule, _ = self.history(p, lon, lat, groups)
            for d, h in zip(dets, rule):
                d["history"] = h
        fire_mask = np.zeros((CHIP, CHIP), bool)
        for d, (_, fire) in zip(dets, groups):
            if d["history"]["status"] != "not_new":
                fire_mask |= fire
        ignored = sorted([d for d in dets if d["history"]["status"] == "not_new"], key=lambda d: -d["confidence"])
        dets = sorted([d for d in dets if d["history"]["status"] != "not_new"], key=lambda d: -d["confidence"])
        t_hist = time.time() - t_hist

        corners = [(0, 0), (0, CHIP), (CHIP, CHIP), (CHIP, 0)]
        cx, cy = zip(*[rasterio.transform.xy(p["transform"], r, c, offset="ul") for r, c in corners])
        clon, clat = warp_transform(p["crs"], "EPSG:4326", list(cx), list(cy))
        taken = datetime.fromisoformat(it["properties"]["datetime"].replace("Z", "+00:00"))
        recent = recent_f.result()
        for pt in recent.get("points", []):                    # where it falls on the photo, if inside
            (x,), (y,) = warp_transform("EPSG:4326", p["crs"], [pt["lon"]], [pt["lat"]])
            c, r = ~p["transform"] * (x, y)
            pt["inside"] = bool(0 <= r < CHIP and 0 <= c < CHIP)
            pt["row"], pt["col"] = round(r, 1), round(c, 1)
        rid = f"{it['id']}_{lon:.4f}_{lat:.4f}"
        self.cache[rid] = dict(crs=p["crs"], transform=p["transform"], day=taken.date().isoformat(),
                               bbox=(min(clon), min(clat), max(clon), max(clat)), taken=taken)
        return dict(ok=True, id=rid, scene=it["id"], taken=it["properties"]["datetime"],
                    days_old=(date.today() - taken.date()).days, cloud=round(p["cloud"], 3),
                    nodata=round(p["nodata"], 3), corners=[[a, b] for a, b in zip(clat, clon)],
                    true_colour=jpeg(true_colour(p["vis"])), infrared=jpeg(infrared(p["swir"])),
                    fire_mask=png_mask(outline(fire_mask)), detections=dets, ignored=ignored, compared=compared,
                    recent=recent, mode=mode,
                    threshold=round(threshold, 3),
                    seconds=dict(photo=round(t_photo, 1), before_wait=round(t_wait, 1), history=round(t_hist, 1),
                                 total=round(time.time() - t0, 1)))

    # ------------------------------------------------------------ NASA day + night watch
    def recent(self, lon: float, lat: float, when: date | None) -> dict:
        """NASA thermal detections (VIIRS + MODIS, day AND night) around the point in the RECENT_HOURS up to
        now (or up to the end of the chosen date). Independent of the Sentinel-2 photo, which can be days old."""
        if not self.firms_key:
            return dict(ok=False, message="No NASA FIRMS key on the server.")
        now = datetime.now(timezone.utc)
        end = now if when is None else min(now, datetime(when.year, when.month, when.day, 23, 59, tzinfo=timezone.utc))
        start = end - timedelta(hours=RECENT_HOURS)
        dlat = RECENT_KM / 111.0
        dlon = RECENT_KM / (111.0 * max(np.cos(np.radians(lat)), 0.1))
        area = f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}"
        days = (end.date() - start.date()).days + 1
        sources = RECENT_NRT + (RECENT_SP if (now - end).days > 2 else [])

        def get(src):
            try:
                r = self.http.get(RECENT_FIRMS.format(key=self.firms_key, source=src, area=area, days=days,
                                                      day=start.date().isoformat()), timeout=12)
                return src, (r.text if r.ok and r.text.startswith("latitude") else "")
            except requests.RequestException:
                return src, None
        with ThreadPoolExecutor(len(sources)) as pool:
            texts = list(pool.map(get, sources))
        pts, seen = [], set()
        for src, text in texts:
            for line in (text or "").strip().splitlines()[1:]:
                v = dict(zip(text.splitlines()[0].split(","), line.split(",")))
                t = datetime.strptime(v["acq_date"] + v["acq_time"].zfill(4), "%Y-%m-%d%H%M").replace(tzinfo=timezone.utc)
                key = (round(float(v["latitude"]), 3), round(float(v["longitude"]), 3), v["acq_date"], v["acq_time"])
                if not (start <= t <= end) or key in seen:
                    continue
                seen.add(key)
                pts.append(dict(lat=float(v["latitude"]), lon=float(v["longitude"]), time=t.isoformat(),
                                hours_ago=round((end - t).total_seconds() / 3600, 1), night=v.get("daynight") == "N",
                                satellite=SAT_NAMES.get(v.get("satellite", ""), src.split("_")[0]),
                                frp=float(v["frp"]) if v.get("frp") else None, confidence=v.get("confidence", "")))
        pts.sort(key=lambda q: q["hours_ago"])
        failed = [s for s, t in texts if t is None]
        return dict(ok=True, points=pts, hours=RECENT_HOURS, until=end.isoformat(), failed=failed)

    # ------------------------------------------------------------ NASA detections on the photo's day
    def nasa(self, rid: str) -> dict:
        g = self.cache.get(rid)
        if g is None:
            return dict(ok=False, message="Analyse a place first.")
        if not self.firms_key:
            return dict(ok=False, message="No NASA FIRMS key on the server.")
        area = ",".join(f"{v:.4f}" for v in g["bbox"])

        def get(src):
            try:
                r = self.http.get(FIRMS.format(key=self.firms_key, source=src, area=area, day=g["day"]), timeout=30)
                return r.text if r.ok and r.text.startswith("latitude") else ""
            except requests.RequestException:
                return ""
        with ThreadPoolExecutor(len(FIRMS_SOURCES)) as pool:
            texts = list(pool.map(get, FIRMS_SOURCES))
        pts, seen = [], set()
        for text in texts:
            lines = text.strip().splitlines()
            if len(lines) < 2:
                continue
            head = lines[0].split(",")
            for line in lines[1:]:
                v = dict(zip(head, line.split(",")))
                k = (v["latitude"], v["longitude"], v["acq_time"])
                if k in seen:
                    continue
                seen.add(k)
                la, lo = float(v["latitude"]), float(v["longitude"])
                (x,), (y,) = warp_transform("EPSG:4326", g["crs"], [lo], [la])
                c, r = ~g["transform"] * (x, y)
                if not (0 <= r < CHIP and 0 <= c < CHIP):
                    continue
                t = datetime.strptime(v["acq_date"] + v["acq_time"].zfill(4), "%Y-%m-%d%H%M").replace(
                    tzinfo=g["taken"].tzinfo)
                pts.append(dict(row=round(r, 1), col=round(c, 1), lat=la, lon=lo, time=v["acq_time"].zfill(4),
                                hours_after=round((t - g["taken"]).total_seconds() / 3600, 1)))
        return dict(ok=True, points=pts, day=g["day"])


# ---------------------------------------------------------------- pictures
def true_colour(vis: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(vis[:3]).transpose(1, 2, 0)
    return (np.clip(rgb / 0.3, 0, 1) ** (1 / 1.5) * 255).astype(np.uint8)


def infrared(swir: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(swir).transpose(1, 2, 0)                 # B12, B11, B8A as red, green, blue
    return (np.clip(v / 0.5, 0, 1) ** 0.8 * 255).astype(np.uint8)


def jpeg(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def outline(m: np.ndarray) -> np.ndarray:
    """1 px ring just outside the burning pixels, so the glow itself stays visible."""
    p = np.pad(m, 1)
    h, w = m.shape
    grown = np.zeros_like(m)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            grown |= p[dr:dr + h, dc:dc + w]
    return grown & ~m


def png_mask(m: np.ndarray) -> str:
    rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[m] = (255, 230, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
