"""Step 3: cut every chip into 32 x 32 pixel tiles (640 m) and label them fire / no fire.

Why labels come from the image AND VIIRS together
  VIIRS passes ~3.4 h after Sentinel-2. In the first version (label = "a VIIRS point is in the tile")
  only 23% of fire tiles showed any fire in the image: the rest were places the fire reached later.
  No model can learn from that. So now:

  hot pixel   a pixel that passes a simple active-fire test (HOT_RULE below): bright in B12 and
              brighter in B12 than in B11 and B8A, which burning ground is and bare soil is not
  fire        the tile has >= 1 hot pixel AND a same-day VIIRS detection within 2 km
              (VIIRS confirms a real fire is burning there; the image shows exactly where)
  no fire     no VIIRS detection from +-2 days within 2 km. These tiles still contain look-alikes
              that trip the hot-pixel test (bright roofs, sun glint, hot bare ground): the hard
              negatives the CNN has to learn to reject
  dropped     everything in between (near a fire but nothing visible), and cloudy / missing tiles

The train / val / test split is by SITE, so the same fire never appears in two splits.
Output: data/tiles.npz
"""
import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TILE = 32
CONFIRM_PX = 100          # 2 km at 20 m: a VIIRS point this close confirms the fire
MAX_CLOUD = 0.3
MAX_NODATA = 0.05
CLOUD_SCL = [8, 9, 10]


def hot_pixels(swir: np.ndarray) -> np.ndarray:
    """Simple active-fire test per pixel. swir = (B12, B11, B8A) reflectance.

    0.25 on correct brightness = the old 0.15 on data that was 0.1 too dark (the brightness bug), so the
    labels and the eye-check answers stay the same."""
    b12, b11, b8a = swir
    return (b12 > 0.25) & (b12 > b11) & (b12 > b8a)


def split_of(site_id: str) -> int:
    """0 = train, 1 = val, 2 = test; stable across runs."""
    h = zlib.crc32(site_id.encode()) % 100
    return 0 if h < 70 else 1 if h < 85 else 2


def rect_dist(r0: int, c0: int, rows: np.ndarray, cols: np.ndarray) -> float:
    """Smallest distance in pixels from points to the tile rectangle (0 if inside)."""
    if len(rows) == 0:
        return np.inf
    dr = np.maximum(np.maximum(r0 - rows, rows - (r0 + TILE)), 0)
    dc = np.maximum(np.maximum(c0 - cols, cols - (c0 + TILE)), 0)
    return float(np.hypot(dr, dc).min())


def main() -> None:
    X, y, split, chip_idx, tile_rc, n_hot = [], [], [], [], [], []
    chips = sorted((ROOT / "data" / "chips").glob("*.npz"))
    metas, kept_chips = [], []
    counts = dict(cloudy=0, ambiguous=0)
    for path in chips:
        try:
            d = np.load(path)
            swir = np.nan_to_num(d["swir"].astype(np.float32), nan=0.0)
        except Exception as e:      # a chip still being written, or corrupt
            print(f"skipping {path.name}: {e}")
            continue
        ci = len(kept_chips)
        kept_chips.append(path.stem)
        meta = json.loads(str(d["meta"]))
        metas.append(meta)
        scl = d["scl"]
        hot = hot_pixels(swir)
        rows, cols, days = d["hot_row"], d["hot_col"], d["hot_day"]
        same = days == 0
        s = split_of(meta["site_id"])
        n = swir.shape[1] // TILE
        for tr in range(n):
            for tc in range(n):
                r0, c0 = tr * TILE, tc * TILE
                tscl = scl[r0:r0 + TILE, c0:c0 + TILE]
                if np.isin(tscl, CLOUD_SCL).mean() > MAX_CLOUD or (tscl == 0).mean() > MAX_NODATA:
                    counts["cloudy"] += 1
                    continue
                nh = int(hot[r0:r0 + TILE, c0:c0 + TILE].sum())
                if nh > 0 and rect_dist(r0, c0, rows[same], cols[same]) <= CONFIRM_PX:
                    label = 1
                elif rect_dist(r0, c0, rows, cols) > CONFIRM_PX:
                    label = 0
                else:
                    counts["ambiguous"] += 1
                    continue
                X.append(swir[:, r0:r0 + TILE, c0:c0 + TILE])
                y.append(label)
                split.append(s)
                chip_idx.append(ci)
                tile_rc.append((tr, tc))
                n_hot.append(nh)

    X = np.stack(X).astype(np.float16)
    y, split, n_hot = np.array(y, np.int8), np.array(split, np.int8), np.array(n_hot, np.int32)
    np.savez_compressed(ROOT / "data" / "tiles.npz", X=X, y=y, split=split,
                        chip_idx=np.array(chip_idx, np.int32), tile_rc=np.array(tile_rc, np.int16),
                        n_hot=n_hot, chips=json.dumps(kept_chips), metas=json.dumps(metas))
    print(f"{len(kept_chips)} chips -> {len(y):,} tiles kept; dropped {counts['cloudy']:,} cloudy, "
          f"{counts['ambiguous']:,} ambiguous")
    print(f"look-alikes: {int(((y == 0) & (n_hot > 0)).sum()):,} no-fire tiles contain hot pixels")
    for s, name in enumerate(["train", "val", "test"]):
        m = split == s
        sites = {metas[c]["site_id"] for c in np.array(chip_idx)[m]}
        print(f"  {name:5s} {m.sum():6,} tiles  {int(y[m].sum()):5,} fire ({y[m].mean():.1%})  {len(sites)} sites")


if __name__ == "__main__":
    main()
