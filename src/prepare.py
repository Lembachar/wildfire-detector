"""Step 3 (v2): pixel labels + training arrays for both the tile model and the pixel (U-Net) model.

Per pixel (20 m):
  1   fire      hot pixel (make_tiles.hot_pixels) with a same-day VIIRS detection within 2 km
  255 ignore    cloud / missing data, or a hot pixel near a fire from another day (uncertain)
  0   not fire  everything else, INCLUDING burned ground next to fires (dropped in v1, which is
                why the first model flagged burn scars) and look-alikes far from any fire

Per tile (32 x 32 px = 640 m), derived from the pixel labels:
  fire if it has a fire pixel; dropped if it has an uncertain hot pixel or > 30% cloud; else not fire.
  The model sees the tile plus 16 px (320 m) of surroundings on every side: a 64 x 64 window.

Inputs: 7 bands in this order: B12, B11, B8A, B4, B3, B2, B8 (reflectance, clipped to [0, 2]).
Output (data/prepared/):
  X.npy      float16 (N, 7, 544, 544)  chips padded by 16 px, memory-mapped during training
  M.npy      uint8   (N, 544, 544)     pixel labels, padding = 255
  tiles.npz  one row per kept tile: chip, tr, tc, y, n_fire (fire pixels), split, background flag
  chips.json chip ids + metadata
"""
import json
from pathlib import Path

import numpy as np

from make_tiles import CLOUD_SCL, hot_pixels, split_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prepared"
CHIP, PAD, TILE = 512, 16, 32
CONFIRM_PX = 100          # 2 km
BLOCK = 8                 # distance computed on an 8 px (160 m) grid: fast and precise enough
MAX_CLOUD_TILE = 0.3
BANDS = ["B12", "B11", "B8A", "B4", "B3", "B2", "B8"]


def near(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pixels within CONFIRM_PX of any of the points."""
    n = CHIP // BLOCK
    if len(rows) == 0:
        return np.zeros((CHIP, CHIP), bool)
    centres = (np.arange(n) + 0.5) * BLOCK
    dr = centres[:, None, None] - rows[None, None, :]
    dc = centres[None, :, None] - cols[None, None, :]
    close = (np.hypot(dr, dc) <= CONFIRM_PX + BLOCK / 2).any(-1)
    return np.repeat(np.repeat(close, BLOCK, 0), BLOCK, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted((ROOT / "data" / "chips").glob("*.npz")) if ".tmp" not in p.name]
    usable = []
    for p in paths:
        try:
            with np.load(p) as z:
                if "vis" in z.files:
                    usable.append(p)
        except Exception as e:
            print(f"skipping {p.name}: {e}")
    print(f"{len(usable)} of {len(paths)} chips have all 7 bands")

    review_file = ROOT / "data" / "review_labels.json"
    reviewed = {}
    if review_file.exists():
        reviewed = {(v["chip"], v["tr"], v["tc"]): v["label"] for v in json.loads(review_file.read_text()).values()}
        print(f"{len(reviewed)} eye-check answers loaded")

    S = CHIP + 2 * PAD
    X = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float16, shape=(len(usable), 7, S, S))
    M = np.lib.format.open_memmap(OUT / "M.npy", mode="w+", dtype=np.uint8, shape=(len(usable), S, S))
    rows_out, metas = [], []
    for ci, p in enumerate(usable):
        with np.load(p) as z:
            swir = np.nan_to_num(z["swir"].astype(np.float32))
            vis = np.nan_to_num(z["vis"].astype(np.float32))
            scl, hr, hc, hd = z["scl"], z["hot_row"], z["hot_col"], z["hot_day"]
            meta = json.loads(str(z["meta"]))
        meta["background"] = "_bg" in p.stem
        metas.append(meta)
        x = np.clip(np.concatenate([swir, vis]), 0, 2)
        hot = hot_pixels(swir)
        near_same = near(hr[hd == 0], hc[hd == 0])
        near_any = near(hr, hc)
        cloud = np.isin(scl, CLOUD_SCL) | (scl == 0)
        uncertain_hot = hot & near_any & ~near_same

        m = np.zeros((CHIP, CHIP), np.uint8)
        m[hot & near_same] = 1
        m[uncertain_hot | cloud] = 255

        # Your eye-check answers override the automatic labels for those tiles.
        overrides = {(tr, tc): lab for (c, tr, tc), lab in reviewed.items() if c == p.stem}
        for (tr, tc), lab in overrides.items():
            sl = (slice(tr * TILE, (tr + 1) * TILE), slice(tc * TILE, (tc + 1) * TILE))
            tm, th = m[sl], hot[sl]
            if lab == "fire":            # hot pixels in the tile are fire; no hot pixel -> pixels unknown
                tm[th] = 1
                if not th.any():
                    tm[~cloud[sl]] = 255
            elif lab == "not":
                tm[th] = 0
            else:                        # not sure: keep it out of training and scoring
                tm[:] = 255

        X[ci] = 0
        X[ci, :, PAD:PAD + CHIP, PAD:PAD + CHIP] = x.astype(np.float16)
        M[ci] = 255
        M[ci, PAD:PAD + CHIP, PAD:PAD + CHIP] = m

        s = split_of(meta["site_id"])
        n = CHIP // TILE
        for tr in range(n):
            for tc in range(n):
                sl = (slice(tr * TILE, (tr + 1) * TILE), slice(tc * TILE, (tc + 1) * TILE))
                if cloud[sl].mean() > MAX_CLOUD_TILE:
                    continue
                n_fire = int((m[sl] == 1).sum())
                lab = overrides.get((tr, tc))
                if lab == "unsure":
                    continue
                if lab == "fire":
                    y = 1
                    n_fire = max(n_fire, 1)
                elif lab == "not":
                    y = 0
                else:
                    if n_fire == 0 and uncertain_hot[sl].any():
                        continue
                    y = int(n_fire > 0)
                rows_out.append((ci, tr, tc, y, n_fire, s, int(meta["background"]), int(hot[sl].sum()),
                                 int(lab is not None)))
        if (ci + 1) % 100 == 0:
            print(f"  {ci + 1}/{len(usable)} chips", flush=True)
    X.flush()
    M.flush()

    t = np.array(rows_out, dtype=np.int32)
    np.savez(OUT / "tiles.npz", chip=t[:, 0], tr=t[:, 1], tc=t[:, 2], y=t[:, 3], n_fire=t[:, 4],
             split=t[:, 5], background=t[:, 6], n_hot=t[:, 7], reviewed=t[:, 8])
    print(f"  tiles labelled by your eye check: {int(t[:, 8].sum())}")
    (OUT / "chips.json").write_text(json.dumps(dict(chips=[p.stem for p in usable], metas=metas, bands=BANDS)))

    y, split, bg, n_hot = t[:, 3], t[:, 5], t[:, 6], t[:, 7]
    print(f"{len(t):,} tiles kept from {len(usable)} chips")
    print(f"  burned-ground / near-fire tiles now labelled 'not fire': "
          f"{int(((y == 0) & (bg == 0) & (n_hot == 0)).sum()):,}")
    print(f"  look-alike tiles (hot pixels, not fire): {int(((y == 0) & (n_hot > 0)).sum()):,}")
    for s, name in enumerate(["train", "val", "test"]):
        k = split == s
        print(f"  {name:5s} {k.sum():7,} tiles  {int(y[k].sum()):6,} fire ({y[k].mean():.1%})")


if __name__ == "__main__":
    main()
