"""Add the visible + near-infrared bands (B4, B3, B2, B8) to chips downloaded before they were saved.

Uses each chip's stored scene id and window, so no STAC search is needed: one item lookup per scene,
then a 10 km read per band. Chips that already have `vis` are skipped.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio

from fetch_background import get_item
from fetch_s2 import CHIP, CHIPS, GDAL_ENV, read_visible

_items: dict = {}


def add(path) -> str:
    with np.load(path) as z:          # close the file before replacing it (Windows locks open files)
        d = dict(z)
    if "vis" in d:
        return f"{path.stem}: already has vis"
    meta = json.loads(str(d["meta"]))
    a, b, c, dd, e, f = meta["transform"]
    bounds = (c, f + e * CHIP, c + a * CHIP, f)          # left, bottom, right, top
    item = _items.get(meta["scene"]) or _items.setdefault(meta["scene"], get_item(meta["scene"]))
    with rasterio.Env(**GDAL_ENV):
        vis = read_visible(item["assets"], bounds, CHIP, item["properties"])
    d["vis"] = vis.astype(np.float16)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(path)
    return f"{path.stem}: added"


def main() -> None:
    todo = sorted(CHIPS.glob("*.npz"))
    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(add, p): p for p in todo}
        for fut in as_completed(futures):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futures[fut].stem}: error {type(e).__name__}: {e}"
            print(f"[{done}/{len(todo)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
