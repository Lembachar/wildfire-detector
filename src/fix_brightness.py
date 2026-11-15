"""One-off repair: saved chips were made 0.1 too dark where Earth Search had already applied the
Sentinel-2 offset (earthsearch:boa_offset_applied) and we subtracted it again. Adds it back to swir and vis.

Safe to re-run: each repaired chip records meta["offset_fix"] and is skipped next time.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from fetch_background import get_item
from fetch_s2 import CHIPS

_items: dict = {}
_lock = threading.Lock()


def item_for(scene: str) -> dict:
    with _lock:
        if scene in _items:
            return _items[scene]
    it = get_item(scene)
    with _lock:
        _items[scene] = it
    return it


def fix(path) -> str:
    with np.load(path) as z:
        d = dict(z)
    meta = json.loads(str(d["meta"]))
    if "offset_fix" in meta:
        return "skip"
    it = item_for(meta["scene"])
    band = (it["assets"]["swir22"].get("raster:bands") or [{}])[0]
    delta = -band.get("offset", 0.0) if it["properties"].get("earthsearch:boa_offset_applied") else 0.0
    if delta:
        d["swir"] = (d["swir"].astype(np.float32) + delta).astype(np.float16)
        d["vis"] = (d["vis"].astype(np.float32) + delta).astype(np.float16)
    meta["offset_fix"] = delta
    d["meta"] = json.dumps(meta)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)
    return "fixed" if delta else "ok"


def main() -> None:
    paths = [p for p in sorted(CHIPS.glob("*.npz")) if ".tmp" not in p.name]
    counts = {"fixed": 0, "ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(fix, p): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                counts[f.result()] += 1
            except Exception as e:
                counts["error"] += 1
                print(f"{futs[f].name}: {type(e).__name__}: {e}", flush=True)
            if i % 200 == 0:
                print(f"  {i}/{len(paths)} {counts}", flush=True)
    print(f"done: {counts}  ({len(_items)} distinct photos checked)")


if __name__ == "__main__":
    main()
