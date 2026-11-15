"""Test the built-in map of permanent heat sources (data/static_heat.npz, from build_static_heat.py).

Each val / test tile is placed on the map. A flagged tile that holds a known permanent heat source (gas flare,
factory, volcano: NASA type 1-3, seen on at least MIN_DAYS days in the year) within MARGIN pixels is dropped.
MIN_DAYS and MARGIN are picked on val only; test is reported once with them. Also reports, before any choice,
how many real fire tiles vs false alarms the map touches: the map is only useful if it hits false alarms
and not fires. Works on the current models and on the photo-only students if they exist.

    python src/eval_static.py
"""
import json

import numpy as np
from rasterio.warp import transform as warp_transform

from compare_models import CLS
from eval_history import TILE_KM2, model_scores
from train_v2 import OUT, ROOT, TILE, Data, prf

MIN_DAYS = [1, 3, 10, 30]
MARGINS = [0, 10, 20]                                 # px of 20 m around the 640 m tile


def tile_hits(d: Data, chips, z, min_days: int, margin: int) -> dict:
    """chip -> (16, 16) bool: a permanent heat source within the tile (+ margin)."""
    keep = z["days"] >= min_days
    lat, lon = z["lat"][keep], z["lon"][keep]
    n = 512 // TILE
    out = {}
    for c in chips:
        m = d.metas[c]
        tr = m["transform"]
        t = [float(v) for v in (tr if isinstance(tr, list) else tr.strip("[]").split(","))]
        lat0, lon0 = float(m["lat"]), float(m["lon"])
        near = (np.abs(lat - lat0) < 0.15) & (np.abs(lon - lon0) < 0.15 / max(np.cos(np.radians(lat0)), 0.1))
        hits = np.zeros((n, n), bool)
        if near.any():
            x, y = warp_transform("EPSG:4326", m["crs"], lon[near].tolist(), lat[near].tolist())
            col, row = (np.asarray(x) - t[2]) / t[0], (np.asarray(y) - t[5]) / t[4]
            for r0 in range(n):
                for c0 in range(n):
                    a, b = r0 * TILE - margin, (r0 + 1) * TILE + margin
                    e, f = c0 * TILE - margin, (c0 + 1) * TILE + margin
                    hits[r0, c0] = ((row >= a) & (row < b) & (col >= e) & (col < f)).any()
        out[c] = hits
    return out


def main() -> None:
    z = np.load(ROOT / "data" / "static_heat.npz")
    print(f"map: {len(z['lat']):,} cells, period {z['period']}", flush=True)
    d = Data()
    va, te = d.rows(1), d.rows(2)
    yv = d.t["y"][va]
    variants = {}
    ens = json.loads((OUT / "ensemble.json").read_text())
    S = model_scores(d, va, te, "")
    variants["current"] = (sum(ens["weights"][k] * S[k][0] for k in CLS),
                           sum(ens["weights"][k] * S[k][1] for k in CLS), ens["threshold"])
    if (OUT / "ensemble_distil.json").exists():
        ed = json.loads((OUT / "ensemble_distil.json").read_text())
        Sd = model_scores(d, va, te, "_distil")
        variants["student"] = (sum(ed["weights"][k] * Sd[k][0] for k in CLS),
                               sum(ed["weights"][k] * Sd[k][1] for k in CLS), ed["threshold"])
    chips = np.unique(d.t["chip"][np.concatenate([va, te])])
    kind_of = np.array(["city" if c.startswith("hcity") else "cloud" if c.startswith("hcloud") else "other"
                        for c in d.chips])
    H = {(md, mg): tile_hits(d, chips, z, md, mg) for md in MIN_DAYS for mg in MARGINS}

    def hit(rows, key):
        return np.array([H[key][c][r, q] for c, r, q in zip(d.t["chip"][rows], d.t["tr"][rows], d.t["tc"][rows])])

    def report(split, rows, name, pred):
        y, bg, nf = d.t["y"][rows], d.t["background"][rows], d.t["n_fire"][rows]
        kind = kind_of[d.t["chip"][rows]]
        m = prf(y, pred)
        fa = {k: float(pred[mask].sum() / max(mask.sum() * TILE_KM2, 1e-9) * 1000) for k, mask in
              [("all", (bg == 1) & (y == 0)), ("city", (kind == "city") & (y == 0)),
               ("cloud", (kind == "cloud") & (y == 0))]}
        sizes = {k: float(pred[(y == 1) & (nf >= lo) & (nf < hi)].mean())
                 for k, lo, hi in [("tiny", 1, 5), ("small", 5, 25), ("large", 25, 10 ** 9)]}
        print(f"{split:4s} {name:40s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} | FA all "
              f"{fa['all']:5.1f} city {fa['city']:5.1f} cloud {fa['cloud']:5.1f} | tiny {sizes['tiny']:.2f} "
              f"small {sizes['small']:.2f} large {sizes['large']:.2f}", flush=True)
        return dict(m, false_alarms=fa, recall_by_size=sizes)

    res = {}
    for vname, v in variants.items():
        base_v, base_t = v[0] >= v[2], v[1] >= v[2]
        for key in H:
            h = hit(va, key)
            fp, tp = (base_v & (yv == 0) & h).sum(), (base_v & (yv == 1) & h).sum()
            print(f"val  {vname}: map days>={key[0]:2d} margin {key[1]:2d}px touches {fp} of "
                  f"{(base_v & (yv == 0)).sum()} false alarms and {tp} of {(base_v & (yv == 1)).sum()} caught fires",
                  flush=True)
        # pick on val: best F1 (ties: fewer fires lost)
        best = max(H, key=lambda k: (prf(yv, base_v & ~hit(va, k))["f1"], -k[1]))
        print(f"chosen on val for {vname}: days>={best[0]}, margin {best[1]}px", flush=True)
        for split, rows, base in (("val", va, base_v), ("test", te, base_t)):
            res[f"{split} {vname}"] = report(split, rows, vname, base)
            res[f"{split} {vname} + map"] = report(split, rows, f"{vname} + map", base & ~hit(rows, best))
        res[f"{vname} choice"] = dict(min_days=best[0], margin=best[1])
    (OUT / "static_check.json").write_text(json.dumps(res, indent=1, default=float))
    print("saved outputs/static_check.json")


if __name__ == "__main__":
    main()
