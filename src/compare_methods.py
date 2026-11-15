"""Our models vs published active-fire methods, on the SAME test tiles, scored the SAME way.

A published method (src/rule_methods.py) marks fire pixels; a 640 m tile counts as flagged if any pixel
in it is marked, which is how our tile scores are read too. Published settings are used unchanged (no
tuning on our data). Our models use the weights and threshold picked on the validation set only.

Scored against the test labels the project already has (NASA + hot-pixel rule + 300 eye-checked tiles).
Those labels partly come from a hot-pixel rule, so they are not neutral; the fair final score uses the
blind hand labels (src/make_method_review.py). Writes outputs/methods_test.json and the per-tile flags
outputs/methods_test_flags.npz (used to build the blind review set).

    python src/compare_methods.py
"""
import json
import warnings
from pathlib import Path

import numpy as np

import rule_methods as R
from train_v2 import DATA, OUT, ROOT, SIZE_BINS, TILE, average_precision, prf

TILE_KM2 = (TILE * 0.02) ** 2
BIOMES = ROOT / "data" / "biomes.json"          # chip name -> RESOLVE biome (src/chip_biomes.py), for Hu et al.


def tile_flags(mask: np.ndarray, rows_rc: np.ndarray) -> np.ndarray:
    n = 512 // TILE
    any_px = mask.reshape(n, TILE, n, TILE).any(axis=(1, 3))
    return any_px[rows_rc[:, 0], rows_rc[:, 1]]


def score(t: dict, kind: np.ndarray, rows: np.ndarray, pred: np.ndarray, s: np.ndarray | None = None) -> dict:
    y, bg, nf = t["y"][rows], t["background"][rows], t["n_fire"][rows]
    m = prf(y, pred)
    m["ap"] = average_precision(y, s) if s is not None else None
    m["tiles"], m["fire_tiles"] = int(len(y)), int(y.sum())
    m["by_size"] = {name: float(pred[(nf >= lo) & (nf < hi)].mean()) for name, lo, hi in SIZE_BINS}
    k = (bg == 1) & (y == 0)
    m["false_alarms_per_1000_km2"] = float(pred[k].sum() / (k.sum() * TILE_KM2) * 1000)
    for g in ("city", "cloud"):
        k = (kind[rows] == g) & (y == 0)
        m[f"false_alarms_per_1000_km2_{g}"] = float(pred[k].sum() / (k.sum() * TILE_KM2) * 1000)
    return m


def main() -> None:
    warnings.simplefilter("ignore")
    t = dict(np.load(DATA / "tiles.npz"))
    chips = json.loads((DATA / "chips.json").read_text())["chips"]
    kind = np.array(["city" if c.startswith("hcity") else "cloud" if c.startswith("hcloud") else "other"
                     for c in chips])[t["chip"]]
    te = np.where(t["split"] == 2)[0]
    X = np.load(DATA / "X.npy", mmap_mode="r")
    biomes = json.loads(BIOMES.read_text()) if BIOMES.exists() else None

    methods = ["Schroeder 2016", "Kumar-Roy 2018", "Murphy 2016"] + (["Hu 2021 (AFD-S2)"] if biomes else [])
    # TOA versions (as designed): needs data/l1c from src/fetch_l1c.py for every test photo.
    l1c_dir = ROOT / "data" / "l1c"
    have_toa = all((l1c_dir / f"{chips[c]}.npz").exists() for c in np.unique(t["chip"][te]))
    toa_methods = [f"{m} TOA" for m in methods] if have_toa else []
    flags = {m: np.zeros(len(te), bool) for m in methods + toa_methods}
    hu_covered = np.zeros(len(te), bool)
    hot_rule = np.zeros(len(te), bool)
    pos = {r: i for i, r in enumerate(te)}
    test_chips = np.unique(t["chip"][te])
    for n, c in enumerate(test_chips):
        rows = te[t["chip"][te] == c]
        idx = [pos[r] for r in rows]
        rc = np.stack([t["tr"][rows], t["tc"][rows]], 1)
        x7 = np.asarray(X[c, :, 16:528, 16:528], np.float32)
        b = R.landsat_bands(x7)
        with np.load(ROOT / "data" / "chips" / f"{chips[c]}.npz") as z:
            saturated = z["scl"] == 1
        flags["Schroeder 2016"][idx] = tile_flags(R.schroeder(b), rc)
        flags["Kumar-Roy 2018"][idx] = tile_flags(R.kumar_roy(b), rc)
        flags["Murphy 2016"][idx] = tile_flags(R.murphy(b, saturated), rc)
        if biomes:
            hm = R.hu(b, biomes.get(chips[c], ""))
            if hm is not None:
                flags["Hu 2021 (AFD-S2)"][idx] = tile_flags(hm, rc)
                hu_covered[idx] = True
        if have_toa:
            with np.load(l1c_dir / f"{chips[c]}.npz") as z:
                toa, zen, lm = z["toa"].astype(np.float64), float(z["sun_zen"]), json.loads(str(z["meta"]))
            dn = toa * 10000.0 - lm["offset"]                        # back to digital numbers: 65535 = saturated
            sat_toa = np.nan_to_num(dn[4:7] >= 65534.5).any(axis=0)
            tb = {k + 1: np.nan_to_num(toa[k]) for k in range(7)}    # B01,B02,B03,B04,B8A,B11,B12 -> OLI 1..7
            # Sentinel-2 L1C reflectance is already divided by cos(sun zenith); Schroeder used reflectance
            # WITHOUT that correction (Pereira code: `reflectance`, not `corrected`), so it gets rho * cos(zen).
            tb_unc = {k: v * np.cos(np.radians(zen)) for k, v in tb.items()}
            flags["Schroeder 2016 TOA"][idx] = tile_flags(R.schroeder(tb_unc), rc)
            flags["Kumar-Roy 2018 TOA"][idx] = tile_flags(R.kumar_roy(tb), rc)
            flags["Murphy 2016 TOA"][idx] = tile_flags(R.murphy(tb, sat_toa), rc)
            if biomes:
                hm = R.hu(tb, biomes.get(chips[c], ""))
                if hm is not None:
                    flags["Hu 2021 (AFD-S2) TOA"][idx] = tile_flags(hm, rc)
        b12, b11, b8a = x7[:3]
        hot_rule[idx] = tile_flags((b12 > 0.25) & (b12 > b11) & (b12 > b8a), rc)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(test_chips)} photos", flush=True)

    # Our models: saved test scores (same row order as te), combined with the val-chosen weights/threshold.
    def ens(name: str, suffix: str) -> np.ndarray:
        cfg = json.loads((OUT / name).read_text())
        return sum(w * np.load(OUT / f"scores_{k}{suffix}_test.npy") for k, w in cfg["weights"].items() if w), \
            cfg["threshold"]
    s_photo, thr_photo = ens("ensemble.json", "")
    s_hist, thr_hist = ens("ensemble_hist.json", "_hist")

    res = {}
    rows_all = te
    res["ours: before+now (live)"] = score(t, kind, rows_all, s_hist >= thr_hist, s_hist)
    res["ours: photo only"] = score(t, kind, rows_all, s_photo >= thr_photo, s_photo)
    for m in toa_methods + methods:                             # TOA = as designed; plain = on surface reflectance
        if m.startswith("Hu"):
            continue
        res[m if m.endswith("TOA") else f"{m} (on surface reflectance)"] = score(t, kind, rows_all, flags[m])
    res["hot-pixel rule used to make the labels (circular)"] = score(t, kind, rows_all, hot_rule)
    if biomes:                                                   # Hu only on the biomes it was made for
        sub = rows_all[hu_covered]
        res["[Hu biomes only] ours: before+now (live)"] = score(t, kind, sub, (s_hist >= thr_hist)[hu_covered],
                                                               s_hist[hu_covered])
        for m in [x for x in toa_methods if x.startswith("Hu")] + ["Hu 2021 (AFD-S2)"]:
            res[f"[Hu biomes only] {m if m.endswith('TOA') else m + ' (on surface reflectance)'}"] = \
                score(t, kind, sub, flags[m][hu_covered])
        for m in [x for x in toa_methods if not x.startswith("Hu")]:
            res[f"[Hu biomes only] {m}"] = score(t, kind, sub, flags[m][hu_covered])

    print(f"\n{'method':62s} {'prec':>6s} {'recall':>6s} {'F1':>5s} | {'FA all':>6s} {'city':>6s} {'cloud':>6s} | "
          f"tiny small large")
    for name, r in res.items():
        b = list(r["by_size"].values())
        print(f"{name:62s} {r['precision']:6.1%} {r['recall']:6.1%} {r['f1']:5.3f} | "
              f"{r['false_alarms_per_1000_km2']:6.1f} {r['false_alarms_per_1000_km2_city']:6.1f} "
              f"{r['false_alarms_per_1000_km2_cloud']:6.1f} | {b[0]:4.0%} {b[1]:5.0%} {b[2]:5.0%}")
    (OUT / "methods_test.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / "methods_test_flags.npz", rows=te, hot_rule=hot_rule, hu_covered=hu_covered,
                        ours_hist=s_hist >= thr_hist, ours_photo=s_photo >= thr_photo,
                        **{m.split()[0].lower().replace("-", "_") + ("_toa" if m.endswith("TOA") else ""): f
                           for m, f in flags.items()})
    print("\nsaved outputs/methods_test.json, outputs/methods_test_flags.npz")


if __name__ == "__main__":
    main()
