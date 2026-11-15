"""Check every downloaded top-of-atmosphere window against our L2A version of the same photo: the no-data area
must match, and the two must line up pixel for pixel. A cut-off download shows up as a block of zeros / no data;
a wrong window or grid shows up as the best match being at a shifted position.
The plain correlation is only printed: it is legitimately low in smoky, snowy or very uniform scenes (TOA and
surface reflectance differ by haze; with little contrast, haze dominates), which was checked by hand on the
2019-20 Australian bushfire photos.
Bad files are listed (and deleted with --delete so fetch_l1c.py downloads them again).

    python src/check_l1c.py [--delete]
"""
import json
import sys

import numpy as np

from train_v2 import DATA, ROOT

L1C = ROOT / "data" / "l1c"
PAIRS = [("B12", 6, 0), ("B11", 5, 1), ("B8A", 4, 2), ("B04", 3, 3), ("B03", 2, 4), ("B02", 1, 5)]   # (name, toa idx, X idx)


def main() -> None:
    chips = json.loads((DATA / "chips.json").read_text())["chips"]
    X = np.load(DATA / "X.npy", mmap_mode="r")
    bad = []
    for f in sorted(L1C.glob("*.npz")):
        if ".tmp" in f.name:
            continue
        c = chips.index(f.stem)
        x = np.asarray(X[c, :, 16:528, 16:528], np.float32)
        with np.load(f) as z:
            toa = z["toa"]
        l2a_nodata = (x[0] == 0)
        problems = []
        for name, ti, xi in PAIRS:
            t = toa[ti]
            nd = ~np.isfinite(t)
            extra = (nd & ~l2a_nodata).mean()                  # no data in TOA where L2A has data
            ok = np.isfinite(t) & ~l2a_nodata
            r = np.corrcoef(t[ok], x[xi][ok])[0, 1] if ok.sum() > 1000 else np.nan
            # blue / green: haze makes TOA and surface reflectance differ a lot, so only a weak pattern check
            if extra > 0.01 or not (r > (0.4 if name in ("B02", "B03") else 0.8)):
                problems.append(f"{name} extra-nodata {extra:.1%} corr {r:.2f}")
        # alignment: the B12 match must be best with no shift (B12 is the band least affected by haze)
        best = max((np.corrcoef(*[(a[ok2], b[ok2]) for a, b in [(toa[6][8 + dy:504 + dy, 8 + dx:504 + dx], x[0][8:504, 8:504])]
                                  for ok2 in [np.isfinite(a) & (b > 0)]][0])[0, 1], dy, dx)
                   for dy in range(-2, 3) for dx in range(-2, 3))
        a0 = toa[6][8:504, 8:504]; b0 = x[0][8:504, 8:504]; ok0 = np.isfinite(a0) & (b0 > 0)
        r_here = np.corrcoef(a0[ok0], b0[ok0])[0, 1]
        # a clearly better match elsewhere = misaligned. A sub-pixel offset (reprocessed L1C with refined
        # geolocation, e.g. a-90_143_2021-10-10: 0.995 shifted vs 0.994 in place) is not a misalignment.
        if best[1:] != (0, 0) and best[0] - r_here > 0.01:
            problems.append(f"shifted: best B12 match at {best[1:]} ({best[0]:.3f} vs {r_here:.3f} in place)")
        if any("extra-nodata" in q and float(q.split("extra-nodata ")[1].split("%")[0]) > 1 for q in problems)                 or any(q.startswith("shifted") for q in problems):
            pass
        elif problems:
            print(f"ok (low contrast or haze) {f.stem}: {'; '.join(problems)}")
            problems = []
        if problems:
            bad.append(f.stem)
            print(f"BAD {f.stem}: {'; '.join(problems)}")
    print(f"checked {len([f for f in L1C.glob('*.npz') if '.tmp' not in f.name])} files, {len(bad)} bad")
    if "--delete" in sys.argv:
        for b in bad:
            (L1C / f"{b}.npz").unlink()
        print(f"deleted {len(bad)} for re-download")


if __name__ == "__main__":
    main()
