"""Check every downloaded top-of-atmosphere window against our L2A version of the same photo: the no-data area
must match, and each band must follow the same spatial pattern (correlation; TOA and surface reflectance
differ by haze, not by layout). A cut-off download shows up as a block of zeros / no data or low correlation.
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
        if problems:
            bad.append(f.stem)
            print(f"BAD {f.stem}: {'; '.join(problems)}")
    print(f"checked {len(list(L1C.glob('*.npz')))} files, {len(bad)} bad")
    if "--delete" in sys.argv:
        for b in bad:
            (L1C / f"{b}.npz").unlink()
        print(f"deleted {len(bad)} for re-download")


if __name__ == "__main__":
    main()
