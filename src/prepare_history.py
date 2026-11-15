"""Line the "before" pictures (data/history, from fetch_history.py) up with data/prepared/X.npy.

  H.npy  float16 (N, 3, 544, 544)  per chip, same padding as X:
           0  B12 before   (median over clear earlier photos)
           1  B11 before
           2  seen before  (1 = at least one clear earlier photo saw this pixel, else 0)
Where a pixel was never seen clearly (clouds, no earlier photo, history not downloaded) the "before" bands
are set to the current values and "seen before" to 0: no change, no information. The model learns to deal
with that, exactly like the website must when every earlier photo is cloudy.

    python src/prepare_history.py
"""
import json

import numpy as np

from train_v2 import DATA, ROOT

HIST = ROOT / "data" / "history"
CHIP, PAD = 512, 16


def main() -> None:
    X = np.load(DATA / "X.npy", mmap_mode="r")
    chips = json.loads((DATA / "chips.json").read_text())["chips"]
    S = CHIP + 2 * PAD
    H = np.lib.format.open_memmap(DATA / "H.npy", mode="w+", dtype=np.float16, shape=(len(chips), 3, S, S))
    have, seen = 0, []
    for i, name in enumerate(chips):
        now = np.asarray(X[i, :2], dtype=np.float32)              # B12, B11 now (padded)
        h = np.zeros((3, S, S), np.float32)
        h[:2] = now
        f = HIST / f"{name}.npz"
        if f.exists():
            with np.load(f) as z:
                bef = z["before"][:2].astype(np.float32)
            ok = np.isfinite(bef).all(axis=0)
            inner = h[:, PAD:PAD + CHIP, PAD:PAD + CHIP]
            for b in range(2):
                inner[b][ok] = bef[b][ok]
            inner[2] = ok
            have += 1
            seen.append(ok.mean())
        H[i] = h.astype(np.float16)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(chips)}", flush=True)
    H.flush()
    print(f"history for {have} of {len(chips)} photos; on those, {np.mean(seen) if seen else 0:.0%} of pixels "
          f"were seen clearly before")


if __name__ == "__main__":
    main()
