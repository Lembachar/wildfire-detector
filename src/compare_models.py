"""Before vs after the brightness fix, on the SAME test tiles (incl. the new city / cloud-edge chips).

Old models are fed the input they were trained on (0.1 too dark, then cut at 0), recreated per chip from
meta["offset_fix"]; new models get the corrected input. Also picks the combined model (weights + threshold
on val) and saves it to outputs/ensemble.json for the website.

    python src/compare_models.py
"""
import json

import numpy as np
import torch

from train_v2 import (DEVICE, OUT, Data, PreNet, TileNet, UNet, average_precision, best_threshold, evaluate)

CLS = {"tile": TileNet, "unet": UNet, "pretrained": PreNet}


def old_darkening(metas: list) -> np.ndarray:
    """How much darker each chip was under the old (buggy) reading. Repaired chips know it (offset_fix);
    chips downloaded after the fix (offset_fix 0) are looked up: the old code subtracted the listed offset."""
    from concurrent.futures import ThreadPoolExecutor
    from fetch_background import get_item
    out = np.array([m.get("offset_fix", 0.0) for m in metas], np.float32)
    todo = sorted({m["scene"] for m in metas if m.get("offset_fix", 0.0) == 0.0})
    with ThreadPoolExecutor(8) as pool:
        items = dict(zip(todo, pool.map(get_item, todo)))
    for i, m in enumerate(metas):
        if m.get("offset_fix", 0.0) == 0.0:
            band = (items[m["scene"]]["assets"]["swir22"].get("raster:bands") or [{}])[0]
            out[i] = -band.get("offset", 0.0)
    return out


@torch.no_grad()
def score(model, d: Data, rows: np.ndarray, kind: str, darken: np.ndarray | None) -> np.ndarray:
    model.eval()
    out = np.zeros(len(rows), np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    for sel, xc, mc, local in d.chunks(rows, shuffle=False):
        if darken is not None:                              # recreate the old, buggy input
            cs = np.unique(d.t["chip"][sel])
            delta = torch.as_tensor(darken[cs], device=DEVICE, dtype=xc.dtype).view(-1, 1, 1, 1)
            xc = (xc - delta).clamp_(0, 2)
        for i in range(0, len(sel), 1024):
            x, _ = d.crops(xc, mc, local[i:i + 1024], sel[i:i + 1024], normalise=kind != "pretrained")
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                p = torch.sigmoid(model(x).float())
            if kind == "unet":
                p = p[:, 16:48, 16:48].amax(dim=(1, 2))
            out[[pos[r] for r in sel[i:i + 1024]]] = p.cpu().numpy()
    return out


def main() -> None:
    d = Data()
    va, te = d.rows(1), d.rows(2)
    yv = d.t["y"][va]
    darken = old_darkening(d.metas)
    chips = np.array(d.chips)
    kind_of = np.array(["city" if c.startswith("hcity") else "cloud" if c.startswith("hcloud") else "other"
                        for c in chips])
    te_kind = kind_of[d.t["chip"][te]]
    res, S = {}, {}

    def report(name, sv, st):
        r = evaluate(d, te, st, best_threshold(yv, sv))
        pred = st >= r["threshold"]
        for k in ("city", "cloud"):
            m = (te_kind == k) & (d.t["y"][te] == 0)
            r[f"false_alarms_per_1000_km2_{k}"] = float(pred[m].sum() / max(m.sum() * 0.4096, 1e-9) * 1000)
        b = list(r["by_size"].values())
        print(f"{name:40s} AP {r['ap']:.3f} P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} | FA/1000km2 "
              f"all {r['false_alarms_per_1000_km2']:6.1f} city {r['false_alarms_per_1000_km2_city']:6.1f} "
              f"cloud {r['false_alarms_per_1000_km2_cloud']:6.1f} | tiny {b[0]:.2f} small {b[1]:.2f} large {b[2]:.2f}",
              flush=True)
        res[name] = r

    for tag, folder, dark in [("before", OUT / "before_brightness_fix", darken), ("after", OUT, None)]:
        for kind in CLS:
            ck = torch.load(folder / f"model_{kind}.pt", map_location=DEVICE, weights_only=False)
            m = CLS[kind]().to(DEVICE)
            m.load_state_dict(ck["state"])
            d.mean, d.std = ck["mean"].to(DEVICE), ck["std"].to(DEVICE)
            S[(tag, kind)] = (score(m, d, va, kind, dark), score(m, d, te, kind, dark))
            report(f"{tag} {kind}", *S[(tag, kind)])
            del m
            torch.cuda.empty_cache()

    for tag in ("before", "after"):
        best = None
        for w in np.array(np.meshgrid(*[np.linspace(0, 1, 11)] * 3)).reshape(3, -1).T:
            if abs(w.sum() - 1) > 1e-6:
                continue
            ap = average_precision(yv, sum(wi * S[(tag, k)][0] for wi, k in zip(w, CLS)))
            if best is None or ap > best[0]:
                best = (ap, w)
        w = dict(zip(CLS, (round(float(v), 1) for v in best[1])))
        name = f"{tag} combined " + "+".join(f"{k}({v})" for k, v in w.items())
        report(name, sum(w[k] * S[(tag, k)][0] for k in CLS), sum(w[k] * S[(tag, k)][1] for k in CLS))
        if tag == "after":
            (OUT / "ensemble.json").write_text(json.dumps(dict(weights=w, threshold=res[name]["threshold"]), indent=1))
    (OUT / "compare_v5.json").write_text(json.dumps(res, indent=1))
    print("saved outputs/ensemble.json and outputs/compare_v5.json")


if __name__ == "__main__":
    main()
