"""Score the "was it already there?" check on the real val / test tiles, and pick its threshold on val.

For each tile the combined model flags, look at the pixels that look like heat now (B12 above B11 and B8A)
and ask how much their heat signature (B12 - B11) rose compared with the same pixels in the clear earlier
photos (data/history, from fetch_history.py; widened by 1 px for misalignment). The flag is kept if the rise
is at least RISE, or if those pixels were never seen clearly before (unknown: we can't tell, so we keep it).

Reports fire recall (overall and by fire size) and false alarms (all fire-free / cities / cloud edges),
with and without the check, for several RISE values. Writes outputs/history_check.json.

    python src/eval_history.py
"""
import json

import numpy as np
import torch

from compare_models import CLS, score
from train_v2 import DEVICE, OUT, ROOT, TILE, Data, average_precision, best_threshold, prf

HIST = ROOT / "data" / "history"
RISES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
TILE_KM2 = (TILE * 0.02) ** 2


def dilate_max(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, constant_values=np.nan)
    out = np.full_like(a, np.nan)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out = np.fmax(out, p[dr:dr + a.shape[0], dc:dc + a.shape[1]])
    return out


def tile_rise(d: Data, chip: int, use_max: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per 32 px tile of one chip: largest rise in B12 - B11 among heat-like pixels, and whether known."""
    n = 512 // TILE
    f = HIST / f"{d.chips[chip]}.npz"
    if not f.exists():
        return np.full((n, n), np.nan), np.zeros((n, n), bool)
    x = np.asarray(d.X[chip, :3, 16:528, 16:528], dtype=np.float32)
    b12, b11, b8a = x
    with np.load(f) as z:
        bef = z["before"].astype(np.float32)
        hmax = z["heat_max"].astype(np.float32)
    before_heat = dilate_max(hmax if use_max else bef[0] - bef[1])
    now_heat = b12 - b11
    heat_like = (b12 > b11) & (b12 > b8a) & (b12 >= 0.12)
    rise = np.where(heat_like & np.isfinite(before_heat), now_heat - before_heat, np.nan)
    # tiles with no heat-like pixel: use the tile's hottest pixel instead
    t_rise = np.full((n, n), np.nan)
    t_known = np.zeros((n, n), bool)
    for r in range(n):
        for c in range(n):
            sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            hl = heat_like[sl]
            if hl.any():
                rr = rise[sl][hl]
                t_known[r, c] = np.isfinite(rr).all()
                t_rise[r, c] = np.nanmax(rr) if np.isfinite(rr).any() else np.nan
            else:
                k = np.unravel_index(np.argmax(now_heat[sl]), hl.shape)
                bh = before_heat[sl][k]
                t_known[r, c] = np.isfinite(bh)
                t_rise[r, c] = now_heat[sl][k] - bh if np.isfinite(bh) else np.nan
    return t_rise, t_known


def model_scores(d: Data, va, te, suffix: str) -> dict:
    S = {}
    for kind in CLS:
        ck = torch.load(OUT / f"model_{kind}{suffix}.pt", map_location=DEVICE, weights_only=False)
        m = CLS[kind](ck.get("bands", 7)).to(DEVICE)
        m.load_state_dict(ck["state"])
        d.mean, d.std = ck["mean"].to(DEVICE), ck["std"].to(DEVICE)
        S[kind] = (score(m, d, va, kind, None), score(m, d, te, kind, None))
        del m
        torch.cuda.empty_cache()
    return S


def main() -> None:
    d = Data()
    va, te = d.rows(1), d.rows(2)
    yv = d.t["y"][va]
    variants = {}                                   # name -> (val scores, test scores, threshold)
    ens = json.loads((OUT / "ensemble.json").read_text())
    S = model_scores(d, va, te, "")
    variants["model only"] = (sum(ens["weights"][k] * S[k][0] for k in CLS),
                              sum(ens["weights"][k] * S[k][1] for k in CLS), ens["threshold"])
    if all((OUT / f"model_{k}_hist.pt").exists() for k in CLS):     # the before + now models
        dh = Data(history=True)
        Sh = model_scores(dh, va, te, "_hist")
        best = None
        for w in np.array(np.meshgrid(*[np.linspace(0, 1, 11)] * 3)).reshape(3, -1).T:
            if abs(w.sum() - 1) < 1e-6:
                ap = average_precision(yv, sum(wi * Sh[k][0] for wi, k in zip(w, CLS)))
                if best is None or ap > best[0]:
                    best = (ap, w)
        w = dict(zip(CLS, (round(float(v), 1) for v in best[1])))
        sv_h = sum(w[k] * Sh[k][0] for k in CLS)
        thr_h = best_threshold(yv, sv_h)
        variants["before+now model"] = (sv_h, sum(w[k] * Sh[k][1] for k in CLS), thr_h)
        (OUT / "ensemble_hist.json").write_text(json.dumps(dict(weights=w, threshold=thr_h), indent=1))
        print(f"before+now combined weights {w}, threshold {thr_h:.3f}", flush=True)

    chips = np.unique(d.t["chip"][np.concatenate([va, te])])
    have = sum((HIST / f"{d.chips[c]}.npz").exists() for c in chips)
    print(f"history available for {have} of {len(chips)} val/test photos", flush=True)
    feats = {}
    for use_max in (False, True):
        R, K = {}, {}
        for c in chips:
            R[c], K[c] = tile_rise(d, c, use_max)
        feats[use_max] = R, K

    kind_of = np.array(["city" if c.startswith("hcity") else "cloud" if c.startswith("hcloud") else "other"
                        for c in d.chips])
    res = {}
    for split, rows, vi in (("val", va, 0), ("test", te, 1)):
        y, bg, nf = d.t["y"][rows], d.t["background"][rows], d.t["n_fire"][rows]
        kind = kind_of[d.t["chip"][rows]]
        tr, tc, ch = d.t["tr"][rows], d.t["tc"][rows], d.t["chip"][rows]

        def report(name, pred):  # noqa: E306
            m = prf(y, pred)
            fa = {k: float(pred[mask].sum() / max(mask.sum() * TILE_KM2, 1e-9) * 1000) for k, mask in
                  [("all", (bg == 1) & (y == 0)), ("city", (kind == "city") & (y == 0)),
                   ("cloud", (kind == "cloud") & (y == 0))]}
            sizes = {k: float(pred[(y == 1) & (nf >= lo) & (nf < hi)].mean()) if ((y == 1) & (nf >= lo) & (nf < hi)).any()
                     else None for k, lo, hi in [("tiny", 1, 5), ("small", 5, 25), ("large", 25, 10 ** 9)]}
            res[f"{split} {name}"] = dict(m, false_alarms=fa, recall_by_size=sizes)
            print(f"{split:4s} {name:42s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} | FA/1000km2 all "
                  f"{fa['all']:6.1f} city {fa['city']:6.1f} cloud {fa['cloud']:6.1f} | tiny {sizes['tiny'] or 0:.2f} "
                  f"small {sizes['small'] or 0:.2f} large {sizes['large'] or 0:.2f}", flush=True)

        for vname, v in variants.items():
            base = v[vi] >= v[2]
            report(vname, base)
            for use_max in (False, True):
                R, K = feats[use_max]
                rise = np.array([R[c][r, q] for c, r, q in zip(ch, tr, tc)])
                known = np.array([K[c][r, q] for c, r, q in zip(ch, tr, tc)])
                print(f"     ({'before = max heat seen' if use_max else 'before = median'}: "
                      f"{known[base].mean():.0%} of flagged tiles have a clear before)", flush=True)
                for t in RISES:
                    keep = ~known | (np.nan_to_num(rise, nan=9) >= t)
                    report(f"{vname} + check {t:.2f} {'max' if use_max else 'med'}", base & keep)
    (OUT / "history_check.json").write_text(json.dumps(res, indent=1))
    print("saved outputs/history_check.json")


if __name__ == "__main__":
    main()
