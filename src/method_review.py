"""Blind hand labels for a fair comparison of our models with published fire methods.

The test labels the project already has were partly made with a hot-pixel rule and NASA points, so they
are not neutral between methods. Here a person labels a random sample of test tiles without seeing which
method flagged them, and every method is scored on those labels.

Rules fixed BEFORE any answer is seen:
  Done in two batches, because the top-of-atmosphere photos for the "as designed" versions were still
  downloading when labelling started. Both batches are random samples inside fixed strata, so the
  combined estimate stays unbiased.
  Batch 1 pool: ours (before+now, photo only) + Schroeder, Kumar-Roy, Murphy on surface reflectance (POOL1).
  Strata over all 72,592 test tiles:
    A  flagged by every POOL1 method                -> 120 random tiles
    B  flagged by some, not all                     -> 280 random tiles
    C  flagged by none, but has a hot pixel          -> 100 random tiles (fires every method missed)
    D  flagged by none, no hot pixel                 -> not sampled, counted as no fire
  Batch 2 (after the TOA download): the D tiles flagged by any TOA method (POOL_TOA) become stratum
    E                                                -> 150 random tiles (all if fewer), seed 2027
    and the rest of D stays "no fire". Batch 2 is appended to the same queue (earlier answers keep their place).
  Random seed 2026. The tiles are shuffled; the reviewer sees only the pictures.
  "Not sure" answers are left out. Each stratum's sample is weighted up to the stratum's size, so the
  estimates are for the whole test set. 95% intervals by bootstrap within strata (2,000 draws).
  Fire = anything burning or glowing hot at the time of the photo (incl. flares, volcanoes), as before.

    python src/method_review.py build      -> data/method_review_queue.json (+ hidden key)
    python src/method_review.py build2     -> appends batch 2 (after compare_methods.py has the TOA flags)
    python src/review.py --queue data/method_review_queue.json --labels data/method_review_labels.json
    python src/method_review.py score      -> outputs/methods_handlabels.json
"""
import json
import sys

import numpy as np

from train_v2 import DATA, OUT, ROOT, TILE

QUEUE = ROOT / "data" / "method_review_queue.json"
KEY = ROOT / "data" / "method_review_key.json"
LABELS = ROOT / "data" / "method_review_labels.json"
POOL1 = ["ours_hist", "ours_photo", "schroeder", "kumar_roy", "murphy"]
POOL_TOA = ["schroeder_toa", "kumar_roy_toa", "murphy_toa", "hu_toa"]
POOL = POOL1[:2] + POOL_TOA + POOL1[2:]
NAMES = {"ours_hist": "ours: before+now (live)", "ours_photo": "ours: photo only",
         "schroeder_toa": "Schroeder 2016 (TOA, as designed)", "kumar_roy_toa": "Kumar-Roy 2018 (TOA, as designed)",
         "murphy_toa": "Murphy 2016 (TOA, as designed)", "hu_toa": "Hu 2021 AFD-S2 (TOA, as designed; its biomes)",
         "schroeder": "Schroeder 2016 (surface refl.)", "kumar_roy": "Kumar-Roy 2018 (surface refl.)",
         "murphy": "Murphy 2016 (surface refl.)"}
SIZES = {"A": 120, "B": 280, "C": 100}
SIZE_E = 150
STRATA = "ABCE"
TILE_KM2 = (TILE * 0.02) ** 2


def strata(f, pool: list, pool2: list | None = None) -> np.ndarray:
    F = np.stack([f[m] for m in pool], 1)
    s = np.full(len(F), "D")
    s[F.any(1) & ~F.all(1)] = "B"
    s[F.all(1)] = "A"
    s[~F.any(1) & f["hot_rule"]] = "C"
    if pool2:
        s[(s == "D") & np.stack([f[m] for m in pool2], 1).any(1)] = "E"
    return s


def build() -> None:
    if QUEUE.exists():
        sys.exit("queue already exists; not rebuilding (answers would no longer match)")
    f = np.load(OUT / "methods_test_flags.npz")
    t = np.load(DATA / "tiles.npz")
    chips = json.loads((DATA / "chips.json").read_text())["chips"]
    pool = [m for m in POOL1 if m in f.files]
    assert pool == POOL1, f"missing flags: {set(POOL1) - set(pool)}"
    s = strata(f, pool)
    rng = np.random.default_rng(2026)
    picked = []
    for k, n in SIZES.items():
        idx = np.where(s == k)[0]
        picked += [(k, int(i)) for i in rng.choice(idx, min(n, len(idx)), replace=False)]
    rng.shuffle(picked)
    rows = f["rows"]
    queue = [dict(chip=chips[t["chip"][rows[i]]], tr=int(t["tr"][rows[i]]), tc=int(t["tc"][rows[i]]))
             for _, i in picked]
    QUEUE.write_text(json.dumps(queue))
    KEY.write_text(json.dumps(dict(pool=pool, items=[dict(stratum=k, test_index=i) for k, i in picked],
                                   stratum_sizes={k: int((s == k).sum()) for k in "ABCD"})))
    print(f"{len(queue)} tiles; stratum sizes {json.loads(KEY.read_text())['stratum_sizes']}; pool {pool}")


def build2() -> None:
    key = json.loads(KEY.read_text())
    if key.get("pool2"):
        sys.exit("batch 2 already built")
    f = np.load(OUT / "methods_test_flags.npz")
    missing = [m for m in POOL_TOA if m not in f.files]
    if missing:
        sys.exit(f"TOA flags missing: {missing}; run compare_methods.py after the download")
    t = np.load(DATA / "tiles.npz")
    chips = json.loads((DATA / "chips.json").read_text())["chips"]
    s = strata(f, key["pool"], POOL_TOA)
    idx = np.where(s == "E")[0]
    rng = np.random.default_rng(2027)
    pick = rng.choice(idx, min(SIZE_E, len(idx)), replace=False)
    rows = f["rows"]
    queue = json.loads(QUEUE.read_text())
    queue += [dict(chip=chips[t["chip"][rows[i]]], tr=int(t["tr"][rows[i]]), tc=int(t["tc"][rows[i]])) for i in pick]
    key["items"] += [dict(stratum="E", test_index=int(i)) for i in pick]
    key["pool2"] = POOL_TOA
    key["stratum_sizes"] = {k: int((s == k).sum()) for k in "ABCDE"}
    QUEUE.write_text(json.dumps(queue))
    KEY.write_text(json.dumps(key))
    print(f"added {len(pick)} tiles (queue now {len(queue)}); stratum sizes {key['stratum_sizes']}")


def estimate(fl: np.ndarray, fire: np.ndarray, st: np.ndarray, sizes: dict) -> dict:
    tp = fp = fn = 0.0
    for k in STRATA:
        m = st == k
        if not m.any():
            continue
        w = sizes[k] / m.sum()
        tp += w * (fl[m] & fire[m]).sum()
        fp += w * (fl[m] & ~fire[m]).sum()
        fn += w * (~fl[m] & fire[m]).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, fp=fp, tp=tp, fn=fn)


def score() -> None:
    f = np.load(OUT / "methods_test_flags.npz")
    key = json.loads(KEY.read_text())
    answers = json.loads(LABELS.read_text())
    items = key["items"]
    lab = np.array([answers.get(str(i), {}).get("label", "missing") for i in range(len(items))])
    sure = (lab == "fire") | (lab == "not")
    st = np.array([it["stratum"] for it in items])[sure]
    ti = np.array([it["test_index"] for it in items])[sure]
    fire = (lab == "fire")[sure]
    sizes = key["stratum_sizes"]
    area = len(f["rows"]) * TILE_KM2
    rng = np.random.default_rng(0)
    res = dict(answered=int(sure.sum()), unsure=int((lab == "unsure").sum()), missing=int((lab == "missing").sum()),
               fire_share_by_stratum={k: float(fire[st == k].mean()) for k in STRATA if (st == k).any()})
    s_all = strata(f, key["pool"], key.get("pool2"))
    cov = f["hu_covered"]
    sizes_cov = {k: int(((s_all == k) & cov).sum()) for k in "ABCDE"}       # Hu: only the biomes it was made for
    area_cov = int(cov.sum()) * TILE_KM2
    methods = [m for m in POOL if m in key["pool"] + key.get("pool2", [])]   # scored only if sampled for
    runs = [(m, NAMES[m], False) for m in methods]
    if "hu_toa" in methods:
        runs += [(m, f"[Hu biomes only] {NAMES[m]}", True) for m in methods if m != "hu_toa"]
        runs = [r for r in runs if not (r[0] == "hu_toa" and not r[2])] + [("hu_toa", f"[Hu biomes only] {NAMES['hu_toa']}", True)]
    for m, name, only_cov in runs:
        keep = cov[ti] if only_cov else np.ones(len(ti), bool)
        fl, fi, s_, sz = f[m][ti][keep], fire[keep], st[keep], (sizes_cov if only_cov else sizes)
        e = estimate(fl, fi, s_, sz)
        boot = []
        for _ in range(2000):
            sel = np.concatenate([rng.choice(np.where(s_ == k)[0], (s_ == k).sum()) for k in STRATA if (s_ == k).any()])
            b = estimate(fl[sel], fi[sel], s_[sel], sz)
            boot.append((b["precision"], b["recall"], b["f1"]))
        lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
        e["ci95"] = dict(precision=[lo[0], hi[0]], recall=[lo[1], hi[1]], f1=[lo[2], hi[2]])
        e["false_alarms_per_1000_km2"] = e["fp"] / (area_cov if only_cov else area) * 1000
        res[name] = e
    (OUT / "methods_handlabels.json").write_text(json.dumps(res, indent=1))
    print(f"answered {res['answered']}, not sure {res['unsure']}, missing {res['missing']}; "
          f"fire share A/B/C {res['fire_share_by_stratum']}")
    for _, name, _ in runs:
        e = res[name]
        c = e["ci95"]
        print(f"{name:66s} precision {e['precision']:5.1%} [{c['precision'][0]:.0%}-{c['precision'][1]:.0%}]  "
              f"recall {e['recall']:5.1%} [{c['recall'][0]:.0%}-{c['recall'][1]:.0%}]  F1 {e['f1']:.3f}  "
              f"FA/1000km2 {e['false_alarms_per_1000_km2']:.1f}")


if __name__ == "__main__":
    {"build": build, "build2": build2, "score": score}[sys.argv[1]]()
