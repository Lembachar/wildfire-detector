"""Compare, on the same val / test tiles: the current photo-only models, the before+now models (need earlier
photos at check time), and the photo-only students trained to copy the before+now teachers (no earlier photos).

Combined weights and threshold for the students are picked on val only (same as for the others).
Writes outputs/distil_check.json and outputs/ensemble_distil.json.

    python src/eval_distil.py
"""
import json

import numpy as np

from compare_models import CLS
from eval_history import TILE_KM2, model_scores
from train_v2 import OUT, Data, average_precision, best_threshold, prf


def combine(S, yv):
    best = None
    for w in np.array(np.meshgrid(*[np.linspace(0, 1, 11)] * 3)).reshape(3, -1).T:
        if abs(w.sum() - 1) < 1e-6:
            ap = average_precision(yv, sum(wi * S[k][0] for wi, k in zip(w, CLS)))
            if best is None or ap > best[0]:
                best = (ap, w)
    w = dict(zip(CLS, (round(float(v), 1) for v in best[1])))
    sv = sum(w[k] * S[k][0] for k in CLS)
    return w, sv, sum(w[k] * S[k][1] for k in CLS), best_threshold(yv, sv)


def main() -> None:
    d = Data()
    va, te = d.rows(1), d.rows(2)
    yv = d.t["y"][va]
    variants = {}
    ens = json.loads((OUT / "ensemble.json").read_text())
    S = model_scores(d, va, te, "")
    variants["current (photo only)"] = (sum(ens["weights"][k] * S[k][0] for k in CLS),
                                        sum(ens["weights"][k] * S[k][1] for k in CLS), ens["threshold"])
    eh = json.loads((OUT / "ensemble_hist.json").read_text())
    Sh = model_scores(Data(history=True), va, te, "_hist")
    variants["before+now (needs earlier photos)"] = (sum(eh["weights"][k] * Sh[k][0] for k in CLS),
                                                     sum(eh["weights"][k] * Sh[k][1] for k in CLS), eh["threshold"])
    Sd = model_scores(d, va, te, "_distil")
    w, sv, st, thr = combine(Sd, yv)
    variants["student (photo only, learned from before+now)"] = (sv, st, thr)
    (OUT / "ensemble_distil.json").write_text(json.dumps(dict(weights=w, threshold=thr), indent=1))
    print(f"student combined weights {w}, threshold {thr:.3f}", flush=True)
    for k in CLS:
        print(f"  single model {k:10s} test AP: current {average_precision(d.t['y'][te], S[k][1]):.3f} | "
              f"before+now {average_precision(d.t['y'][te], Sh[k][1]):.3f} | "
              f"student {average_precision(d.t['y'][te], Sd[k][1]):.3f}", flush=True)

    kind_of = np.array(["city" if c.startswith("hcity") else "cloud" if c.startswith("hcloud") else "other"
                        for c in d.chips])
    res = {}
    for split, rows, vi in (("val", va, 0), ("test", te, 1)):
        y, bg, nf = d.t["y"][rows], d.t["background"][rows], d.t["n_fire"][rows]
        kind = kind_of[d.t["chip"][rows]]
        for name, v in variants.items():
            pred = v[vi] >= v[2]
            m = prf(y, pred)
            fa = {k: float(pred[mask].sum() / max(mask.sum() * TILE_KM2, 1e-9) * 1000) for k, mask in
                  [("all", (bg == 1) & (y == 0)), ("city", (kind == "city") & (y == 0)),
                   ("cloud", (kind == "cloud") & (y == 0))]}
            sizes = {k: float(pred[(y == 1) & (nf >= lo) & (nf < hi)].mean())
                     for k, lo, hi in [("tiny", 1, 5), ("small", 5, 25), ("large", 25, 10 ** 9)]}
            res[f"{split} {name}"] = dict(m, ap=average_precision(y, v[vi]), false_alarms=fa, recall_by_size=sizes)
            print(f"{split:4s} {name:47s} AP {res[f'{split} {name}']['ap']:.3f} P {m['precision']:.3f} "
                  f"R {m['recall']:.3f} F1 {m['f1']:.3f} | FA all {fa['all']:5.1f} city {fa['city']:5.1f} "
                  f"cloud {fa['cloud']:5.1f} | tiny {sizes['tiny']:.2f} small {sizes['small']:.2f} "
                  f"large {sizes['large']:.2f}", flush=True)
    (OUT / "distil_check.json").write_text(json.dumps(res, indent=1))
    print("saved outputs/distil_check.json")


if __name__ == "__main__":
    main()
