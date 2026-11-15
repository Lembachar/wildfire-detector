"""Step 4: no-learning baselines the CNN has to beat.

  hot pixel count   number of pixels passing the active-fire test (make_tiles.hot_pixels).
                    At ">= 1" it finds every labelled fire by construction, so its weakness is
                    precision: it also fires on every look-alike.
  max(B12 - B8A)    the hottest pixel's short-wave minus near infrared
  max(B12)          the hottest pixel's short-wave infrared

Thresholds are fitted on train, the variant is picked on val, then scored once on test.
"""
import numpy as np

from common import (OUT, best_threshold, false_alarm_rate, load_tiles, recall_by_size, report,
                    save_metrics)

B12, B11, B8A = 0, 1, 2


def scores(X: np.ndarray, n_hot: np.ndarray) -> dict:
    return {
        "hot pixel count": n_hot.astype(np.float32),
        "max(B12 - B8A)": (X[:, B12] - X[:, B8A]).reshape(len(X), -1).max(1),
        "max(B12)": X[:, B12].reshape(len(X), -1).max(1),
    }


def main() -> None:
    d = load_tiles()
    X, y, split, n_hot = d["X"], d["y"], d["split"], d["n_hot"]
    all_scores = scores(X, n_hot)
    best = None
    for name, s in all_scores.items():
        thr = best_threshold(y[split == 0], s[split == 0])
        m = report(f"val   {name}", y[split == 1], s[split == 1], thr)
        if best is None or m["f1"] > best[2]:
            best = (name, thr, m["f1"])
    name, thr, _ = best
    s = all_scores[name]
    te = split == 2
    test = report(f"TEST  baseline {name}", y[te], s[te], thr)
    test["rule"] = f"{name} >= {thr:.3g}"
    test["by_size"] = recall_by_size(y[te], n_hot[te], (s[te] >= thr).astype(int))
    test["background"] = false_alarm_rate(d, te, (s >= thr).astype(int))
    print("false alarms on fire-free land:", test["background"])
    save_metrics("baseline", test)
    np.save(OUT / "baseline_scores.npy", s)


if __name__ == "__main__":
    main()
