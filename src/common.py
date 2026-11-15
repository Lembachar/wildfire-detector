"""Shared helpers: loading tiles and scoring predictions the same way for every model."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load_tiles() -> dict:
    d = np.load(ROOT / "data" / "tiles.npz")
    return dict(X=d["X"].astype(np.float32), y=d["y"].astype(np.int64), split=d["split"],
                chip_idx=d["chip_idx"], tile_rc=d["tile_rc"], n_hot=d["n_hot"],
                chips=json.loads(str(d["chips"])), metas=json.loads(str(d["metas"])))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise), the standard metric for rare positives."""
    order = np.argsort(-score, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold with the best F1, chosen on a split the model wasn't scored on."""
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 400)))
    f1s = [prf(y, (score >= t).astype(int))["f1"] for t in cands]
    return float(cands[int(np.argmax(f1s))])


def report(name: str, y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    m = prf(y, (score >= thr).astype(int))
    m.update(ap=average_precision(y, score), threshold=thr, n=int(len(y)), positives=int(y.sum()))
    print(f"{name:28s} AP {m['ap']:.3f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
          f"F1 {m['f1']:.3f}  (thr {thr:.4g})")
    return m


# Burning area in the tile, from the number of 20 m hot pixels (1 pixel = 0.04 ha).
SIZE_BINS = [("tiny: 1-4 hot pixels (< 0.2 ha)", 1, 5), ("small: 5-24 px (0.2-1 ha)", 5, 25),
             ("large: 25+ px (> 1 ha)", 25, np.inf)]


def recall_by_size(y: np.ndarray, n_hot: np.ndarray, pred: np.ndarray) -> dict:
    """Share of fire tiles caught, split by how much of the tile is visibly burning."""
    out = {}
    for name, lo, hi in SIZE_BINS:
        m = (y == 1) & (n_hot >= lo) & (n_hot < hi)
        out[name] = dict(n=int(m.sum()), recall=float(pred[m].mean()) if m.any() else None)
    return out


TILE_KM2 = 0.64 * 0.64


def false_alarm_rate(d: dict, mask: np.ndarray, pred: np.ndarray) -> dict:
    """False alarms per 1,000 km2 of fire-free land, measured on background chips only.

    This is the number that decides whether a worldwide alert system is usable: scanning the
    whole planet, even a tiny rate turns into many false alerts a day.
    """
    bg = np.array(["_bg" in d["chips"][c] for c in d["chip_idx"]])
    m = mask & bg & (d["y"] == 0)
    km2 = m.sum() * TILE_KM2
    fa = int(pred[m].sum())
    return dict(false_alarms=fa, km2=float(km2), per_1000_km2=fa / km2 * 1000 if km2 else None)


def save_metrics(name: str, metrics: dict) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / "metrics.json"
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[name] = metrics
    path.write_text(json.dumps(allm, indent=2))
