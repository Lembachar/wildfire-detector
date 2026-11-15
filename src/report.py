"""Step 6: build outputs/report/index.html, the demo page.

For test-set chips only (sites the models never saw):
  - true colour | SWIR false colour | SWIR with VIIRS detections and CNN fire tiles marked
  - the CNN's most confident false alarms and its clearest misses
  - baseline vs CNN metrics
"""
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common import OUT, ROOT, load_tiles

REPORT = OUT / "report"
TILE = 32


def swir_rgb(swir: np.ndarray) -> np.ndarray:
    """B12, B11, B8A as R, G, B: active fire shows up bright red/orange, vegetation green."""
    img = np.nan_to_num(swir.astype(np.float32)).transpose(1, 2, 0)
    return np.clip(img / 0.5, 0, 1) ** 0.8


def chip_figure(name: str, meta: dict, cnn_tiles: list, thr: float) -> str:
    d = np.load(ROOT / "data" / "chips" / f"{name}.npz")
    same = d["hot_day"] == 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(d["rgb"])
    axes[0].set_title("True colour (what a camera sees)")
    axes[1].imshow(swir_rgb(d["swir"]))
    axes[1].set_title("Short-wave infrared: B12, B11, B8A")
    axes[2].imshow(swir_rgb(d["swir"]))
    axes[2].scatter(d["hot_col"][same], d["hot_row"][same], s=28, facecolors="none",
                    edgecolors="cyan", linewidths=1.2, label="VIIRS detection, same day")
    for (tr, tc), p in cnn_tiles:
        if p >= thr:
            axes[2].add_patch(Rectangle((tc * TILE, tr * TILE), TILE, TILE, fill=False,
                                        edgecolor="yellow", linewidth=1.4))
    axes[2].plot([], [], "s", mfc="none", mec="yellow", label="CNN says fire")
    axes[2].legend(loc="lower left", fontsize=8, framealpha=0.7)
    axes[2].set_title("Labels vs CNN")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{meta['date']}  ·  {meta['lat']:.2f}, {meta['lon']:.2f}  ·  scene {meta['scene']}",
                 fontsize=10)
    fig.tight_layout()
    out = f"chip_{name}.png"
    fig.savefig(REPORT / out, dpi=90)
    plt.close(fig)
    return out


def gallery(X: np.ndarray, idx: np.ndarray, scores: np.ndarray, title: str, out: str) -> str:
    n = len(idx)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(2 * max(n, 1), 2.4))
    axes = np.atleast_1d(axes)
    for a, i in zip(axes, idx):
        a.imshow(swir_rgb(X[i]), interpolation="nearest")
        a.set_title(f"p = {scores[i]:.2f}", fontsize=9)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORT / out, dpi=90)
    plt.close(fig)
    return out


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    d = load_tiles()
    y, split, chip_idx, tile_rc = d["y"], d["split"], d["chip_idx"], d["tile_rc"]
    cnn = np.load(OUT / "cnn_scores.npy")
    metrics = json.loads((OUT / "metrics.json").read_text())
    thr = metrics["cnn"]["threshold"]

    # Chips from test sites, most fire first.
    test_chips = sorted({int(c) for c in chip_idx[split == 2]},
                        key=lambda c: -int(y[(chip_idx == c)].sum()))
    figs = []
    for c in test_chips[:6]:
        m = chip_idx == c
        tiles = [((int(r), int(cc)), float(p)) for (r, cc), p in zip(tile_rc[m], cnn[m])]
        figs.append(chip_figure(d["chips"][c], d["metas"][c], tiles, thr))

    te = np.where(split == 2)[0]
    fp = te[(y[te] == 0) & (cnn[te] >= thr)]
    fn = te[(y[te] == 1) & (cnn[te] < thr)]
    fp = fp[np.argsort(-cnn[fp])][:8]
    fn = fn[np.argsort(cnn[fn])][:8]
    fp_png = gallery(d["X"], fp, cnn, "Most confident false alarms (no same-day VIIRS point)", "false_alarms.png")
    fn_png = gallery(d["X"], fn, cnn, "Clearest misses (VIIRS says fire, CNN says no)", "misses.png")

    # How far apart in time the label (VIIRS) and the image (Sentinel-2) were taken.
    gaps = []
    for name in d["chips"]:
        z = np.load(ROOT / "data" / "chips" / f"{name}.npz")
        gaps.append(z["hot_gap_min"][z["hot_day"] == 0])
    gaps = np.concatenate(gaps)
    gap_txt = (f"median {np.median(gaps) / 60:.1f} h (VIIRS later)" if np.median(gaps) > 0
               else f"median {-np.median(gaps) / 60:.1f} h (VIIRS earlier)") if len(gaps) else "n/a"

    rows = ""
    for name, label in [("baseline", f"Baseline rule: {metrics['baseline'].get('rule', '')}"), ("cnn", "Small CNN")]:
        m = metrics[name]
        bg = m.get("background", {})
        fa = "" if bg.get("per_1000_km2") is None else f"{bg['per_1000_km2']:.1f} ({bg['false_alarms']})"
        rows += (f"<tr><td>{html.escape(label)}</td><td>{m['ap']:.3f}</td><td>{m['precision']:.3f}</td>"
                 f"<td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td><td>{fa}</td></tr>")
    size_rows = ""
    for size in metrics["cnn"]["by_size"]:
        cells = ""
        for name in ("baseline", "cnn"):
            rec = metrics[name]["by_size"][size]["recall"]
            cells += f"<td>{'' if rec is None else format(rec, '.0%')}</td>"
        size_rows += f"<tr><td>{size}</td><td>{metrics['cnn']['by_size'][size]['n']}</td>{cells}</tr>"
    n_tiles = len(y)
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Wildfire detector prototype</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#fff;color:#1d1d1f}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border-bottom:1px solid #ddd;padding:6px 14px;text-align:right}}
td:first-child,th:first-child{{text-align:left}} img{{max-width:100%;margin:10px 0;border-radius:6px}}
.note{{color:#555}} h2{{margin-top:36px}}
</style></head><body>
<h1>Wildfire detector prototype</h1>
<p>Sentinel-2 images (20 m) labelled with NASA FIRMS VIIRS same-day fire detections.
{len(d['chips'])} chips, {n_tiles:,} tiles of 640 m, split by site so test fires were never seen in training.</p>
<h2>Test results</h2>
<table><tr><th>Model</th><th>Avg precision</th><th>Precision</th><th>Recall</th><th>F1</th>
<th>False alarms per 1,000 km² fire-free land (count)</th></tr>{rows}</table>
<p class="note">Test set: {metrics['cnn']['n']:,} tiles, {metrics['cnn']['positives']} with fire.
Time between the Sentinel-2 image and the VIIRS detections: {gap_txt}. Because of that gap,
a fire tile is one with visible hot pixels in the image confirmed by a same-day VIIRS detection
within 2 km. Hot pixels with no VIIRS detection nearby count as false alarms, though some may be
small real fires VIIRS missed.</p>
<h2>Small fires: share caught, by visibly burning area in the tile</h2>
<table><tr><th>Fire size</th><th>Fire tiles</th><th>Baseline</th><th>CNN</th></tr>{size_rows}</table>
<p class="note">Early warning lives in the "small" row: a fire caught while small is one that can still be stopped.</p>
<h2>Test scenes</h2>
{''.join(f'<img src="{f}" alt="test chip">' for f in figs)}
<h2>Where it goes wrong</h2>
<img src="{fp_png}" alt="false alarms"><img src="{fn_png}" alt="misses">
</body></html>"""
    (REPORT / "index.html").write_text(body, encoding="utf-8")
    print(f"wrote {REPORT / 'index.html'}")


if __name__ == "__main__":
    main()
