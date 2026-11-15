"""Run the trained model on the latest Sentinel-2 image of any place. The core of a live watcher.

    python src/detect.py --lat -14.88 --lon 144.51                # latest clear image, last 10 days
    python src/detect.py --lat -14.88 --lon 144.51 --date 2026-11-11

Prints one ALERT line per tile the model thinks is burning (centre lat/lon + probability), and saves
outputs/detections/<place>_<date>.png and .json. A live version would run this whenever a new image
of a watched area appears, and send each ALERT as a notification.
"""
import argparse
import json
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from matplotlib.patches import Rectangle
from rasterio.warp import transform as warp_transform

from common import OUT
from fetch_s2 import CLOUD_SCL, MAX_CLOUD_SCENE, STAC, read_window
from report import swir_rgb
from train_cnn import FireNet

TILE = 32


def latest_scene(lon: float, lat: float, day: str | None, days: int) -> dict | None:
    end = date.fromisoformat(day) if day else date.today()
    start = end - timedelta(days=0 if day else days)
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 50,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    feats = requests.post(STAC, json=body, timeout=60).json().get("features", [])
    feats = [f for f in feats if f["properties"].get("eo:cloud_cover", 100) <= MAX_CLOUD_SCENE]
    return feats[0] if feats else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", help="YYYY-MM-DD; default: latest image in the last --days days")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--size", type=int, default=512, help="window size in 20 m pixels (512 = 10 km)")
    args = ap.parse_args()

    ckpt = torch.load(OUT / "firenet.pt", weights_only=False, map_location="cpu")
    model = FireNet()
    model.load_state_dict(ckpt["state"])
    model.eval()

    item = latest_scene(args.lon, args.lat, args.date, args.days)
    if item is None:
        print("no clear Sentinel-2 image found for that place and time")
        return
    w = read_window(item, args.lon, args.lat, args.size)
    if w is None:
        print(f"{item['id']}: point too close to the scene edge, try a nearby point")
        return
    swir, scl = np.nan_to_num(w["swir"]), w["scl"]
    n = args.size // TILE
    tiles = swir.reshape(3, n, TILE, n, TILE).transpose(1, 3, 0, 2, 4).reshape(-1, 3, TILE, TILE)
    x = torch.from_numpy((np.clip(tiles, 0, 2) - ckpt["mean"]) / ckpt["std"]).float()
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).numpy().reshape(n, n)
    cloudy = np.isin(scl, CLOUD_SCL).reshape(n, TILE, n, TILE).mean(axis=(1, 3)) > 0.3

    alerts = []
    for tr, tc in zip(*np.where(prob >= ckpt["threshold"])):
        px, py = w["transform"] * (tc * TILE + TILE / 2, tr * TILE + TILE / 2)
        (lon,), (lat,) = warp_transform(w["crs"], "EPSG:4326", [px], [py])
        alerts.append(dict(lat=round(lat, 5), lon=round(lon, 5), probability=round(float(prob[tr, tc]), 3),
                           tile=[int(tr), int(tc)], cloudy=bool(cloudy[tr, tc])))
    alerts.sort(key=lambda a: -a["probability"])

    when = item["properties"]["datetime"]
    print(f"scene {item['id']}  taken {when}  cloud {item['properties']['eo:cloud_cover']:.0f}%")
    for a in alerts:
        print(f"ALERT fire  lat {a['lat']:.5f}  lon {a['lon']:.5f}  p={a['probability']:.2f}"
              f"{'  (partly cloudy tile)' if a['cloudy'] else ''}")
    print(f"{len(alerts)} fire tile(s) of {n * n} checked")

    out_dir = OUT / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.lat:.3f}_{args.lon:.3f}_{when[:10]}"
    (out_dir / f"{stem}.json").write_text(json.dumps(dict(scene=item["id"], datetime=when, alerts=alerts), indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2))
    ax[0].imshow(w["rgb"])
    ax[0].set_title("True colour")
    ax[1].imshow(swir_rgb(w["swir"]))
    for a in alerts:
        tr, tc = a["tile"]
        ax[1].add_patch(Rectangle((tc * TILE, tr * TILE), TILE, TILE, fill=False, edgecolor="yellow", lw=1.4))
    ax[1].set_title(f"Infrared + model detections ({len(alerts)} fire tiles)")
    for a_ in ax:
        a_.set_xticks([])
        a_.set_yticks([])
    fig.suptitle(f"{args.lat:.3f}, {args.lon:.3f}  ·  {when[:16].replace('T', ' ')} UTC  ·  {item['id']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=90)
    print(f"saved {out_dir / stem}.png")


if __name__ == "__main__":
    main()
