"""Pick the tiles for the eye check: where a trained model disagrees most with the automatic labels.

    python src/make_review_queue.py --model tile --n 300

Half are the model's most confident "fire" calls on tiles labelled not-fire (likely real fires VIIRS
missed, or real false alarms), half its most confident "not fire" calls on tiles labelled fire
(likely bad labels: cloud edges, noise). Tiles from every split are included, so corrected labels
also make the test score honest. Already-reviewed tiles are kept out.
"""
import argparse
import json

import numpy as np
import torch

from train_v2 import DEVICE, ROOT, Data, PreNet, TileNet, UNet, predict

QUEUE = ROOT / "data" / "review_queue.json"
LABELS = ROOT / "data" / "review_labels.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tile", choices=["tile", "unet", "pretrained"])
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    d = Data()
    ckpt = torch.load(ROOT / "outputs" / f"model_{args.model}.pt", map_location=DEVICE, weights_only=False)
    model = {"tile": TileNet, "unet": UNet, "pretrained": PreNet}[args.model]().to(DEVICE)
    model.load_state_dict(ckpt["state"])
    rows = np.arange(len(d.t["y"]))
    s = predict(model, d, rows, args.model)
    y = d.t["y"]

    done = set()
    if LABELS.exists():
        done = {(v["chip"], v["tr"], v["tc"]) for v in json.loads(LABELS.read_text()).values()}
    key = lambda r: (d.chips[d.t["chip"][r]], int(d.t["tr"][r]), int(d.t["tc"][r]))
    fresh = np.array([key(r) not in done for r in rows])

    fp = rows[(y == 0) & fresh][np.argsort(-s[(y == 0) & fresh])][: args.n // 2]
    fn = rows[(y == 1) & fresh][np.argsort(s[(y == 1) & fresh])][: args.n - len(fp)]
    picked = np.concatenate([fp, fn])
    np.random.default_rng(0).shuffle(picked)          # mixed, so the answer isn't guessable from order
    queue = [dict(chip=key(r)[0], tr=key(r)[1], tc=key(r)[2], auto_label=int(y[r]), score=round(float(s[r]), 3),
                  split=int(d.t["split"][r])) for r in picked]
    QUEUE.write_text(json.dumps(queue, indent=1))
    print(f"{len(queue)} tiles queued ({len(fp)} likely false alarms, {len(fn)} likely misses) -> {QUEUE}")


if __name__ == "__main__":
    main()
