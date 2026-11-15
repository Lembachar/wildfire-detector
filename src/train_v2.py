"""Step 5 (v2): train and score the tile model or the pixel model on data/prepared.

    python src/train_v2.py --model tile     # says fire / not fire per 640 m tile
    python src/train_v2.py --model unet     # outlines fire pixel by pixel (20 m); tile score = hottest pixel
    python src/train_v2.py --model rule     # no learning: "tile has a hot pixel" (the bar to beat)

Both models see the tile plus 320 m of surroundings (a 64 x 64 px window, 7 bands) and are scored on
EVERY usable tile of the test images (burned ground included), not just the easy ones.

Data streaming: photos are loaded to the GPU in chunks, and crops are cut on the GPU, so the full
dataset never has to fit in RAM.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared"
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE, WIN = 32, 64
CHUNK = 40                 # photos on the GPU at once (~170 MB)
BATCH = 256
TILE_KM2 = 0.64 * 0.64
SIZE_BINS = [("tiny (1-4 fire px, < 0.2 ha)", 1, 5), ("small (5-24 px, 0.2-1 ha)", 5, 25),
             ("large (25+ px, > 1 ha)", 25, 10 ** 9)]


# ---------------------------------------------------------------- models
def block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TileNet(nn.Module):
    """Looks at the 64 px window, decides only about the centre 32 px tile."""

    def __init__(self, bands: int = 7) -> None:
        super().__init__()
        self.f = nn.Sequential(block(bands, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2), block(64, 128))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.f(x)                       # (B, 128, 16, 16): each cell = 4 px
        centre = f[:, :, 4:12, 4:12]        # the cells covering the centre tile
        return self.head(centre.amax(dim=(2, 3))).squeeze(1)


class UNet(nn.Module):
    """Small U-Net: one fire probability per pixel."""

    def __init__(self, bands: int = 7) -> None:
        super().__init__()
        self.e1, self.e2, self.e3 = block(bands, 32), block(32, 64), block(64, 128)
        self.up2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
        self.up1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1).squeeze(1)      # (B, 64, 64) logits


PRETRAINED_URL = ("https://hf.co/torchgeo/resnet18_sentinel2_all_moco/resolve/"
                  "5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth")
# Our 7 bands (B12, B11, B8A, B4, B3, B2, B8) as positions in SSL4EO's 13 (B1..B12 incl. B8A).
SSL4EO_IDX = [12, 11, 8, 3, 2, 1, 7]


class PreNet(nn.Module):
    """ResNet-18 pre-trained on 1M+ Sentinel-2 images (SSL4EO-S12, MoCo), fine-tuned for fire.

    The 64 px window is upsampled to 128 so the final 4 x 4 feature map keeps a centre 2 x 2 that
    covers exactly the tile. Input is plain reflectance, the scale the weights were trained on.
    """

    def __init__(self, bands: int = 7) -> None:
        super().__init__()
        import torchvision
        net = torchvision.models.resnet18(weights=None)
        sd = torch.hub.load_state_dict_from_url(PRETRAINED_URL, map_location="cpu", progress=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        w13 = sd.pop("conv1.weight")                         # (64, 13, 7, 7)
        net.conv1 = nn.Conv2d(bands, 64, 7, 2, 3, bias=False)
        net.load_state_dict(sd, strict=False)
        with torch.no_grad():
            net.conv1.weight.zero_()                         # extra "before" bands start at 0: same as without
            net.conv1.weight[:, :7].copy_(w13[:, SSL4EO_IDX] * 13 / 7)
        self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(f[:, :, 1:3, 1:3].amax(dim=(2, 3))).squeeze(1)


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, history: bool = False) -> None:
        self.X = np.load(DATA / "X.npy", mmap_mode="r")
        self.M = np.load(DATA / "M.npy", mmap_mode="r")
        # history: 3 extra bands per pixel (B12 before, B11 before, seen before) from prepare_history.py
        self.H = np.load(DATA / "H.npy", mmap_mode="r") if history else None
        self.bands = 10 if history else 7
        t = np.load(DATA / "tiles.npz")
        self.t = {k: t[k] for k in t.files}
        info = json.loads((DATA / "chips.json").read_text())
        self.chips, self.metas = info["chips"], info["metas"]
        train_chips = np.unique(self.t["chip"][self.t["split"] == 0])
        sample = np.random.default_rng(0).choice(train_chips, min(40, len(train_chips)), replace=False)
        xs = np.stack([np.asarray(self.photo(c)[:, 16:528:4, 16:528:4], dtype=np.float32) for c in sorted(sample)])
        self.mean = torch.tensor(xs.mean((0, 2, 3)), device=DEVICE).view(1, -1, 1, 1)
        self.std = torch.tensor(xs.std((0, 2, 3)) + 1e-4, device=DEVICE).view(1, -1, 1, 1)

    def photo(self, c):
        """All input bands of photo(s) c: the 7 current ones, plus the 3 "before" ones if history is on."""
        x = np.asarray(self.X[c])
        return x if self.H is None else np.concatenate([x, np.asarray(self.H[c])], axis=-3)

    def rows(self, split: int) -> np.ndarray:
        return np.where(self.t["split"] == split)[0]

    def chunks(self, rows: np.ndarray, shuffle: bool):
        """Yield (rows, X chunk on GPU, M chunk on GPU, local chip index per row)."""
        chips = np.unique(self.t["chip"][rows])
        if shuffle:
            np.random.shuffle(chips)
        for i in range(0, len(chips), CHUNK):
            cs = np.sort(chips[i:i + CHUNK])
            sel = rows[np.isin(self.t["chip"][rows], cs)]
            xc = torch.from_numpy(self.photo(cs)).to(DEVICE)
            mc = torch.from_numpy(np.asarray(self.M[cs])).to(DEVICE)
            local = np.searchsorted(cs, self.t["chip"][sel])
            yield sel, xc, mc, local

    def crops(self, xc, mc, local, sel, normalise: bool = True):
        """Cut the 64 x 64 window around each tile (on the GPU) and normalise it."""
        ci = torch.as_tensor(local, device=DEVICE)
        r0 = torch.as_tensor(self.t["tr"][sel] * TILE, device=DEVICE)
        c0 = torch.as_tensor(self.t["tc"][sel] * TILE, device=DEVICE)
        ar = torch.arange(WIN, device=DEVICE)
        rr = (r0[:, None] + ar)[:, :, None]
        cc = (c0[:, None] + ar)[:, None, :]
        x = xc[ci[:, None, None, None], torch.arange(xc.shape[1], device=DEVICE)[None, :, None, None],
               rr[:, None], cc[:, None]].float()
        m = mc[ci[:, None, None], rr, cc]
        return ((x - self.mean) / self.std if normalise else x), m


def augment(x: torch.Tensor, m: torch.Tensor):
    if np.random.rand() < 0.5:
        x, m = x.flip(3), m.flip(2)
    if np.random.rand() < 0.5:
        x, m = x.flip(2), m.flip(1)
    k = int(np.random.randint(4))
    return torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (1, 2))


# ---------------------------------------------------------------- scoring
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s, kind="stable")
    y = y[o]
    return float(((np.cumsum(y) / np.arange(1, len(y) + 1)) * y).sum() / max(y.sum(), 1))


def prf(y, p) -> dict:
    tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
    pr, rc = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, tp=tp, fp=fp, fn=fn)


def best_threshold(y, s) -> float:
    cands = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, 500)))
    return float(cands[int(np.argmax([prf(y, s >= c)["f1"] for c in cands]))])


def evaluate(d: Data, rows: np.ndarray, s: np.ndarray, thr: float) -> dict:
    y, bg, nf, nh = (d.t[k][rows] for k in ("y", "background", "n_fire", "n_hot"))
    pred = s >= thr
    m = prf(y, pred)
    m.update(ap=average_precision(y, s), threshold=thr, tiles=int(len(y)), fire_tiles=int(y.sum()))
    m["by_size"] = {name: (float(pred[(nf >= lo) & (nf < hi)].mean()) if ((nf >= lo) & (nf < hi)).any() else None)
                    for name, lo, hi in SIZE_BINS}
    k = (bg == 1) & (y == 0)
    m["false_alarms_per_1000_km2"] = float(pred[k].sum() / max(k.sum() * TILE_KM2, 1e-9) * 1000)
    k = (bg == 0) & (y == 0) & (nh == 0)            # near fires, nothing hot: burned ground etc.
    m["burned_ground_flagged"] = float(pred[k].mean()) if k.any() else None
    return m


@torch.no_grad()
def predict(model, d: Data, rows: np.ndarray, kind: str, nb: int | None = None) -> np.ndarray:
    """Score rows; nb = only feed the first nb bands (a photo-only student on history data)."""
    model.eval()
    out = np.zeros(len(rows), np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    for sel, xc, mc, local in d.chunks(rows, shuffle=False):
        for i in range(0, len(sel), 1024):
            x, _ = d.crops(xc, mc, local[i:i + 1024], sel[i:i + 1024], normalise=kind != "pretrained")
            x = x[:, :nb] if nb else x
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                z = model(x)
            p = torch.sigmoid(z.float())
            if kind == "unet":
                p = p[:, 16:48, 16:48].amax(dim=(1, 2))
            out[[pos[r] for r in sel[i:i + 1024]]] = p.cpu().numpy()
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tile", "unet", "pretrained", "rule"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--tiny-weight", type=float, default=3.0,
                    help="loss weight for tiles with 1-4 fire px (5-24 px get half the extra); 1 = off")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="weight averaging: score/save a smoothed copy of the model; 0 = off")
    ap.add_argument("--history", action="store_true",
                    help="also give the model the clear 'before' picture (prepare_history.py); saved as <model>_hist")
    ap.add_argument("--teacher", default=None,
                    help="teacher-student: a before+now model (e.g. pretrained_hist) whose answers this photo-only "
                         "model also learns from; it needs no earlier photos when used. Saved as <model>_distil")
    ap.add_argument("--alpha", type=float, default=0.5, help="share of the loss from the teacher's answers")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    d = Data(history=args.history or bool(args.teacher))
    name = args.model + ("_hist" if args.history else "_distil" if args.teacher else "")
    nb = 7 if args.teacher else None                      # the student sees only the current photo's 7 bands
    teacher = None
    if args.teacher:
        tck = torch.load(OUT / f"model_{args.teacher}.pt", map_location=DEVICE, weights_only=False)
        tkind = tck["kind"]
        teacher = {"tile": TileNet, "unet": UNet, "pretrained": PreNet}[tkind](tck["bands"]).to(DEVICE).eval()
        teacher.load_state_dict(tck["state"])
        t_mean, t_std = tck["mean"].to(DEVICE), tck["std"].to(DEVICE)
        assert (tkind == "unet") == (args.model == "unet"), "pixel teacher for pixel student, tile for tile"
        print(f"teacher {args.teacher} ({tkind}, {tck['bands']} bands), alpha {args.alpha}", flush=True)
    tr, va, te = d.rows(0), d.rows(1), d.rows(2)
    print(f"{name}: {len(tr):,} train / {len(va):,} val / {len(te):,} test tiles, {d.bands} bands on {DEVICE}")

    if args.model == "rule":
        s = d.t["n_hot"].astype(np.float32)
        thr = best_threshold(d.t["y"][tr], s[tr])
        res = evaluate(d, te, s[te], thr)
    else:
        model = {"tile": TileNet, "unet": UNet, "pretrained": PreNet}[args.model](nb or d.bands).to(DEVICE)
        if args.model == "pretrained":   # gentle on the pre-trained body, normal speed for the new head
            opt = torch.optim.AdamW([{"params": model.body.parameters(), "lr": 2e-4},
                                     {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
        y_tr = d.t["y"][tr]
        tile_pw = torch.tensor(np.sqrt((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)), device=DEVICE)
        # Tiny fires are rare, so the model under-learns them: give their tiles extra weight.
        nf = d.t["n_fire"]
        tw = np.ones(len(nf), np.float32)
        tw[(nf >= 1) & (nf < 5)] = args.tiny_weight
        tw[(nf >= 5) & (nf < 25)] = 1 + (args.tiny_weight - 1) / 2
        # Weight averaging: a slowly-updated copy of the weights, steadier than the raw model.
        ema = None
        if args.ema > 0:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
        scored = ema.module if ema is not None else model
        print(f"tiny-fire weight {args.tiny_weight}, weight averaging {args.ema or 'off'}", flush=True)
        best, best_state = -1.0, None
        for ep in range(args.epochs):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for sel, xc, mc, local in d.chunks(tr, shuffle=True):
                order = np.random.permutation(len(sel))
                for i in range(0, len(order), BATCH):
                    b = order[i:i + BATCH]
                    if teacher is None:
                        x, m = augment(*d.crops(xc, mc, local[b], sel[b], normalise=args.model != "pretrained"))
                    else:                                  # same augmented crop for teacher and student
                        raw, m = augment(*d.crops(xc, mc, local[b], sel[b], normalise=False))
                        x = (raw if args.model == "pretrained" else (raw - d.mean) / d.std)[:, :nb]
                        with torch.no_grad(), torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                            soft = torch.sigmoid(teacher(raw if tkind == "pretrained"
                                                         else (raw - t_mean) / t_std).float())
                    with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                        z = model(x)
                    w = torch.as_tensor(tw[sel[b]], device=DEVICE)
                    a = args.alpha if teacher is not None else 0.0
                    if args.model in ("tile", "pretrained"):
                        y = torch.as_tensor(d.t["y"][sel[b]], device=DEVICE).float()
                        loss = F.binary_cross_entropy_with_logits(z.float(), y, weight=w, pos_weight=tile_pw)
                        if a:
                            loss = (1 - a) * loss + a * F.binary_cross_entropy_with_logits(
                                z.float(), soft, weight=w, pos_weight=tile_pw)
                    else:
                        valid = m != 255
                        target = (m == 1).float()
                        pw5 = torch.tensor(5.0, device=DEVICE)
                        lp = F.binary_cross_entropy_with_logits(z.float(), target, reduction="none", pos_weight=pw5)
                        if a:
                            lp = (1 - a) * lp + a * F.binary_cross_entropy_with_logits(
                                z.float(), soft, reduction="none", pos_weight=pw5)
                        loss = (lp * valid * w[:, None, None]).sum() / valid.sum().clamp(min=1)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    if ema is not None and ep >= 1:        # start averaging after a warm-up epoch
                        ema.update_parameters(model)
                    tot += loss.item() * len(b)
                    n += len(b)
            sched.step()
            cur = scored if ep >= 1 else model
            ap_val = average_precision(d.t["y"][va], predict(cur, d, va, args.model, nb))
            if ap_val > best:
                best, best_state = ap_val, {k: v.detach().cpu().clone() for k, v in cur.state_dict().items()}
            print(f"epoch {ep + 1:2d}  loss {tot / n:.4f}  val AP {ap_val:.3f}{'  *' if ap_val == best else ''}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        model = scored
        model.load_state_dict(best_state)
        s_va = predict(model, d, va, args.model, nb)
        thr = best_threshold(d.t["y"][va], s_va)
        s_te = predict(model, d, te, args.model, nb)
        res = evaluate(d, te, s_te, thr)
        OUT.mkdir(exist_ok=True)
        k = nb or d.bands                                  # a student keeps only the 7 current bands' scaling
        torch.save(dict(state=best_state, kind=args.model, mean=d.mean[:, :k].cpu(), std=d.std[:, :k].cpu(),
                        threshold=thr, bands=k, teacher=args.teacher, alpha=args.alpha if args.teacher else None),
                   OUT / f"model_{name}.pt")
        np.save(OUT / f"scores_{name}_test.npy", s_te)

    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    path = OUT / "metrics_v2.json"
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[name] = res
    path.write_text(json.dumps(allm, indent=2))


if __name__ == "__main__":
    main()
