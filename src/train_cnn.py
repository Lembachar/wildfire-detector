"""Step 5: a small CNN on the three SWIR/NIR bands of each 32 x 32 tile.

Design choices:
  - Global MAX pooling at the end: a fire is a few hot pixels, so "is there a hot spot anywhere"
    matters more than the tile's average.
  - Flips and 90-degree rotations as augmentation: fire has no preferred orientation.
  - Positive class weighted by sqrt(neg/pos) because fire tiles are rare.
  - The epoch and threshold are picked on val; test is scored once at the end.
"""
import numpy as np
import torch
import torch.nn as nn

from common import (OUT, average_precision, best_threshold, false_alarm_rate, load_tiles,
                    recall_by_size, report, save_metrics)

torch.manual_seed(0)
np.random.seed(0)
EPOCHS = 30
BATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FireNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.features = nn.Sequential(block(3, 32), block(32, 32), nn.MaxPool2d(2),
                                      block(32, 64), block(64, 64), nn.MaxPool2d(2),
                                      block(64, 128))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).amax(dim=(2, 3))).squeeze(1)


def augment(x: torch.Tensor) -> torch.Tensor:
    if np.random.rand() < 0.5:
        x = x.flip(3)
    if np.random.rand() < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(np.random.randint(4)), dims=(2, 3))


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([torch.sigmoid(model(X[i:i + 4096])) for i in range(0, len(X), 4096)]).cpu().numpy()


def main() -> None:
    d = load_tiles()
    X = np.clip(d["X"], 0, 2)
    y, split = d["y"], d["split"]
    mean = X[split == 0].mean(axis=(0, 2, 3), keepdims=True)
    std = X[split == 0].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    # The whole dataset (~1.4 GB as float32) fits on the 8 GB GPU, so it lives there.
    Xt = torch.from_numpy((X - mean) / std).float().to(DEVICE)
    yt = torch.from_numpy(y).float().to(DEVICE)
    tr, va, te = (np.where(split == s)[0] for s in range(3))

    pos = y[tr].sum()
    pos_weight = torch.tensor(np.sqrt((len(tr) - pos) / max(pos, 1)), dtype=torch.float32, device=DEVICE)
    model = FireNet().to(DEVICE)
    print(f"training on {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"train {len(tr):,} tiles ({pos} fire), pos_weight {pos_weight:.1f}")

    best_ap, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = torch.as_tensor(perm[i:i + BATCH], device=DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(augment(Xt[idx])), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        ap = average_precision(y[va], predict(model, Xt[va]))
        if ap > best_ap:
            best_ap, best_state = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:2d}  loss {total / len(tr):.4f}  val AP {ap:.3f}{'  *' if ap == best_ap else ''}")

    model.load_state_dict(best_state)
    scores = predict(model, Xt)
    thr = best_threshold(y[va], scores[va])
    report("val   CNN", y[va], scores[va], thr)
    test = report("TEST  CNN", y[te], scores[te], thr)
    test["by_size"] = recall_by_size(y[te], d["n_hot"][te], (scores[te] >= thr).astype(int))
    print("recall by fire size:", {k: v for k, v in test["by_size"].items()})
    test["background"] = false_alarm_rate(d, split == 2, (scores >= thr).astype(int))
    print("false alarms on fire-free land:", test["background"])
    save_metrics("cnn", test)
    OUT.mkdir(exist_ok=True)
    torch.save({"state": best_state, "mean": mean, "std": std, "threshold": thr}, OUT / "firenet.pt")
    np.save(OUT / "cnn_scores.npy", scores)


if __name__ == "__main__":
    main()
