"""Build gate_stats.pt: the 'what a cotton leaf looks like' fingerprint used by the cotton-leaf gate.

Run it in Colab right after training, on the SAME data folder and seed as train.py:

    python make_gate_stats.py --data "/content/orig/Cotton-Original" --model model.pt

Optional: put a few NON-cotton photos (tomato leaf, maize leaf, a hand, a shoe...) in a folder and pass
--other that_folder to see how many of them get rejected (good number for your README).

The threshold is chosen on the VALIDATION split only. The test split is only used to report how many
real cotton photos would wrongly be rejected.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--out", default="gate_stats.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--other", default=None, help="folder of non-cotton images to measure the rejection rate")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    grab = {}
    model.avgpool.register_forward_hook(lambda m, i, o: grab.__setitem__("f", o.flatten(1).detach().cpu()))

    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(args.data, transform=eval_tf)
    assert ds.classes == classes, f"class folders {ds.classes} differ from the model's {classes}"

    # Same stratified 70/15/15 split as train.py (same seed, same order of shuffling).
    random.seed(args.seed)
    by_class = {}
    for idx, (_, y) in enumerate(ds.samples):
        by_class.setdefault(y, []).append(idx)
    tr, va, te = [], [], []
    for idxs in by_class.values():
        random.shuffle(idxs)
        a, b = int(0.7 * len(idxs)), int(0.85 * len(idxs))
        tr += idxs[:a]; va += idxs[a:b]; te += idxs[b:]

    @torch.no_grad()
    def feats_of(dataset):
        out, ys = [], []
        for x, y in DataLoader(dataset, batch_size=args.batch, num_workers=2):
            model(x.to(device))
            out.append(grab["f"]); ys += y.tolist()
        return torch.cat(out), torch.tensor(ys)

    ftr, ytr = feats_of(Subset(ds, tr))
    fva, _ = feats_of(Subset(ds, va))
    fte, _ = feats_of(Subset(ds, te))

    means = torch.stack([ftr[ytr == k].mean(0) for k in range(len(classes))])
    centred = ftr - means[ytr]
    cov = centred.T @ centred / max(len(ftr) - 1, 1)
    d = cov.shape[0]
    shrink = 0.3  # few images per class -> blend with a plain diagonal for a stable inverse
    cov = (1 - shrink) * cov + shrink * (torch.trace(cov) / d) * torch.eye(d)
    precision = torch.linalg.inv(cov.double()).float()

    def score(f):
        dd = f.unsqueeze(1) - means.unsqueeze(0)                 # (N, K, D)
        m = torch.einsum("nkd,de,nke->nk", dd, precision, dd)
        return m.min(dim=1).values

    s_tr, s_va, s_te = score(ftr), score(fva), score(fte)
    thr = float(max(np.percentile(s_va.numpy(), 97), np.percentile(s_tr.numpy(), 99)) * 1.15)
    report = {
        "threshold": thr,
        "train_accepted": float((s_tr <= thr).float().mean()),
        "val_accepted": float((s_va <= thr).float().mean()),
        "test_cotton_accepted": float((s_te <= thr).float().mean()),
    }

    if args.other:
        paths = [os.path.join(r, f) for r, _, fs in os.walk(args.other) for f in fs
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        scores = []
        for p in paths:
            try:
                x = eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            except Exception:
                continue
            with torch.no_grad():
                model(x)
            scores.append(float(score(grab["f"])[0]))
        if scores:
            report["non_cotton_images"] = len(scores)
            report["non_cotton_rejected"] = float(np.mean([s > thr for s in scores]))

    torch.save({"means": means, "precision": precision, "threshold": thr, "classes": classes}, args.out)
    with open("gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("saved", args.out, "and gate_report.json  -> upload gate_stats.pt next to app.py")


if __name__ == "__main__":
    main()
