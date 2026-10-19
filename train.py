"""Fine-tune a pretrained ResNet18 on a cotton-leaf disease image dataset.

Dataset layout (one folder per class, folder name = class name):
    data/
      healthy/ img1.jpg ...
      bacterial_blight/ ...
      curl_virus/ ...

Run (Colab with GPU, or CPU for a quick test):
    python train.py --data data --epochs 8 --out model.pt

Outputs: model.pt (weights + class names) and metrics.json (put these numbers
in your README: test accuracy, per-class accuracy, confusion matrix).

IMPORTANT: if the dataset ships pre-augmented copies of the same photo, a random
split leaks near-duplicates across train/test and inflates accuracy. Use the
original (non-augmented) images, or the dataset's own train/test folders.
"""
import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per class (the TRAIN folder if --test is given)")
    ap.add_argument("--test", default=None, help="optional separate TEST folder with the same class names; if given, it is used as the untouched test set")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sampling", choices=["balanced", "sqrt", "none"], default="balanced",
                    help="balanced = equal chance per class; sqrt = halfway (less over-repeating of tiny classes); none = natural")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke test only)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.25, 0.25, 0.25),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Two views of the same folder so train gets augmentation, val/test do not.
    full_train = datasets.ImageFolder(args.data, transform=train_tf)
    full_eval = datasets.ImageFolder(args.data, transform=eval_tf)
    classes = full_train.classes
    n = len(full_train)
    assert n > 0 and len(classes) >= 2, "Need at least 2 class folders with images"

    by_class = {}
    for idx, (_, y) in enumerate(full_train.samples):
        by_class.setdefault(y, []).append(idx)
    tr, va, te = [], [], []
    if args.test:
        # Dataset ships its own test folder: split TRAIN 85/15 into train/val, keep TEST untouched.
        test_eval = datasets.ImageFolder(args.test, transform=eval_tf)
        assert test_eval.classes == classes, (
            f"Class folders differ between train and test.\ntrain: {classes}\ntest:  {test_eval.classes}\n"
            "Rename the test folders so the names match exactly."
        )
        for idxs in by_class.values():
            random.shuffle(idxs)
            a = int(0.85 * len(idxs))
            tr += idxs[:a]
            va += idxs[a:]
        test_set = test_eval
    else:
        # Single folder: stratified 70/15/15 split, fixed seed so results are reproducible.
        for idxs in by_class.values():
            random.shuffle(idxs)
            a, b = int(0.7 * len(idxs)), int(0.85 * len(idxs))
            tr += idxs[:a]
            va += idxs[a:b]
            te += idxs[b:]
        test_set = Subset(full_eval, te)
    print(f"classes: {classes}")
    print(f"train {len(tr)} | val {len(va)} | test {len(test_set)}")

    kw = dict(batch_size=args.batch, num_workers=args.workers)
    ys_all = np.array([y for _, y in full_train.samples])
    counts = np.bincount(ys_all[tr], minlength=len(classes))
    print("train images per class:", dict(zip(classes, counts.tolist())))
    if args.sampling == "none":
        train_dl = DataLoader(Subset(full_train, tr), shuffle=True, **kw)
    else:
        # Rare classes are sampled more often so the model does not just learn "healthy".
        power = 1.0 if args.sampling == "balanced" else 0.5
        w = [1.0 / (counts[ys_all[i]] ** power) for i in tr]
        sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
        train_dl = DataLoader(Subset(full_train, tr), sampler=sampler, **kw)
    val_dl = DataLoader(Subset(full_eval, va), **kw)
    test_dl = DataLoader(test_set, **kw)

    model = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    @torch.no_grad()
    def evaluate(dl):
        model.eval()
        preds, ys = [], []
        for x, y in dl:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
        return np.array(preds), np.array(ys)

    def scores(p, y, k):
        cm = np.zeros((k, k), dtype=int)
        for t, q in zip(y, p):
            cm[t, q] += 1
        tp = np.diag(cm).astype(float)
        rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
        prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
        present = cm.sum(1) > 0
        return cm, prec, rec, f1, float(f1[present].mean()), float(rec[present].mean())

    best_val, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        p, y = evaluate(val_dl)
        val_acc = float((p == y).mean()) if len(y) else 0.0
        val_f1 = scores(p, y, len(classes))[4]
        print(f"epoch {ep}/{args.epochs} loss {total / len(tr):.4f} val_acc {val_acc:.4f} val_macro_f1 {val_f1:.4f}")
        if val_f1 > best_val:  # choose the epoch on VAL macro-F1 (not accuracy, which hides rare classes), never on test
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    p, y = evaluate(test_dl)
    k = len(classes)
    cm, prec, rec, f1, macro_f1, bal_acc = scores(p, y, k)
    test_counts = cm.sum(1).tolist()
    for c, n in zip(classes, test_counts):
        if n < 10:
            print(f"WARNING: only {n} test images for '{c}', its score is very noisy. Say so in the README.")
    metrics = {
        "classes": classes,
        "n_train": len(tr), "n_val": len(va), "n_test": len(test_set),
        "train_images_per_class": dict(zip(classes, counts.tolist())),
        "test_images_per_class": dict(zip(classes, test_counts)),
        "sampling": args.sampling,
        "label_smoothing": args.label_smoothing,
        "epochs": args.epochs,
        "best_val_macro_f1": best_val,
        "test_accuracy": float((p == y).mean()),
        "test_balanced_accuracy_mean_recall": bal_acc,
        "test_macro_f1": macro_f1,
        "per_class_test_recall": dict(zip(classes, [float(v) for v in rec])),
        "per_class_test_precision": dict(zip(classes, [float(v) for v in prec])),
        "per_class_test_f1": dict(zip(classes, [float(v) for v in f1])),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    }
    print(json.dumps(metrics, indent=2))
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"arch": "resnet18", "classes": classes, "state_dict": best_state}, args.out)
    print("saved", args.out, "and metrics.json")


if __name__ == "__main__":
    main()
