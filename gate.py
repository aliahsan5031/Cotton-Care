"""Cotton-leaf gate: stops the app from confidently labelling photos that are not cotton leaves.

A classifier trained on 4 cotton classes MUST pick one of them, even for a tomato leaf or a shoe.
This module adds two independent checks before any diagnosis is shown:

1. Feature-distance check (calibrated). During training we store the typical "feature fingerprint"
   of real cotton-leaf photos (gate_stats.pt, made by make_gate_stats.py). A new photo whose
   fingerprint is far from all of them is rejected as "not a cotton leaf".
2. Colour check (always on). Photos with almost no plant colours (faces, screens, objects)
   are rejected without needing the model.

Also: simple photo-quality hints (blur, too dark, too bright) so farmers retake bad photos.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

MIN_GREEN_FRACTION = 0.04   # below this there is basically no vegetation colour in the photo
BLUR_VAR = 60.0             # Laplacian variance (on a 256 px grey image) below this = blurry
DARK_MEAN, BRIGHT_MEAN = 45.0, 215.0


def plant_colour_fraction(img: Image.Image) -> float:
    """Share of pixels that look like green/yellow-green vegetation (hue 40-170 deg)."""
    small = img.convert("RGB").resize((128, 128))
    hsv = np.asarray(small.convert("HSV")).astype(np.float32)
    h, s, v = hsv[..., 0] * 360.0 / 255.0, hsv[..., 1], hsv[..., 2]
    mask = (h >= 40) & (h <= 170) & (s >= 45) & (v >= 35)
    return float(mask.mean())


def photo_quality(img: Image.Image) -> list[str]:
    """Plain-language hints; empty list means the photo looks fine."""
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    notes = []
    if float(lap.var()) < BLUR_VAR:
        notes.append("blurry (hold the phone steady and tap to focus)")
    mean = float(g.mean())
    if mean < DARK_MEAN:
        notes.append("too dark (use daylight)")
    elif mean > BRIGHT_MEAN:
        notes.append("too bright or overexposed (avoid direct glare)")
    return notes


def distance_score(feat, stats: dict) -> float:
    """Smallest Mahalanobis-style distance from the photo's features to any class centre."""
    import torch

    d = feat.unsqueeze(0) - stats["means"]                   # (K, D)
    m = ((d @ stats["precision"]) * d).sum(dim=1)            # (K,)
    return float(torch.clamp(m, min=0).min())


def check(img: Image.Image, feat=None, stats: dict | None = None) -> dict:
    """Return {ok, reason, score, mode}. ok=False means 'do not show a cotton diagnosis'."""
    green = plant_colour_fraction(img)
    if green < MIN_GREEN_FRACTION:
        return {"ok": False, "reason": "no leaf colours found in the photo", "score": None,
                "mode": "calibrated" if stats else "basic"}
    if stats is not None and feat is not None:
        score = distance_score(feat, stats)
        thr = float(stats["threshold"])
        if score > thr:
            return {"ok": False, "reason": "the leaf does not look like the cotton leaves the model was trained on",
                    "score": score, "mode": "calibrated"}
        return {"ok": True, "reason": "", "score": score, "mode": "calibrated"}
    return {"ok": True, "reason": "", "score": None, "mode": "basic"}
