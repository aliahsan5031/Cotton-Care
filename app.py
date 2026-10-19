"""CottonCare: cotton leaf disease diagnosis for farmers, batch field/lot reports for industry."""
import os
import tempfile
from datetime import datetime

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Free Hugging Face Gradio Spaces run on ZeroGPU, which refuses to start unless the
# app contains at least one @spaces.GPU function. This model is small, so inference
# stays on CPU (no GPU quota used). This function is never called; it only satisfies
# the startup check. Locally the `spaces` package does not exist, so we skip it.
try:
    import spaces

    @spaces.GPU
    def _gpu_startup_check():
        return True
except ImportError:
    pass

from core import (LOW_CONFIDENCE, MIN_MARGIN, NOT_COTTON, NOT_COTTON_TEXT, llm_advice, lot_risk,
                  status_for)
import gate

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
GATE_PATH = os.environ.get("GATE_PATH", "gate_stats.pt")
_model = None
_classes = None
_tf = None
_gate = None
_feat = {}


def load_model():
    """Lazy-load so the UI still opens (with a clear message) if model.pt is missing."""
    global _model, _classes, _tf, _gate
    if _model is not None:
        return
    import torch
    from torchvision import models, transforms

    if not os.path.exists(MODEL_PATH):
        raise gr.Error(f"{MODEL_PATH} not found. Train with train.py and upload model.pt next to app.py.")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(ckpt["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    m.avgpool.register_forward_hook(lambda mod, i, o: _feat.__setitem__("f", o.flatten(1).detach()))
    if os.path.exists(GATE_PATH):
        _gate = torch.load(GATE_PATH, map_location="cpu")
    _model, _classes = m, ckpt["classes"]
    _tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def analyze(img: Image.Image, k: int = 3) -> dict:
    """One pass: top-k predictions, margin between the top two, cotton-leaf gate, photo quality."""
    import torch

    load_model()
    img = img.convert("RGB")
    x = _tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    top = probs.topk(min(k, len(_classes)))
    top = [(_classes[i], float(p)) for p, i in zip(top.values, top.indices)]
    margin = top[0][1] - top[1][1] if len(top) > 1 else 1.0
    g = gate.check(img, _feat["f"][0], _gate)
    return {"top": top, "margin": margin, "gate": g, "quality": gate.photo_quality(img), "img": img}


def pretty(name: str) -> str:
    return name.replace("_", " ").strip().title().replace("Fussarium", "Fusarium")


def final_status(a: dict) -> str:
    """Not a cotton leaf > Uncertain (low confidence or torn between classes) > Healthy / Problem."""
    if not a["gate"]["ok"]:
        return NOT_COTTON
    label, conf = a["top"][0]
    return status_for(label, conf, a["margin"])


def heatmap(img: Image.Image, class_name: str) -> Image.Image:
    """Grad-CAM: where on the leaf the model looked for the shown class (a sanity check, not proof)."""
    import numpy as np
    import torch
    from torchvision import transforms

    acts, grads = {}, {}

    def fwd(mod, inp, out):
        acts["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    h = _model.layer4.register_forward_hook(fwd)
    try:
        view = transforms.CenterCrop(224)(transforms.Resize(256)(img.convert("RGB")))
        with torch.enable_grad():
            out = _model(_tf(img.convert("RGB")).unsqueeze(0))
            _model.zero_grad()
            out[0, _classes.index(class_name)].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    finally:
        h.remove()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)
    rgba = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    over = 0.55 * np.asarray(view, dtype=np.float32) / 255.0 + 0.45 * rgba
    return Image.fromarray((np.clip(over, 0, 1) * 255).astype("uint8"))


def diagnose(img, lang):
    if img is None:
        raise gr.Error("Please upload or take a photo of a cotton leaf first.")
    lang_code = "ur" if lang.startswith("اردو") else "en"
    a = analyze(img)
    label, conf = a["top"][0]
    status = final_status(a)
    quality = a["quality"]
    qnote = f"\n\n📷 Photo is {'; '.join(quality)}. A clearer photo gives a more reliable result." if quality else ""

    if status == NOT_COTTON:
        head = (f"### 🚫 Not a cotton leaf\nNo diagnosis given because {a['gate']['reason']}." + qnote)
        return head, None, NOT_COTTON_TEXT[lang_code], "CottonCare only analyses cotton leaves.", None

    advice, source = llm_advice(label, conf if status != "Uncertain" else 0.0, lang_code)
    head = f"### {pretty(label)}\n**Status:** {status}  |  **Confidence:** {conf:.0%}"
    if conf < LOW_CONFIDENCE:
        head += f"\n\n⚠️ Confidence is below {LOW_CONFIDENCE:.0%}. Treat this as **uncertain** and get an expert to check."
    elif status == "Uncertain":
        head += (f"\n\n⚠️ The top two answers are within {MIN_MARGIN:.0%} of each other, so the model is torn. "
                 "Treat this as **uncertain** and get an expert to check.")
    head += qnote
    scores = {pretty(c): p for c, p in a["top"]}
    try:
        cam = heatmap(a["img"], label)
    except Exception:
        cam = None
    return head, scores, advice, f"Advice source: {source}. Not a substitute for an agronomist.", cam


def batch_report(files, farm, field, lot):
    if not files:
        raise gr.Error("Upload at least one leaf photo.")
    if len(files) > 30:
        raise gr.Error("Please upload at most 30 photos per batch.")
    rows = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        try:
            a = analyze(Image.open(path))
        except gr.Error:
            raise
        except Exception:
            rows.append({"photo": os.path.basename(path), "prediction": "unreadable image", "confidence": 0.0,
                         "status": "Uncertain", "photo_quality": ""})
            continue
        label, conf = a["top"][0]
        status = final_status(a)
        rows.append({
            "photo": os.path.basename(path),
            "prediction": "-" if status == NOT_COTTON else pretty(label),
            "confidence": round(conf, 3),
            "status": status,
            "photo_quality": "; ".join(a["quality"]) or "ok",
        })
    df = pd.DataFrame(rows)
    level, share, note = lot_risk(df)
    meta = {"farm": farm or "-", "field": field or "-", "lot": lot or "-", "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    out = df.copy()
    for k, v in reversed(list(meta.items())):
        out.insert(0, k, v)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="cottoncare_report_")
    out.to_csv(tmp.name, index=False)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    counts = df[~df["status"].isin(["Uncertain", NOT_COTTON])]["prediction"].value_counts()
    if len(counts):
        counts.plot.bar(ax=ax, color="#1f7a5c")
    ax.set_ylabel("Photos")
    ax.set_xlabel("")
    ax.set_title("Detected condition (confident cotton photos only)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(level, "⚪")
    blurry = int((df["photo_quality"].fillna("") .isin(["", "ok"]) == False).sum()) if "photo_quality" in df else 0
    summary = (
        f"### {icon} Field/lot risk: {level}\n"
        f"**Farm:** {meta['farm']} | **Field:** {meta['field']} | **Lot:** {meta['lot']}\n\n{note}\n\n"
        + (f"📷 {blurry} photo(s) have quality problems (blur/light). Retake them for a better report.\n\n" if blurry else "")
        + "_Risk is a simple rule of thumb on this sample, not a validated quality measure._"
    )
    return df, fig, summary, tmp.name


with gr.Blocks(title="CottonCare") as demo:
    gr.Markdown(
        "# 🌿 CottonCare\nAI diagnosis of cotton leaf disease for farmers, with batch field and lot reports "
        "for ginneries, mills and extension programs."
    )
    with gr.Tab("Diagnose a leaf"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Cotton leaf photo (daylight, leaf fills the frame)", sources=["upload", "webcam"])
                lang = gr.Radio(["English", "اردو (Urdu)"], value="English", label="Advice language")
                btn = gr.Button("Diagnose", variant="primary")
            with gr.Column():
                head = gr.Markdown()
                scores = gr.Label(label="Top predictions", num_top_classes=3)
                advice = gr.Textbox(label="What to do", lines=6, interactive=False)
                src = gr.Markdown()
                with gr.Accordion("Why this result? (where the model looked)", open=False):
                    cam = gr.Image(label="Heatmap: red/yellow = areas that influenced the answer most", interactive=False)
        btn.click(diagnose, [img, lang], [head, scores, advice, src, cam])
    with gr.Tab("Batch field / lot report"):
        with gr.Row():
            farm = gr.Textbox(label="Farm / supplier name")
            field = gr.Textbox(label="Field ID")
            lot = gr.Textbox(label="Lot / batch ID")
        files = gr.File(label="Upload up to 30 leaf photos", file_count="multiple", file_types=["image"], type="filepath")
        bbtn = gr.Button("Generate report", variant="primary")
        summary = gr.Markdown()
        with gr.Row():
            table = gr.Dataframe(label="Per-photo results")
            plot = gr.Plot(label="Distribution")
        csv = gr.File(label="Download CSV report")
        bbtn.click(batch_report, [files, farm, field, lot], [table, plot, summary, csv])
    gr.Markdown(
        "**Limits:** works on **cotton leaves only**; photos of other plants or objects are rejected by a cotton-leaf check. "
        "Trained on a small public dataset (CC BY 4.0). It is most reliable for healthy leaves and bacterial blight, "
        "and **weak for curl virus and fusarium wilt** (very few training photos). Accuracy can also drop on blurry or real-field photos. "
        "Advice is general guidance only. Always confirm with a local agriculture officer."
    )

if __name__ == "__main__":
    demo.launch()
