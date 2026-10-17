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

from core import LOW_CONFIDENCE, llm_advice, lot_risk, status_for

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
_model = None
_classes = None
_tf = None


def load_model():
    """Lazy-load so the UI still opens (with a clear message) if model.pt is missing."""
    global _model, _classes, _tf
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
    _model, _classes = m, ckpt["classes"]
    _tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(img: Image.Image, k: int = 3):
    import torch

    load_model()
    x = _tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    top = probs.topk(min(k, len(_classes)))
    return [(_classes[i], float(p)) for p, i in zip(top.values, top.indices)]


def pretty(name: str) -> str:
    return name.replace("_", " ").strip().title().replace("Fussarium", "Fusarium")


def diagnose(img, lang):
    if img is None:
        raise gr.Error("Please upload or take a photo of a cotton leaf first.")
    lang_code = "ur" if lang.startswith("اردو") else "en"
    top = predict(img)
    label, conf = top[0]
    status = status_for(label, conf)
    advice, source = llm_advice(label, conf, lang_code)
    headline = f"### {pretty(label)}\n**Status:** {status}  |  **Confidence:** {conf:.0%}"
    if conf < LOW_CONFIDENCE:
        headline += f"\n\n⚠️ Confidence is below {LOW_CONFIDENCE:.0%}. Treat this as **uncertain** and get an expert to check."
    scores = {pretty(c): p for c, p in top}
    return headline, scores, advice, f"Advice source: {source}. Not a substitute for an agronomist."


def batch_report(files, farm, field, lot):
    if not files:
        raise gr.Error("Upload at least one leaf photo.")
    if len(files) > 30:
        raise gr.Error("Please upload at most 30 photos per batch.")
    rows = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        try:
            label, conf = predict(Image.open(path))[0]
        except gr.Error:
            raise
        except Exception:
            rows.append({"photo": os.path.basename(path), "prediction": "unreadable image", "confidence": 0.0, "status": "Uncertain"})
            continue
        rows.append({
            "photo": os.path.basename(path),
            "prediction": pretty(label),
            "confidence": round(conf, 3),
            "status": status_for(label, conf),
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
    counts = df[df["status"] != "Uncertain"]["prediction"].value_counts()
    if len(counts):
        counts.plot.bar(ax=ax, color="#1f7a5c")
    ax.set_ylabel("Photos")
    ax.set_xlabel("")
    ax.set_title("Detected condition (confident photos only)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(level, "⚪")
    summary = (
        f"### {icon} Field/lot risk: {level}\n"
        f"**Farm:** {meta['farm']} | **Field:** {meta['field']} | **Lot:** {meta['lot']}\n\n{note}\n\n"
        "_Risk is a simple rule of thumb on this sample, not a validated quality measure._"
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
        btn.click(diagnose, [img, lang], [head, scores, advice, src])
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
        "**Limits:** trained on a small public dataset (CC BY 4.0). It is most reliable for healthy leaves and bacterial blight, "
        "and **weak for curl virus and fusarium wilt** (very few training photos). Accuracy can also drop on blurry or real-field photos. "
        "Advice is general guidance only. Always confirm with a local agriculture officer."
    )

if __name__ == "__main__":
    demo.launch()
