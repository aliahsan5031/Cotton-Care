"""CottonCare: cotton leaf disease diagnosis for farmers, batch field/lot reports for industry."""
import inspect
import os
import re
import tempfile
from datetime import datetime
from html import escape as esc

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


# ----------------------------------------------------------------------------------------------
# Model + analysis (unchanged logic)
# ----------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------
# HTML building blocks for the interface
# ----------------------------------------------------------------------------------------------
def _rich(text: str) -> str:
    """Escape text, keep **bold** and line breaks (advice may come back from Gemini)."""
    t = esc(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return t.replace("\n", "<br>")


def _meter(conf: float) -> str:
    return (
        f'<div class="cc-conf"><span class="cc-conf-num">{conf:.0%}</span>'
        f'<span class="cc-conf-label">confidence</span></div>'
        f'<div class="cc-meter" role="img" aria-label="Confidence {conf:.0%}">'
        f'<div class="cc-meter-fill" style="width:{conf * 100:.1f}%"></div>'
        f'<div class="cc-meter-tick" style="left:{LOW_CONFIDENCE * 100:.0f}%"></div></div>'
        f'<p class="cc-meter-note">The marker is {LOW_CONFIDENCE:.0%}, the minimum for a firm answer.</p>'
    )


def _quality_notice(quality) -> str:
    if not quality:
        return ""
    return (f'<p class="cc-notice">Photo is {esc("; ".join(quality))}. '
            "A clearer photo gives a more reliable result.</p>")


def _advice_card(text: str, lang_code: str) -> str:
    cls = "cc-advice-text cc-ur" if lang_code == "ur" else "cc-advice-text"
    attrs = ' dir="rtl" lang="ur"' if lang_code == "ur" else ' lang="en"'
    return (f'<section class="cc-panel"><h3 class="cc-h3">What to do</h3>'
            f'<div class="{cls}"{attrs}>{_rich(text)}</div></section>')


def _scores_card(top) -> str:
    rows = "".join(
        f'<div class="cc-pred"><span class="cc-pred-name">{esc(pretty(c))}</span>'
        f'<span class="cc-pred-bar"><span style="width:{p * 100:.1f}%"></span></span>'
        f'<span class="cc-pred-pct">{p:.0%}</span></div>'
        for c, p in top
    )
    return f'<section class="cc-panel"><h3 class="cc-h3">Closest matches</h3>{rows}</section>'


def _source_line(text: str) -> str:
    return f'<p class="cc-source">{esc(text)}</p>'


def _gate_debug(a: dict) -> str:
    """Shown only when the Space variable GATE_DEBUG=1, to help tune the cotton-leaf gate."""
    if os.environ.get("GATE_DEBUG") != "1":
        return ""
    g = a["gate"]
    verdict = "passed" if g["ok"] else "rejected"
    if g.get("score") is None or _gate is None:
        why = "" if g["ok"] else f" ({g['reason']})"
        return f" Gate ({g['mode']}): {verdict}{why}."
    thr = float(_gate["threshold"]) * float(os.environ.get("GATE_SCALE", "1"))
    return f" Gate ({g['mode']}): score {g['score']:.1f}, limit {thr:.1f}, {verdict}."


EMPTY_VERDICT = (
    '<div class="cc-card cc-empty"><h2 class="cc-title cc-title-quiet">Your result will appear here</h2>'
    '<p class="cc-body">Add a photo of one cotton leaf, then press Diagnose leaf.</p></div>'
)
EMPTY_BATCH = (
    '<div class="cc-card cc-empty"><h2 class="cc-title cc-title-quiet">Your field report will appear here</h2>'
    '<p class="cc-body">Upload leaf photos from one field or lot, then press Generate report.</p></div>'
)


# ----------------------------------------------------------------------------------------------
# Diagnose tab
# ----------------------------------------------------------------------------------------------
def diagnose(img, lang):
    if img is None:
        raise gr.Error("Add a photo of a cotton leaf first.")
    lang_code = "ur" if lang.startswith("اردو") else "en"
    a = analyze(img)
    label, conf = a["top"][0]
    status = final_status(a)
    notice = _quality_notice(a["quality"])

    if status == NOT_COTTON:
        card = (
            '<div class="cc-card cc-stop"><span class="cc-pill">No diagnosis</span>'
            '<h2 class="cc-title">Not a cotton leaf</h2>'
            f'<p class="cc-body">No diagnosis given because {esc(a["gate"]["reason"])}.</p>{notice}</div>'
        )
        return (card, "", _advice_card(NOT_COTTON_TEXT[lang_code], lang_code),
                _source_line("CottonCare only analyses cotton leaves." + _gate_debug(a)), None)

    advice, source = llm_advice(label, conf if status != "Uncertain" else 0.0, lang_code)
    kind = {"Healthy": "ok", "Problem detected": "bad"}.get(status, "warn")
    warn = ""
    if conf < LOW_CONFIDENCE:
        warn = (f'<p class="cc-notice">Confidence is below {LOW_CONFIDENCE:.0%}. '
                "Treat this as uncertain and ask an expert to check.</p>")
    elif status == "Uncertain":
        warn = (f'<p class="cc-notice">The top two answers are within {MIN_MARGIN:.0%} of each other, '
                "so the model cannot pick between them. Treat this as uncertain and ask an expert to check.</p>")
    card = (
        f'<div class="cc-card cc-{kind}"><span class="cc-pill">{esc(status)}</span>'
        f'<h2 class="cc-title">{esc(pretty(label))}</h2>{_meter(conf)}{warn}{notice}</div>'
    )
    try:
        cam = heatmap(a["img"], label)
    except Exception:
        cam = None
    return (card, _scores_card(a["top"]), _advice_card(advice, lang_code),
            _source_line(f"Advice source: {source}. Not a substitute for an agronomist." + _gate_debug(a)), cam)


# ----------------------------------------------------------------------------------------------
# Batch tab
# ----------------------------------------------------------------------------------------------
def batch_report(files, farm, field, lot):
    if not files:
        raise gr.Error("Upload at least one leaf photo.")
    if len(files) > 30:
        raise gr.Error("Upload at most 30 photos per batch.")
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
        counts.plot.bar(ax=ax, color="#1E4D3A", width=0.6)
    ax.set_ylabel("Photos")
    ax.set_xlabel("")
    ax.set_title("Conditions found (confident cotton photos only)", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    kind = {"Low": "ok", "Medium": "warn", "High": "bad"}.get(level, "stop")
    title = "Not enough data" if level == "Not enough data" else f"{level} field/lot risk"
    blurry = int((df["photo_quality"].fillna("").isin(["", "ok"]) == False).sum()) if "photo_quality" in df else 0
    quality = (f'<p class="cc-notice">{blurry} photo(s) have quality problems (blur or light). '
               "Retake them for a better report.</p>") if blurry else ""
    summary = (
        f'<div class="cc-card cc-{kind}"><span class="cc-pill">Field/lot risk</span>'
        f'<h2 class="cc-title">{esc(title)}</h2>'
        f'<dl class="cc-meta"><div><dt>Farm</dt><dd>{esc(meta["farm"])}</dd></div>'
        f'<div><dt>Field</dt><dd>{esc(meta["field"])}</dd></div>'
        f'<div><dt>Lot</dt><dd>{esc(meta["lot"])}</dd></div>'
        f'<div><dt>Date</dt><dd>{esc(meta["date"])}</dd></div></dl>'
        f'<p class="cc-body">{esc(note)}</p>{quality}'
        '<p class="cc-source">Risk is a simple rule of thumb on this sample, not a validated quality measure.</p></div>'
    )
    return df, fig, summary, tmp.name


# ----------------------------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------------------------
# Palette: field green (brand), leaf green (accent), mist (page), and one signal colour per outcome.
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EEF5EF", c100="#DCEBDF", c200="#B9D7C0", c300="#8FBC9B", c400="#5E9C70", c500="#3F8F5B",
        c600="#2F7D4F", c700="#1E4D3A", c800="#173D2E", c900="#112E23", c950="#0B1F18",
    ),
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#F1F5F0",
    body_text_color="#17261F",
    block_background_fill="#FFFFFF",
    block_border_color="#D5DFD6",
    block_shadow="none",
    button_primary_background_fill="#1E4D3A",
    button_primary_background_fill_hover="#173D2E",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#1E4D3A",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

:root { color-scheme: light; --cc-ink:#17261F; --cc-green:#1E4D3A; --cc-line:#D5DFD6; --cc-muted:#4C5F55; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
:focus-visible { outline: 3px solid #1E4D3A !important; outline-offset: 2px; }

/* header */
.cc-head { display:flex; align-items:center; gap:14px; padding:8px 2px 4px; }
.cc-mark { flex:none; width:44px; height:44px; border-radius:12px; background:#1E4D3A; display:grid; place-items:center; }
.cc-head h1 { margin:0; font-size:clamp(1.5rem,4vw,2rem); line-height:1.1; font-weight:700; color:var(--cc-green); }
.cc-head p { margin:4px 0 0; color:var(--cc-muted); font-size:1.02rem; max-width:60ch; }

/* tabs and controls */
button[role="tab"] { font-weight:700 !important; font-size:1.02rem !important; }
button[role="tab"][aria-selected="true"] { color:var(--cc-green) !important; border-color:var(--cc-green) !important; }
.cc-btn { min-height:52px; font-size:1.08rem !important; font-weight:700 !important; }
.cc-tips { margin:2px 0 0; padding:0 0 0 18px; color:var(--cc-muted); font-size:.98rem; line-height:1.55; }
.cc-intro { margin:2px 0 4px; color:var(--cc-muted); font-size:1.02rem; max-width:70ch; }

/* result card: the one expressive element */
.cc-card { border:1px solid var(--cc-line); border-left-width:10px; border-radius:14px; padding:22px 24px; background:#fff; }
.cc-pill { display:inline-block; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.95rem; }
.cc-title { margin:10px 0 0; font-size:clamp(1.7rem,5vw,2.3rem); line-height:1.12; font-weight:700; letter-spacing:-.01em; }
.cc-title-quiet { font-size:1.25rem; margin-top:0; color:var(--cc-ink); letter-spacing:0; }
.cc-body { margin:10px 0 0; font-size:1.05rem; line-height:1.55; max-width:62ch; }
.cc-notice { margin:14px 0 0; padding:10px 12px; border-radius:10px; background:rgba(0,0,0,.05); font-size:.98rem; line-height:1.5; }
.cc-source { margin:12px 0 0; color:var(--cc-muted); font-size:.92rem; line-height:1.45; }

.cc-ok   { border-left-color:#2F7D4F; background:#F1F9F3; } .cc-ok .cc-pill   { background:#CFE8D6; color:#14442C; }
.cc-bad  { border-left-color:#A63D2F; background:#FCF1EE; } .cc-bad .cc-pill  { background:#F3D0C9; color:#6F2015; }
.cc-warn { border-left-color:#B07A00; background:#FEF6DD; } .cc-warn .cc-pill { background:#F6E3A4; color:#5E3F00; }
.cc-stop { border-left-color:#4A5A66; background:#EEF1F3; } .cc-stop .cc-pill { background:#D5DCE1; color:#26323B; }
.cc-empty { border-style:dashed; border-left-width:1px; background:transparent; }

/* confidence meter with the 60% marker */
.cc-conf { display:flex; align-items:baseline; gap:8px; margin-top:18px; }
.cc-conf-num { font-size:2rem; font-weight:700; line-height:1; }
.cc-conf-label { color:var(--cc-muted); font-size:1rem; }
.cc-meter { position:relative; height:14px; margin-top:8px; border-radius:7px; background:rgba(0,0,0,.09); }
.cc-meter-fill { height:100%; border-radius:7px; background:var(--cc-green); animation:cc-fill .8s ease-out; }
.cc-meter-tick { position:absolute; top:-5px; bottom:-5px; width:3px; margin-left:-1px; background:var(--cc-ink); border-radius:2px; }
.cc-meter-note { margin:8px 0 0; color:var(--cc-muted); font-size:.92rem; }
@keyframes cc-fill { from { width:0; } }

/* supporting panels */
.cc-panel { border:1px solid var(--cc-line); border-radius:12px; padding:16px 20px; background:#fff; margin-top:12px; }
.cc-h3 { margin:0 0 10px; font-size:1.05rem; font-weight:700; color:var(--cc-green); }
.cc-advice-text { font-size:1.1rem; line-height:1.6; max-width:62ch; }
.cc-ur { direction:rtl; text-align:right; font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq',serif; font-size:1.2rem; line-height:2.5; max-width:none; }
.cc-pred { display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,1fr) 3.2rem; gap:12px; align-items:center; padding:6px 0; }
.cc-pred-name { font-size:1.02rem; overflow-wrap:anywhere; }
.cc-pred-bar { height:10px; border-radius:5px; background:#E3E9E4; overflow:hidden; }
.cc-pred-bar span { display:block; height:100%; background:#5E9C70; border-radius:5px; }
.cc-pred-pct { text-align:right; font-weight:700; font-variant-numeric:tabular-nums; }

/* batch summary */
.cc-meta { display:flex; flex-wrap:wrap; gap:6px 28px; margin:14px 0 0; }
.cc-meta div { min-width:0; }
.cc-meta dt { color:var(--cc-muted); font-size:.9rem; }
.cc-meta dd { margin:0; font-weight:700; overflow-wrap:anywhere; }

/* limits */
.cc-limits { margin-top:8px; padding:18px 22px; border-radius:12px; background:#E6EDE6; }
.cc-limits h3 { margin:0 0 8px; font-size:1.05rem; font-weight:700; color:var(--cc-green); }
.cc-limits ul { margin:0; padding-left:20px; line-height:1.6; font-size:1rem; max-width:80ch; }

@media (max-width:640px) {
  .cc-card { padding:18px 16px; }
  .cc-pred { grid-template-columns:minmax(0,1fr) 3rem; }
  .cc-pred-bar { grid-column:1 / -1; grid-row:2; }
}
@media (prefers-reduced-motion: reduce) { .cc-meter-fill { animation:none; } }
"""

# Daylight is where this app is used, so keep the light theme even if the phone is in dark mode.
JS = """() => {
  const light = () => document.body.classList.remove('dark');
  light();
  new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']});
}"""

# Gradio 6 moved theme/css/js from Blocks() to launch(); Gradio 5 is the other way round.
_UI = {"theme": THEME, "css": CSS, "js": JS}
if "css" in inspect.signature(gr.Blocks.launch).parameters:
    BLOCKS_KW, LAUNCH_KW = {}, _UI
else:
    BLOCKS_KW, LAUNCH_KW = _UI, {}

HEADER = """
<header class="cc-head">
  <div class="cc-mark" aria-hidden="true">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M5 19c0-8 5-14 15-14 0 10-6 15-14 15"/><path d="M5 19 14 10"/>
    </svg>
  </div>
  <div>
    <h1>CottonCare</h1>
    <p>Check a cotton leaf for disease and get clear next steps in English or Urdu.</p>
  </div>
</header>
"""

LIMITS = """
<section class="cc-limits">
  <h3>What CottonCare cannot do</h3>
  <ul>
    <li>It works on <strong>cotton leaves only</strong>. Photos of other plants or objects are rejected by a cotton-leaf check.</li>
    <li>It is most reliable for healthy leaves and bacterial blight, and <strong>weak for curl virus and fusarium wilt</strong> because the training set has very few photos of them.</li>
    <li>Accuracy drops on blurry photos and on real-field photos.</li>
    <li>It was trained on a small public dataset (CC BY 4.0). Advice is general guidance. Always confirm with a local agriculture officer.</li>
  </ul>
</section>
"""

with gr.Blocks(title="CottonCare", **BLOCKS_KW) as demo:
    gr.HTML(HEADER)
    with gr.Tabs():
        with gr.Tab("Diagnose a leaf"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    img = gr.Image(type="pil", label="Leaf photo", sources=["upload", "webcam"])
                    gr.HTML('<ul class="cc-tips"><li>Photograph one leaf in daylight.</li>'
                            "<li>Fill most of the frame with the leaf.</li>"
                            "<li>Hold the phone steady and tap to focus.</li></ul>")
                    lang = gr.Radio(["English", "اردو (Urdu)"], value="English", label="Advice language")
                    btn = gr.Button("Diagnose leaf", variant="primary", elem_classes="cc-btn")
                with gr.Column(scale=6):
                    head = gr.HTML(EMPTY_VERDICT)
                    scores = gr.HTML()
                    advice = gr.HTML()
                    src = gr.HTML()
                    with gr.Accordion("Why this result? See where the model looked", open=False):
                        cam = gr.Image(label="Heatmap: red and yellow areas influenced the answer most",
                                       interactive=False)
            btn.click(diagnose, [img, lang], [head, scores, advice, src, cam])
        with gr.Tab("Batch field / lot report"):
            gr.HTML('<p class="cc-intro">Upload up to 30 leaf photos from one field or lot. '
                    "You get a risk summary, a per-photo table and a CSV for your records.</p>")
            with gr.Row():
                farm = gr.Textbox(label="Farm / supplier name")
                field = gr.Textbox(label="Field ID")
                lot = gr.Textbox(label="Lot / batch ID")
            files = gr.File(label="Leaf photos (up to 30)", file_count="multiple", file_types=["image"], type="filepath")
            bbtn = gr.Button("Generate report", variant="primary", elem_classes="cc-btn")
            summary = gr.HTML(EMPTY_BATCH)
            with gr.Row():
                table = gr.Dataframe(label="Results per photo")
                plot = gr.Plot(label="Distribution")
            csv = gr.File(label="Download CSV report")
            bbtn.click(batch_report, [files, farm, field, lot], [table, plot, summary, csv])
    gr.HTML(LIMITS)

if __name__ == "__main__":
    demo.launch(**LAUNCH_KW)
