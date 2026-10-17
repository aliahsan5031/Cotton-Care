# CottonCare 🌿

AI diagnosis of cotton leaf disease for farmers, with batch field/lot reports for ginneries, mills and extension programs.


**Live demo:** <add Hugging Face Space link>
**Demo video:** <add link>
**Team:** <name(s) and roles>

## Problem
Cotton diseases are often spotted late, and buyers have no cheap way to check crop health before sourcing.
<Add 2-3 sentences with a real, cited statistic about cotton in Pakistan.>

**Who benefits:** smallholder farmers (diagnosis and advice in Urdu/English) and industry users (ginneries, textile mills, contract-farming and extension programs) who need a quick field/lot health overview.

## Solution
1. **Diagnose:** upload a leaf photo, get the condition, a confidence score and plain-language advice.
2. **Responsible by design:** below 60% confidence the app says "uncertain, get an expert" instead of guessing.
3. **Batch field/lot report (industry):** upload up to 30 photos, tag farm/field/lot, get a results table, distribution chart, a rule-based Low/Medium/High risk level and a CSV export.

## Features
- Single-photo diagnosis with top-3 predictions
- Urdu and English advice
- Low-confidence safeguard
- Batch mode with CSV export and lot risk level

## Tech stack
Python, PyTorch/torchvision (ResNet18), Gradio, pandas, matplotlib, Hugging Face Spaces.

## AI / API usage
- **Trained model (the core AI):** ResNet18 pretrained on ImageNet, fine-tuned by us on `<dataset name + link + license>`.
  Stratified 70/15/15 train/val/test split, seed 42. Best epoch chosen on validation, results reported on the untouched test set.
  **Test accuracy: <fill from metrics.json>** (per-class accuracy and confusion matrix in `metrics.json`).
- **Gemini API (optional, third-party):** rewrites the built-in advice into short steps. If the key is missing or the call fails, the app falls back to the built-in rule-based guide. No pesticide names or doses are generated.
- **Rule-based:** the lot risk level (Medium >= 10%, High >= 30% of confidently-diseased photos) is a simple rule of thumb, not a trained or validated quality model.
- AI coding assistants were used as development help; all code was tested and reviewed by the team.

## Setup
```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
# Train (Colab GPU recommended). Dataset folders: data/<class_name>/*.jpg
python train.py --data data --epochs 8 --out model.pt
# Optional: enable Gemini rewriting (never commit the key)
export GEMINI_API_KEY=your_key_here
python app.py
```
On Hugging Face Spaces, add `GEMINI_API_KEY` under Settings > Secrets and upload `model.pt` next to `app.py`.

## Screenshots
<add: diagnosis screen, uncertain case, batch report, CSV>

## Known limitations
- Trained on a public dataset; accuracy can drop on blurry, shaded or real-field photos.
- Class list depends on the dataset, so unsupported diseases are not detected.
- Advice is general guidance, not a substitute for an agronomist. The Urdu text should be reviewed by a native speaker.
- Lot risk is a simple rule on the uploaded sample, not a statistically valid field estimate.

## Team contributions
- <Name>: <role and what they built>

## Acknowledgements
<dataset, libraries, pretrained weights>
