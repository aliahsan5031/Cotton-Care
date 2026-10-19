# CottonCare 🌿

AI screening of cotton leaf health for farmers, with batch field/lot reports for ginneries, mills and extension programs.
Built for the NYLP AI Hackathon 2026.

- **Live demo:** https://huggingface.co/spaces/AliAhsan5031/cotton_care
- **Demo video:** <add link>
- **Source code:** https://github.com/aliahsan5031/Cotton-Care
- **Team:** Ali Ahsan (solo): data, model training, app, deployment and documentation

## Problem
Cotton diseases and pests are often spotted late, and buyers have no cheap way to check crop health before sourcing.
<Add 2-3 sentences with a real, cited statistic about cotton in Pakistan.>

**Who benefits:** smallholder farmers (a quick check with advice in English and Urdu) and industry users (ginneries, textile mills, contract-farming and extension programs) who need a fast field/lot health overview.

## Solution
1. **Diagnose:** upload a leaf photo and get the predicted condition, a confidence score and plain-language advice (English or Urdu).
2. **Responsible by design:** below 60% confidence the app says "uncertain, ask an expert" instead of guessing. The footer and advice state that the model is weak for curl virus and fusarium wilt.
3. **Batch field/lot report:** upload up to 30 photos, tag the farm, field and lot, and get a results table, a distribution chart, a rule-based Low/Medium/High risk level and a CSV download.

## Features
- Single-photo diagnosis with the top-3 predictions
- **Cotton-leaf gate:** photos of other plants or objects are rejected ("Not a cotton leaf") instead of being forced into a cotton class
- **"Why this result?" heatmap** (Grad-CAM) showing where the model looked
- **Photo-quality hints** (blur, too dark, glare) so users retake bad photos
- English and Urdu advice
- Low-confidence and "torn between two answers" safeguards
- Batch mode with CSV export, per-photo quality column, and a lot risk level (not-cotton and uncertain photos are excluded from the risk)

## Tech stack
Python, PyTorch/torchvision (ResNet18), Gradio, pandas, matplotlib, Hugging Face Spaces.

## AI / API usage
- **Trained model (the core AI):** ResNet18 pretrained on ImageNet, fine-tuned on four classes (healthy, bacterial blight, fusarium wilt, curl virus). Training uses class-balanced sampling (square-root weighting), label smoothing and augmentation. The best epoch is chosen on validation macro-F1, and results are reported on a held-out test split. See `train.py` and `metrics.json`.
- **Cotton-leaf gate (`gate.py`, `make_gate_stats.py`):** a closed-set classifier must pick one of its classes even for a tomato leaf or a shoe. After training we store the feature fingerprint of the cotton training photos (`gate_stats.pt`: class means and a shared covariance). At prediction time a photo whose feature distance (Mahalanobis) to every class is above a threshold chosen on the validation split is rejected, and a colour check rejects photos with almost no plant colours. Results: <fill from gate_report.json>.
- **Advice text:** rule-based, built into `core.py` (no pesticide names or doses; it always tells the user to confirm with an agriculture officer).
- **Gemini API (optional, third-party):** `core.py` can rewrite the advice with the Gemini API if a `GEMINI_API_KEY` is set. **The deployed demo does not enable it**, so all advice shown there comes from the built-in guide.
- **Lot risk level:** a simple rule of thumb (Medium at 10% or more and High at 30% or more of confidently-flagged photos), not a trained or validated quality model.
- An AI coding assistant (Claude) was used as development help. All code was run, tested and reviewed by the author.

## Dataset and credit
**Cotton Leaf Disease Dataset with Severity Levels**, Mendeley Data, V2, DOI: 10.17632/74jsdxtmx2.2, published 9 January 2025. Authors: <copy from the dataset page's citation box>.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
**Changes made:** only the `Cotton-Original` images were used (not the augmented sets); severity sub-folders were merged into their disease class; images were resized and normalized; and the data was split 70/15/15 (684 train, 147 validation, 149 test). The dataset authors have not endorsed this project.

## Results (held-out test split, 149 images)
| Class | Test images | Recall | Precision |
|---|---|---|---|
| Healthy | 119 | 97.5% | 93.5% |
| Bacterial blight | 18 | 88.9% | 100% |
| Fusarium wilt | 8 | 50.0% | 44.4% |
| Curl virus | 4 | **0%** | n/a |

Overall accuracy 91.3%, macro-F1 0.59, mean per-class recall 59%. Accuracy looks high mainly because about 80% of the test images are healthy, so the macro-F1 and per-class numbers are the fairer picture. Training images per class: healthy 552, bacterial blight 83, fusarium wilt 32, curl virus 17.

## Model weights
`model.pt` (about 45 MB) is larger than GitHub's browser upload limit, so the weights are stored in the Hugging Face Space repository:
https://huggingface.co/spaces/AliAhsan5031/cotton_care/tree/main
To run locally, download `model.pt` from there and put it next to `app.py`, or retrain with `train.py`.

## Setup
```bash
git clone https://github.com/aliahsan5031/Cotton-Care
cd Cotton-Care
pip install -r requirements.txt
# put model.pt next to app.py (see "Model weights"), or train your own:
#   python train.py --data <folder-with-one-subfolder-per-class> --epochs 15 --sampling sqrt --out model.pt
# Optional: enable Gemini rewriting (never commit the key)
export GEMINI_API_KEY=your_key_here
python app.py
```

## Screenshots
<add: diagnosis screen, uncertain result, batch report, CSV>

## Known limitations
- **Curl virus is not detected** in the held-out test (0 of 4) and fusarium wilt is only partly detected, because the dataset has very few photos of these (17 and 32 training images). A "healthy" result is not a guarantee, and the app says so.
- The cotton-leaf gate is a distance threshold, not a guarantee: unusual real cotton photos may be rejected, and a very similar non-cotton leaf may still pass. It was checked on <N> non-cotton images only.
- The test sets for the rare classes are tiny (4 and 8 images), so their scores are noisy.
- Dataset photos are cleaner than most real-field photos, so real-world accuracy is likely lower. This was not tested on field photos.
- Advice is general guidance, not a substitute for an agronomist. The Urdu text should be reviewed by a native speaker.
- The lot risk level is a simple rule on the uploaded sample, not a statistically valid field estimate.

## Team contributions
- Ali Ahsan: dataset selection and licensing check, model training and evaluation, app, deployment, documentation.

## Acknowledgements
Dataset authors (see above), PyTorch/torchvision, Gradio, Hugging Face Spaces.
