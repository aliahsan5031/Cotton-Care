"""Pure-Python logic for CottonCare (no torch needed, easy to test)."""
from __future__ import annotations

import os

import pandas as pd
import requests

# Below this confidence the app refuses to give a firm diagnosis.
LOW_CONFIDENCE = 0.60
# Lot risk thresholds: share of confidently-diseased samples.
RISK_MEDIUM = 0.10
RISK_HIGH = 0.30
MIN_SAMPLES = 5

# General, non-prescriptive guidance keyed by words found in the class name.
# Class names depend on the dataset you train on, so matching is by keyword.
# No pesticide names or doses on purpose: those must come from an agronomist.
ADVICE = {
    "healthy": {
        "en": "The leaf looks healthy. Keep monitoring the field regularly, especially after rain or humid weather. This tool is weaker at spotting curl virus and fusarium wilt, so if you see curled, twisted or wilting leaves, ask an agriculture officer to check.",
        "ur": "پتا صحت مند نظر آتا ہے۔ کھیت کی باقاعدہ نگرانی جاری رکھیں، خاص طور پر بارش یا نمی کے بعد۔ یہ ٹول کرل وائرس اور فیوزیریم وِلٹ پہچاننے میں کمزور ہے، اس لیے اگر پتے مڑے ہوئے، بل کھائے ہوئے یا مرجھائے ہوں تو زرعی افسر سے معائنہ کروائیں۔",
    },
    "bacterial": {
        "en": "Possible bacterial disease. Remove and destroy badly affected leaves, avoid working in wet fields, avoid overhead irrigation, and ask a local agriculture officer which treatment is approved in your area.",
        "ur": "ممکنہ بیکٹیریا سے ہونے والی بیماری۔ شدید متاثرہ پتے ہٹا کر تلف کریں، گیلے کھیت میں کام نہ کریں، اوپر سے پانی دینے سے بچیں، اور اپنے علاقے میں منظور شدہ علاج کے بارے میں زرعی افسر سے پوچھیں۔",
    },
    "curl": {
        "en": "Possible leaf curl virus, usually spread by whiteflies. Remove heavily infected plants, control weeds around the field, monitor whiteflies, and contact an agriculture officer for the recommended control plan.",
        "ur": "ممکنہ لیف کرل وائرس، جو عموماً سفید مکھی سے پھیلتا ہے۔ شدید متاثرہ پودے ہٹائیں، کھیت کے گرد جھاڑیاں صاف رکھیں، سفید مکھی پر نظر رکھیں، اور تجویز کردہ کنٹرول پلان کے لیے زرعی افسر سے رابطہ کریں۔",
    },
    "wilt": {
        "en": "Possible wilt disease, which is soil-borne. Avoid moving soil or water from this spot to other fields, rotate crops next season, and ask an agriculture officer about soil testing and resistant varieties.",
        "ur": "ممکنہ مرجھانے (وِلٹ) کی بیماری، جو مٹی سے پھیلتی ہے۔ اس جگہ کی مٹی یا پانی دوسرے کھیتوں میں نہ لے جائیں، اگلے سیزن میں فصل بدلیں، اور مٹی کے ٹیسٹ اور مزاحم اقسام کے بارے میں زرعی افسر سے پوچھیں۔",
    },
    "herbicide": {
        "en": "Possible herbicide damage, for example from spray drift. Check for recent spraying nearby, avoid spraying in windy conditions, keep a record of what was applied, and ask an agriculture officer whether the crop is likely to recover.",
        "ur": "ممکنہ جڑی بوٹی مار دوا سے نقصان، مثلاً سپرے کے ہوا میں بہنے سے۔ آس پاس حالیہ سپرے کی جانچ کریں، تیز ہوا میں سپرے نہ کریں، جو دوا استعمال ہوئی اس کا ریکارڈ رکھیں، اور زرعی افسر سے پوچھیں کہ فصل ٹھیک ہونے کا امکان ہے یا نہیں۔",
    },
    "hopper": {
        "en": "Possible leaf hopper (jassid) pest damage. Check the underside of leaves for small green insects, see how widespread it is in the field, and ask an agriculture officer for the locally approved pest-control plan.",
        "ur": "ممکنہ تیلے (جیسڈ) کیڑے کا نقصان۔ پتوں کے نیچے چھوٹے سبز کیڑوں کو دیکھیں، جانچیں کہ کھیت میں کتنا پھیلا ہے، اور مقامی طور پر منظور شدہ کیڑوں کے کنٹرول کے لیے زرعی افسر سے رابطہ کریں۔",
    },
    "redden": {
        "en": "Leaf reddening can be linked to nutrient stress, water stress or pests. Check soil moisture and irrigation, and ask an agriculture officer about a soil or leaf nutrient test before adding anything.",
        "ur": "پتوں کا سرخ ہونا غذائی کمی، پانی کی کمی یا کیڑوں سے منسلک ہو سکتا ہے۔ مٹی کی نمی اور آبپاشی دیکھیں، اور کچھ ڈالنے سے پہلے مٹی یا پتوں کے غذائی ٹیسٹ کے بارے میں زرعی افسر سے پوچھیں۔",
    },
    "variegat": {
        "en": "Patchy leaf colors (variegation) can have several causes, such as viruses, nutrient problems or pests. Take clearer photos of several plants and ask an agriculture officer to inspect the field.",
        "ur": "پتوں پر رنگوں کے دھبے (ویریگیشن) کی کئی وجوہات ہو سکتی ہیں، جیسے وائرس، غذائی مسائل یا کیڑے۔ کئی پودوں کی واضح تصاویر لیں اور زرعی افسر سے کھیت کا معائنہ کروائیں۔",
    },
    "aphid": {
        "en": "Possible aphid infestation. Check the underside of young leaves and shoots for small clustered insects, see how widespread it is, encourage natural predators such as ladybirds, and ask an agriculture officer for the locally approved control plan.",
        "ur": "ممکنہ اَفِڈ (تیلے) کا حملہ۔ نئے پتوں اور شاخوں کے نیچے چھوٹے گچھوں میں کیڑے دیکھیں، جانچیں کہ کتنا پھیلا ہے، قدرتی دشمن کیڑوں جیسے لیڈی برڈ کو بچائیں، اور مقامی طور پر منظور شدہ کنٹرول کے لیے زرعی افسر سے پوچھیں۔",
    },
    "mildew": {
        "en": "Possible powdery mildew, a fungal disease that shows as a white powdery coating. Improve air flow between plants, avoid over-crowding and excess nitrogen, remove badly affected leaves, and ask an agriculture officer which treatment is approved locally.",
        "ur": "ممکنہ پاؤڈری ملڈیو، ایک پھپھوندی کی بیماری جس میں سفید پاؤڈر جیسی تہہ نظر آتی ہے۔ پودوں کے درمیان ہوا کا گزر بہتر کریں، گنجان کاشت اور زیادہ نائٹروجن سے بچیں، شدید متاثرہ پتے ہٹائیں، اور مقامی طور پر منظور شدہ علاج کے لیے زرعی افسر سے پوچھیں۔",
    },
    "worm": {
        "en": "Possible armyworm or caterpillar damage. Look for chewed leaves, droppings and young larvae on the plants (often early morning or evening), check how much of the field is affected, and ask an agriculture officer for the locally approved control plan.",
        "ur": "ممکنہ فوجی سنڈی یا سنڈی کا نقصان۔ کٹے ہوئے پتے، فضلہ اور چھوٹی سنڈیاں دیکھیں (اکثر صبح سویرے یا شام کو)، جانچیں کہ کھیت کا کتنا حصہ متاثر ہے، اور مقامی طور پر منظور شدہ کنٹرول کے لیے زرعی افسر سے رابطہ کریں۔",
    },
    "spot": {
        "en": "Possible fungal leaf spot. Improve air flow between plants, avoid leaving leaves wet for long, remove badly spotted leaves, and check with an agriculture officer before applying any treatment.",
        "ur": "ممکنہ پھپھوندی سے پتوں پر دھبے۔ پودوں کے درمیان ہوا کا گزر بہتر کریں، پتوں کو دیر تک گیلا نہ رہنے دیں، زیادہ دھبوں والے پتے ہٹائیں، اور کوئی علاج کرنے سے پہلے زرعی افسر سے مشورہ کریں۔",
    },
}
GENERIC = {
    "en": "A possible problem was detected. Take a clearer photo of the leaf in daylight and ask a local agriculture officer to confirm before treating.",
    "ur": "کسی ممکنہ مسئلے کی نشاندہی ہوئی ہے۔ دن کی روشنی میں پتے کی واضح تصویر لیں اور علاج سے پہلے زرعی افسر سے تصدیق کروائیں۔",
}
UNCERTAIN = {
    "en": "The model is not confident about this photo. Retake it in daylight with the leaf filling most of the frame, or ask an expert to check it.",
    "ur": "ماڈل کو اس تصویر پر یقین نہیں ہے۔ دن کی روشنی میں پتے کو فریم میں بھر کر دوبارہ تصویر لیں، یا کسی ماہر سے جانچ کروائیں۔",
}


HEALTHY_WORDS = ("healthy", "fresh", "normal")


def is_healthy(label: str) -> bool:
    low = label.lower()
    return any(w in low for w in HEALTHY_WORDS) and "diseased" not in low


def static_advice(label: str, confidence: float, lang: str = "en") -> str:
    """Rule-based advice. Always works offline."""
    lang = lang if lang in ("en", "ur") else "en"
    if confidence < LOW_CONFIDENCE:
        return UNCERTAIN[lang]
    if is_healthy(label):
        return ADVICE["healthy"][lang]
    low = label.lower()
    for key, texts in ADVICE.items():
        if key in low:
            return texts[lang]
    return GENERIC[lang]


def llm_advice(label: str, confidence: float, lang: str = "en") -> tuple[str, str]:
    """Optionally rewrite the advice with Gemini. Returns (text, source).

    Falls back to the static advice on any problem (no key, network, quota).
    Set GEMINI_API_KEY as a secret, never in the repo. GEMINI_MODEL can override
    the model name if the default is retired.
    """
    base = static_advice(label, confidence, lang)
    key = os.environ.get("GEMINI_API_KEY")
    if not key or confidence < LOW_CONFIDENCE:
        return base, "built-in guide"
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    language = "Urdu" if lang == "ur" else "English"
    prompt = (
        f"You help smallholder cotton farmers. A model detected '{label}' on a cotton leaf "
        f"with {confidence:.0%} confidence. Using only this guidance: \"{static_advice(label, confidence, 'en')}\" "
        f"write 3 short, simple steps in {language}. Do not name pesticides or doses. "
        "End by telling the farmer to confirm with a local agriculture officer. Max 80 words."
    )
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=20,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return (text or base), "Gemini API (rewrites the built-in guide)"
    except Exception:
        return base, "built-in guide (Gemini unavailable)"


def status_for(label: str, confidence: float) -> str:
    if confidence < LOW_CONFIDENCE:
        return "Uncertain"
    return "Healthy" if is_healthy(label) else "Problem detected"


def lot_risk(df: pd.DataFrame) -> tuple[str, float | None, str]:
    """Rule-based sourcing/field risk from a batch table with a 'status' column.

    This is a simple rule of thumb, NOT a validated quality model.
    """
    n = len(df)
    certain = df[df["status"] != "Uncertain"]
    uncertain = n - len(certain)
    if len(certain) < MIN_SAMPLES:
        return (
            "Not enough data",
            None,
            f"Only {len(certain)} confident samples. Add at least {MIN_SAMPLES} clear leaf photos for a risk score.",
        )
    share = float((certain["status"] == "Problem detected").mean())
    level = "High" if share >= RISK_HIGH else "Medium" if share >= RISK_MEDIUM else "Low"
    note = (
        f"{share:.0%} of {len(certain)} confident samples show a problem"
        f" ({uncertain} uncertain photo(s) excluded). Thresholds: Medium >= {RISK_MEDIUM:.0%}, High >= {RISK_HIGH:.0%}."
    )
    return level, share, note
