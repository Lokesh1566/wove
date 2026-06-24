"""Item tagging.

Uses OpenCLIP zero-shot classification when torch + open_clip are installed.
Falls back to a deterministic mock so the API boots and responds with no ML
dependencies at all. Install backend/requirements-ml.txt to switch to real CLIP.
"""
import hashlib

import numpy as np
from PIL import Image

CATEGORY_LABELS = [
    "t-shirt", "shirt", "blouse", "sweater", "hoodie", "jacket", "coat",
    "dress", "jeans", "trousers", "shorts", "skirt",
    "sneakers", "boots", "heels", "bag", "hat", "scarf",
]
STYLE_LABELS = [
    "casual", "formal", "streetwear", "minimal",
    "athletic", "bohemian", "vintage", "business",
]

_model = None
_preprocess = None
_tokenizer = None
_device = "cpu"


def _try_load() -> bool:
    """Load OpenCLIP once. Returns False if the ML stack is not available."""
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return True
    try:
        import open_clip
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval().to(_device)
        return True
    except Exception:
        return False


def _zero_shot(img: Image.Image, labels, prefix: str):
    import torch

    text = _tokenizer([f"{prefix} {label}" for label in labels]).to(_device)
    image = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        image_features = _model.encode_image(image)
        text_features = _model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    return image_features[0].cpu().numpy(), probs.cpu().numpy()


def tag_image(img: Image.Image) -> dict:
    img = img.convert("RGB")

    if _try_load():
        embedding, cat_probs = _zero_shot(img, CATEGORY_LABELS, "a photo of a")
        _, style_probs = _zero_shot(img, STYLE_LABELS, "a photo of a")
        best = int(cat_probs.argmax())
        top_styles = [STYLE_LABELS[i] for i in style_probs.argsort()[::-1][:2]]
        return {
            "category": CATEGORY_LABELS[best],
            "category_confidence": round(float(cat_probs[best]), 3),
            "styles": top_styles,
            "embedding": embedding.astype(float).tolist(),
            "backend": "clip:ViT-B-32",
        }

    # Deterministic mock: same image always yields the same tags.
    seed = int(hashlib.sha256(img.tobytes()[:4096]).hexdigest(), 16)
    category = CATEGORY_LABELS[seed % len(CATEGORY_LABELS)]
    styles = list(dict.fromkeys([
        STYLE_LABELS[seed % len(STYLE_LABELS)],
        STYLE_LABELS[(seed // 7) % len(STYLE_LABELS)],
    ]))
    rng = np.random.default_rng(seed % (2 ** 32))
    vec = rng.normal(size=64)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return {
        "category": category,
        "category_confidence": 0.0,
        "styles": styles,
        "embedding": vec.astype(float).tolist(),
        "backend": "mock",
    }
