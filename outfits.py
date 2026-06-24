"""Outfit generation.

Pairs items across wardrobe slots (top, bottom, shoes, optional layer) and
scores each combination by embedding similarity. This is the scaffold version.
Swap the scoring for a learned compatibility model + FAISS retrieval later.
"""
import numpy as np

TOPS = {"t-shirt", "shirt", "blouse", "sweater", "hoodie"}
LAYERS = {"jacket", "coat"}
BOTTOMS = {"jeans", "trousers", "shorts", "skirt"}
SHOES = {"sneakers", "boots", "heels"}
ONE_PIECE = {"dress"}


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(a @ b / denom)


def generate_outfits(items, max_outfits: int = 5):
    pick = lambda slot: [i for i in items if i.get("category") in slot]
    tops, bottoms, shoes = pick(TOPS), pick(BOTTOMS), pick(SHOES)
    dresses = pick(ONE_PIECE)

    combos = []

    for top in tops:
        for bottom in bottoms:
            for shoe in (shoes or [None]):
                score = _cosine(top.get("embedding"), bottom.get("embedding"))
                combos.append({
                    "top": top.get("id"),
                    "bottom": bottom.get("id"),
                    "shoes": shoe.get("id") if shoe else None,
                    "score": round(score, 3),
                })

    for dress in dresses:
        for shoe in (shoes or [None]):
            combos.append({
                "dress": dress.get("id"),
                "shoes": shoe.get("id") if shoe else None,
                "score": 0.5,
            })

    combos.sort(key=lambda c: c["score"], reverse=True)
    return combos[:max_outfits]
