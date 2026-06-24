"""Dominant color extraction. Uses PIL median-cut quantization, no extra deps."""
from collections import Counter

from PIL import Image


def dominant_colors(img: Image.Image, k: int = 4):
    img = img.convert("RGB").resize((128, 128))
    quantized = img.quantize(colors=k, method=Image.MEDIANCUT)
    palette = quantized.getpalette()
    counts = Counter(quantized.getdata())
    total = sum(counts.values()) or 1

    swatches = []
    for index, count in counts.most_common(k):
        r, g, b = palette[index * 3:index * 3 + 3]
        swatches.append({
            "hex": f"#{r:02X}{g:02X}{b:02X}",
            "ratio": round(count / total, 3),
        })
    return swatches
