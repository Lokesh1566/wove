# ML notes

## Tagging
- Model: OpenCLIP `ViT-B-32`, pretrained `laion2b_s34b_b79k`.
- Zero-shot over `CATEGORY_LABELS` and `STYLE_LABELS` in `app/tagging.py`.
- Prompts use `"a photo of a {label}"`. Tune these prompts first when accuracy
  is off, before reaching for a different model.
- A deterministic mock returns the same tags for the same image when torch is
  not installed, so the API and frontend are always demoable.

## Embeddings
- Real path returns the 512-dim CLIP image embedding.
- Mock path returns a 64-dim seeded vector. Outfit scoring handles either, it
  only compares vectors of equal shape.

## Next steps
1. Build a FAISS index over item embeddings for fast nearest-neighbor pairing.
2. Replace the heuristic outfit scorer with a learned compatibility head
   trained on outfit data (positive pairs = real outfits, negatives = random).
3. Add a fit/occasion classifier so suggestions respect formality and weather.
4. Calibrate category confidence and add a low-confidence "needs review" flag.
