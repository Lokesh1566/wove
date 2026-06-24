"""Wove API. Tag wardrobe items and build outfits from what you already own."""
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .colors import dominant_colors
from .outfits import generate_outfits
from .schemas import OutfitRequest, TagResponse
from .tagging import tag_image

app = FastAPI(
    title="Wove API",
    version="0.1.0",
    description="Tag wardrobe items and build outfits from what you already own.",
)

# Open during development. Lock this to your deployed frontend origin before launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "wove"}


@app.post("/tag", response_model=TagResponse)
async def tag(file: UploadFile = File(...)):
    """Upload one clothing photo, get back its category, styles, colors and an embedding."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image.")
    result = tag_image(img)
    result["colors"] = dominant_colors(img)
    return result


@app.post("/outfits")
def outfits(req: OutfitRequest):
    """Given a list of tagged items, return ranked outfit combinations."""
    items = [i.model_dump() for i in req.items]
    return {"outfits": generate_outfits(items, max_outfits=req.max_outfits)}
