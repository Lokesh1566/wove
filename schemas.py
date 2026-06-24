"""Request and response models."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ColorSwatch(BaseModel):
    hex: str
    ratio: float


class TagResponse(BaseModel):
    category: str
    category_confidence: float
    styles: List[str]
    colors: List[ColorSwatch] = []
    embedding: List[float]
    backend: str = Field(description="clip:ViT-B-32 when ML is installed, otherwise mock")


class Item(BaseModel):
    id: str
    category: str
    embedding: List[float] = []
    colors: List[ColorSwatch] = []


class OutfitRequest(BaseModel):
    items: List[Item]
    max_outfits: int = 5
