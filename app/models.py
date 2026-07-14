from typing import List, Optional

from pydantic import BaseModel


class Product(BaseModel):
    id: str
    batchId: Optional[str] = None
    name: str
    normalizedName: Optional[str] = None
    price: float
    matchedCategory: Optional[str] = None
    matchedCategorySourceId: Optional[str] = None
    matchedCategoryScore: Optional[float] = None
    matchStatus: Optional[str] = None


class ExtractResponse(BaseModel):
    batchId: str
    products: List[Product]

class TextExtractRequest(BaseModel):
    text: str    
    