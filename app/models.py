from typing import List

from pydantic import BaseModel


class Product(BaseModel):
    name: str
    price: float


class ExtractResponse(BaseModel):
    products: List[Product]