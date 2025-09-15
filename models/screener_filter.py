from pydantic import BaseModel
from typing import Optional

class ScreenerFilter(BaseModel):
    minPe: Optional[float] = None
    maxPe: Optional[float] = None
    minDividend: Optional[float] = None
    sector: Optional[str] = None