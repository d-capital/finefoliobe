from pydantic import BaseModel
from typing import Optional

class ScreenerFilter(BaseModel):
    maxPe: Optional[float] = 25
    minDividend: Optional[float] = 0