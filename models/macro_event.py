from pydantic import BaseModel
from typing import Optional


class MacroEvent(BaseModel):
        id: int
        dateline: int
        date: str
        actual_formatted: str
        actual: float
        forecast_formatted: Optional[str]
        forecast: Optional[float]
        revision_formatted: str = ""
        revision: float = None
        is_active: bool
        is_most_recent: bool