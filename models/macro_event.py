from sqlmodel import SQLModel, Field
from typing import Optional


class MacroEvent(SQLModel, table=True):
        id: Optional[int] = Field(primary_key=True)
        country: str
        event_type: str
        dateline: Optional[int]
        date: Optional[str]
        actual_formatted: Optional[str]
        actual: float
        forecast_formatted: Optional[str]
        forecast: Optional[float]
        revision_formatted: Optional[str]
        revision: Optional[float]
        is_active: bool
        is_most_recent: bool