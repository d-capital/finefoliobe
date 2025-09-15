from pydantic import BaseModel
from typing import Optional

class ScreenerResult(BaseModel):
    id: Optional[int] = None
    company: Optional[str] = None
    ticker: Optional[str] = None
    result: Optional[str] = None
    fairPrice: Optional[float] = None
    pegTtm: Optional[float] = None
    netIncomeGrowthTtm: Optional[float] = None
    netIncomeGrowth3y: Optional[float] = None
    netIncomeGrowth5y: Optional[float] = None
    netIncomeGrowthNext1y: Optional[float] = None
    netIncomeGrowthNext3y: Optional[float] = None
    freeCashFlow: Optional[float] = None
    deFy: Optional[float] = None
    dividendYield: Optional[float] = None
    peTtm: Optional[float] = None
    forwardPe: Optional[float] = None
    price: Optional[float] = None
    type: Optional[str] = None
    marketCap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None