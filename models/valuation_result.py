from pydantic import BaseModel
from typing import Optional

class NetProfitHistory(BaseModel):
    year: int
    value: float

class AverageGrowth(BaseModel):
    ttm: float
    threeYears: float
    fiveYears: float

class StockInfo(BaseModel):
    name: str
    ticker: str
    exchange: str
    price: float
    country: str
    capitalization: float
    sector: str
    industry: str
    epsTtm: float
    peTtm: float
    dividendYield: float

class Valuation(BaseModel):
    fairPrice: float
    resultPercent: float
    resultLabel: str
    formula: str
    explanation: str
    netProfitHistory:list[NetProfitHistory]
    avgGrowth: AverageGrowth

class ValuationResult(BaseModel):
    stockInfo:StockInfo
    valuation: Optional[Valuation] = None
    