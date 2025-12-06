from pydantic import BaseModel
from typing import Optional

class NetProfitHistory(BaseModel):
    year: Optional[int] = None
    value: Optional[float] = None

class AverageGrowth(BaseModel):
    ttm: Optional[float] = None
    threeYears: Optional[float] = None
    fiveYears: Optional[float] = None

class StockInfo(BaseModel):
    name: str
    ticker: str
    exchange: str
    price: Optional[float] = None
    country: Optional[str] = None
    capitalization: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    sectorRu: Optional[str] = None
    industryRu: Optional[str] = None
    epsTtm: Optional[float] = None
    peTtm: Optional[float] = None
    dividendYield: Optional[float] = None
    freeCashFlow: Optional[float] = None
    debtToEquity: Optional[float] = None

class Valuation(BaseModel):
    fairPrice: Optional[float] = None
    resultPercent: Optional[float] = None
    resultLabel: Optional[str] = None
    formula: Optional[str] = None
    explanation: Optional[str] = None
    netProfitHistory:Optional[list[NetProfitHistory]] = None
    avgGrowth: Optional[AverageGrowth] = None
    peg: Optional[float] = None

class ValuationResult(BaseModel):
    stockInfo:StockInfo
    valuation: Optional[Valuation] = None
    