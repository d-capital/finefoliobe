from fastapi import APIRouter, HTTPException
from fastapi_cache import JsonCoder
from models.valuation_result import ValuationResult
from services.valuation_service import get_valuation, get_dcf_valuation
from fastapi_cache.decorator import cache

router = APIRouter()

@router.get("/{exchange}/{ticker}", response_model=ValuationResult)
@cache(expire=60, coder=JsonCoder)
def valuate_stock(exchange: str, ticker:str):
    result = get_valuation(exchange, ticker)
    if not result:
        raise HTTPException(status_code=404, detail=f"Valuation failed for stock with ticker {ticker} on {exchange} exchange.")
    return result

@router.get("/dcf/{exchange}/{ticker}", response_model=float)
def dcf_valuation(exchange: str, ticker:str):
    result = get_dcf_valuation(exchange, ticker)
    if not result:
        raise HTTPException(status_code=404, detail=f"Valuation failed for stock with ticker {ticker} on {exchange} exchange.")
    return result