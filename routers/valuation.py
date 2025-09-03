from fastapi import APIRouter, HTTPException
from models.valuation_result import ValuationResult
from services.valuation_service import get_valuation

router = APIRouter()

@router.get("/{exchange}/{ticker}", response_model=ValuationResult)
def valuate_stock(exchange: str, ticker:str):
    result = get_valuation(exchange, ticker)
    if not result:
        raise HTTPException(status_code=404, detail=f"Valuation failed for stock with ticker {ticker} on {exchange} exchange.")
    return result
