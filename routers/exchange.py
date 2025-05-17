from fastapi import APIRouter, HTTPException
from models.exchange import ExchangeRate
from services.exchange_service import get_exchange_rate

router = APIRouter()

@router.get("/{currency}", response_model=ExchangeRate)
def exchange_info(currency: str):
    result = get_exchange_rate(currency.upper())
    if not result:
        raise HTTPException(status_code=404, detail="Currency not found")
    return result
