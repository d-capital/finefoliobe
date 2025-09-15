from fastapi import APIRouter, HTTPException
from models.sreener_result import ScreenerResult
from models.screener_filter import ScreenerFilter
from services.screener_service import get_screener

router = APIRouter()

@router.post("/", response_model=list[ScreenerResult])
def screen_stocks(screenerFilters: ScreenerFilter):
    result = get_screener(screenerFilters)
    if not result:
        raise HTTPException(status_code=404, detail=f"Screening failed.")
    return result
