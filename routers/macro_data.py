from fastapi import APIRouter, HTTPException
from models.macro_event import MacroEvent
from services.macro_data import get_event_data

router = APIRouter()

@router.get("/{event}/{country}", response_model=list[MacroEvent])
def exchange_info(event: str, country:str):
    result = get_event_data(event,country)
    if not result:
        raise HTTPException(status_code=404, detail="Currency not found")
    return result
